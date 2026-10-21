"""End-to-end run: data -> distributed features -> models -> diagnostics -> unsupervised and
regression features -> semi-supervised -> drift monitoring and retraining -> bandit -> responsible AI.

  python scripts/run_experiments.py               # full run (about 10 minutes on 2 CPU cores)
  python scripts/run_experiments.py --small       # quick smoke run
Writes results/metrics.json, results/model_card.md and a model registry under results/registry/.

Time windows (days): train 0-49 | validation 50-59 | drift starts 60 | monitor 60-74 | final test 75-89
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fraudml.bandit import daily_rewards, simulate  # noqa: E402
from fraudml.data import GeneratorConfig, generate  # noqa: E402
from fraudml.distributed import build_features_distributed  # noqa: E402
from fraudml.features import customer_profiles  # noqa: E402
from fraudml.modeling import (complexity_curve, diagnose, fraud_metrics, gradient_boosting,  # noqa: E402
                              logistic_regression, tune_gradient_boosting)
from fraudml.monitoring import ModelRegistry, RetrainPolicy, drift_report, psi  # noqa: E402
from fraudml.responsible import global_importance, local_explanation, model_card, slice_metrics  # noqa: E402
from fraudml.semi_supervised import labels_known_at, reviewed_sample, self_training  # noqa: E402
from fraudml.unsupervised import BehaviourSegments, ExpectedAmountModel, expected_amount_features  # noqa: E402

OUT = Path("results")


def _holdout_scores(mask, scores, n):
    out = np.full(n, -np.inf)
    out[np.where(mask)[0]] = scores
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true")
    args = ap.parse_args()
    cfg = GeneratorConfig(n_customers=400 if args.small else 2000)
    OUT.mkdir(exist_ok=True)
    R: dict = {}
    t_all = time.time()

    # ---------------------------------------------------------------- data and distributed features
    df = generate(cfg)
    R["data"] = {"rows": len(df), "customers": cfg.n_customers, "days": cfg.days, "drift_day": cfg.drift_day,
                 "fraud_rows": int(df["is_fraud"].sum()), "fraud_rate_pct": round(float(100 * df["is_fraud"].mean()), 3)}
    X, R["feature_pipeline"] = build_features_distributed(df)
    y, amt, day = df["is_fraud"].to_numpy(), df["amount"].to_numpy(), df["day"].to_numpy()
    tr, va = day < 50, (day >= 50) & (day < 60)
    mon, te = (day >= 60) & (day < 75), day >= 75
    print("data", R["data"], R["feature_pipeline"])

    # ---------------------------------------------------------------- supervised baselines + tuning
    lr = logistic_regression(C=0.5).fit(X[tr], y[tr])
    hgb_default = gradient_boosting().fit(X[tr], y[tr])
    hgb, best_params, cv_score = tune_gradient_boosting(X[tr], y[tr], n_iter=6 if args.small else 12)
    R["supervised_validation"] = {
        "logistic_regression_l2": fraud_metrics(y[va], lr.predict_proba(X[va])[:, 1], amt[va]),
        "gradient_boosting_default": fraud_metrics(y[va], hgb_default.predict_proba(X[va])[:, 1], amt[va]),
        "gradient_boosting_tuned": fraud_metrics(y[va], hgb.predict_proba(X[va])[:, 1], amt[va]),
        "tuning": {"best_params": best_params, "cv_pr_auc_time_series": cv_score},
    }
    print("supervised", {k: v.get("pr_auc") for k, v in R["supervised_validation"].items() if isinstance(v, dict)})

    # ---------------------------------------------------------------- bias / variance
    curve = complexity_curve(X[tr], y[tr], X[va], y[va], leaf_nodes=(2, 8, 31, 127) if args.small else (2, 4, 8, 16, 31, 63, 127))
    tuned_train = float(average_precision_score(y[tr], hgb.predict_proba(X[tr])[:, 1]))
    R["bias_variance"] = {"complexity_curve": curve,
                          "diagnosis_smallest_model": diagnose(curve[0]["train_pr_auc"], curve[0]["val_pr_auc"]),
                          "diagnosis_largest_model": diagnose(curve[-1]["train_pr_auc"], curve[-1]["val_pr_auc"]),
                          "diagnosis_tuned_model": diagnose(tuned_train, R["supervised_validation"]["gradient_boosting_tuned"]["pr_auc"]),
                          "tuned_train_pr_auc": round(tuned_train, 4)}
    print("bias/variance", R["bias_variance"]["diagnosis_largest_model"])

    # ---------------------------------------------------------------- PCA + clustering + regression features
    prof_train = customer_profiles(df[tr])
    seg = BehaviourSegments().fit(prof_train)
    true_seg = df[tr].groupby("customer_id")["segment"].first()
    R["unsupervised"] = {**seg.report(), "adjusted_rand_vs_true_segments": seg.agreement(prof_train, true_seg)}
    cust = seg.transform(prof_train).drop(columns="segment").join(prof_train[["log_amount_mean", "online_share"]])
    amt_X = expected_amount_features(df, cust)
    legit_tr = tr & (y == 0)
    reg = ExpectedAmountModel().fit(amt_X[legit_tr], np.log1p(amt[legit_tr]))
    R["regression_expected_amount"] = reg.evaluate(amt_X[va & (y == 0)], np.log1p(amt[va & (y == 0)]))
    X_aug = X.copy()
    X_aug["amount_residual"] = np.log1p(amt) - reg.predict(amt_X)
    X_aug = X_aug.join(amt_X[[c for c in amt_X.columns if c.startswith(("pc", "segment_"))]])
    hgb_aug = gradient_boosting(**best_params).fit(X_aug[tr], y[tr])
    R["feature_ablation_validation_pr_auc"] = {
        "base_features": R["supervised_validation"]["gradient_boosting_tuned"]["pr_auc"],
        "plus_regression_residual_pca_segments": fraud_metrics(y[va], hgb_aug.predict_proba(X_aug[va])[:, 1])["pr_auc"],
    }
    print("unsupervised", R["unsupervised"]["chosen_k"], R["unsupervised"]["adjusted_rand_vs_true_segments"],
          "regression", R["regression_expected_amount"], "ablation", R["feature_ablation_validation_pr_auc"])
    use_aug = R["feature_ablation_validation_pr_auc"]["plus_regression_residual_pca_segments"] > \
        R["feature_ablation_validation_pr_auc"]["base_features"]
    XF = X_aug if use_aug else X
    feature_names = list(XF.columns)

    # ---------------------------------------------------------------- semi-supervised (delayed, partial labels)
    cutoff = 50 * 86400.0
    y_known = labels_known_at(df[tr], cutoff)
    lab = reviewed_sample(int(tr.sum()), y_known, review_rate=0.10)
    sup = gradient_boosting(class_weight=None, max_iter=200).fit(XF[tr][lab], y_known[lab])
    y_semi = np.where(lab, y_known, -1)
    semi = self_training().fit(XF[tr], y_semi)
    R["semi_supervised_validation"] = {
        "labelled_share_of_train": round(float(lab.mean()), 4),
        "fraud_labels_arrived_by_cutoff": f"{int(y_known.sum())}/{int(y[tr].sum())}",
        "supervised_on_labelled_only_pr_auc": fraud_metrics(y[va], sup.predict_proba(XF[va])[:, 1])["pr_auc"],
        "self_training_pr_auc": fraud_metrics(y[va], semi.predict_proba(XF[va])[:, 1])["pr_auc"],
        "pseudo_labelled_rows": int((semi.transduction_ != -1).sum() - lab.sum()),
    }
    print("semi", R["semi_supervised_validation"])

    # ---------------------------------------------------------------- champion, drift, retraining
    reg_dir = OUT / "registry"
    shutil.rmtree(reg_dir, ignore_errors=True)
    registry = ModelRegistry(reg_dir)
    policy = RetrainPolicy()
    trval = day < 60
    champion = gradient_boosting(**best_params).fit(XF[trval], y[trval])
    # champion's expected performance = the same configuration trained on days 0-49, scored on days 50-59
    holdout = (hgb_aug if use_aug else hgb).predict_proba(XF[va])[:, 1]
    rec = registry.register(champion, feature_names, "days 0-59", fraud_metrics(y[va], holdout), stage="champion")
    ref_scores = champion.predict_proba(XF[va])[:, 1]
    weekly = []
    for start in range(60, 90, 7):
        w = (day >= start) & (day < start + 7)
        if not w.any():
            continue
        rep = drift_report(XF[trval], XF[w])
        weekly.append({"days": f"{start}-{min(start + 6, 89)}", "max_feature_psi": rep["max_psi"],
                       "top_drifted": rep["top_features"],
                       "score_psi": round(psi(ref_scores, champion.predict_proba(XF[w])[:, 1]), 4)})
    # analysts review the top 1% of alerts each day; their outcomes are the only fresh labels
    all_scores = champion.predict_proba(XF)[:, 1]

    def top_daily(mask_days, scores, rate=0.01):
        sel = np.zeros(len(df), bool)
        for d in np.unique(day[mask_days]):
            idx = np.where(day == d)[0]
            sel[idx[np.argsort(-scores[idx])[:max(1, int(round(len(idx) * rate)))]]] = True
        return sel

    reviewed_val = top_daily(va, _holdout_scores(va, holdout, len(df)))
    reviewed = top_daily(mon, all_scores)
    prec_val = float(y[reviewed_val].mean())
    prec_mon = float(y[reviewed].mean())
    labelled_frauds_mon = int((mon & (y == 1) & (df["label_ts"].fillna(np.inf).to_numpy() <= 75 * 86400.0)).sum())
    latest = [w for w in weekly if int(w["days"].split("-")[1]) < 75][-1]  # last full week before the day-75 review
    retrain, reasons = policy.should_retrain(latest["max_feature_psi"], latest["score_psi"], prec_val - prec_mon)
    # challengers: retrain on days 0-74 with labels as known on day 75.
    #  (a) chargebacks only: recent fraud is still unlabelled, so it is learned as "legitimate"
    #  (b) plus analyst feedback: alerts the champion raised during days 60-74 (top 1% per day) were
    #      reviewed, so their outcomes are known immediately; post-drift rows get double weight
    up_to = day < 75
    y_cb = labels_known_at(df[up_to], 75 * 86400.0)
    y_fb = np.maximum(y_cb, (reviewed[up_to] & (y[up_to] == 1)).astype(int))
    weights = np.where(day[up_to] >= 60, 2.0, 1.0)
    chall_a = gradient_boosting(**best_params).fit(XF[up_to], y_cb)
    chall_b = gradient_boosting(**best_params).fit(XF[up_to], y_fb, sample_weight=weights)
    champ_te = fraud_metrics(y[te], champion.predict_proba(XF[te])[:, 1], amt[te])
    a_te = fraud_metrics(y[te], chall_a.predict_proba(XF[te])[:, 1], amt[te])
    b_te = fraud_metrics(y[te], chall_b.predict_proba(XF[te])[:, 1], amt[te])
    registry.register(chall_a, feature_names, "days 0-74, chargeback labels only", a_te)
    challenger, chall_te = (chall_b, b_te) if b_te["pr_auc"] >= a_te["pr_auc"] else (chall_a, a_te)
    crec = registry.register(chall_b, feature_names, "days 0-74, chargebacks + analyst feedback, recency-weighted", b_te)
    if challenger is chall_a:
        crec = registry.index["models"][-2]
        crec_version = crec["version"]
    else:
        crec_version = crec.version
    promoted = policy.promote(champ_te["pr_auc"], chall_te["pr_auc"])
    if promoted:
        registry.promote(crec_version, "beat champion on days 75-89 by >= 0.01 PR-AUC")
    final_model = challenger if promoted else champion
    R["drift_and_retraining"] = {
        "weekly_drift": weekly, "champion_pr_auc_validation": rec.metrics["pr_auc"],
        "reviewed_alert_precision": {"validation_days_50_59": round(prec_val, 4), "after_drift_days_60_74": round(prec_mon, 4)},
        "frauds_days_60_74_with_chargeback_by_day_75": f"{labelled_frauds_mon}/{int((mon & (y == 1)).sum())}",
        "retrain_decision_week": latest["days"], "retrain_triggered": retrain, "trigger_reasons": reasons,
        "labels_known_at_retrain": {"fraud_rows_days_0_74": int(y[up_to].sum()), "chargebacks_arrived": int(y_cb.sum()),
                                    "plus_analyst_confirmed": int(y_fb.sum())},
        "final_window_days_75_89": {"champion": champ_te, "challenger_chargebacks_only": a_te,
                                    "challenger_with_analyst_feedback": b_te},
        "challenger_promoted": promoted,
        "registry_events": registry.index["events"],
    }
    R["champion_final"] = {"version": registry.champion()["version"], "model": "HistGradientBoosting (tuned)",
                           "trained_on": registry.champion()["trained_on"], "metrics": chall_te if promoted else champ_te}
    print("drift", weekly[0], "retrain", retrain, reasons, "promoted", promoted, champ_te["pr_auc"], "->", chall_te["pr_auc"])

    # ---------------------------------------------------------------- RL: contextual bandit for review budget
    frame = pd.DataFrame({"day": day[te], "segment": df["segment"].astype(str).to_numpy()[te],
                          "score": final_model.predict_proba(XF[te])[:, 1], "is_fraud": y[te], "amount": amt[te]})
    R["bandit_review_budget"] = simulate(daily_rewards(frame, review_cost=5.0), rounds=2000)
    print("bandit", {k: v for k, v in R["bandit_review_budget"].items() if k != "curve"})

    # ---------------------------------------------------------------- responsible AI
    s_te = final_model.predict_proba(XF[te])[:, 1]
    thr = float(np.quantile(s_te, 1 - 0.005))
    flagged_row = int(np.argmax(s_te))
    R["responsible_ai"] = {
        "global_importance": global_importance(final_model, XF[te], y[te], n_repeats=2 if args.small else 3),
        "example_explanation": local_explanation(final_model, XF[te].reset_index(drop=True), flagged_row),
        "slices_segment": slice_metrics(y[te], s_te, df["segment"][te], thr),
        "slices_channel": slice_metrics(y[te], s_te, np.where(df["is_online"][te], "online", "in_person"), thr),
    }
    R["runtime_seconds"] = round(time.time() - t_all, 1)
    (OUT / "metrics.json").write_text(json.dumps(R, indent=2, default=str))
    (OUT / "model_card.md").write_text(model_card(R))
    print(f"done in {R['runtime_seconds']} s -> results/metrics.json, results/model_card.md")


if __name__ == "__main__":
    main()
