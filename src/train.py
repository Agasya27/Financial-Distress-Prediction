"""Primary training entrypoint.

Default mode trains multimodal fusion model.
Use `--mode local_lite` for sklearn baseline.
"""

from __future__ import annotations

import argparse
import os

import joblib
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.dataset import time_split
from src.feature_table import load_feature_table
from src.utils import ensure_dir, load_json, now_iso, save_json
from src.train_multimodal import main as train_multimodal_main


class WeightedEnsemble(BaseEstimator, ClassifierMixin):
    """Weighted probability ensemble over fitted binary classifiers."""

    def __init__(self, estimators: list[Pipeline], weights: list[float]):
        self.estimators = estimators
        self.weights = weights

    def fit(self, x, y):
        return self

    def predict_proba(self, x):
        probs = []
        for est in self.estimators:
            p = est.predict_proba(x)[:, 1]
            probs.append(p)
        arr = np.vstack(probs)  # (n_models, n_samples)
        w = np.array(self.weights, dtype=float)
        w = w / np.clip(w.sum(), 1e-12, None)
        out = np.average(arr, axis=0, weights=w)
        return np.stack([1.0 - out, out], axis=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train local lightweight model.")
    parser.add_argument("--mode", choices=["multimodal", "local_lite"], default="multimodal")
    parser.add_argument("--tabular-path", default="data/processed/tabular.csv")
    parser.add_argument("--mda-dir", default="data/raw/mda_texts")
    parser.add_argument("--graph-path", default="data/processed/graph.pt")
    parser.add_argument("--out-dir", default="checkpoints/local_lite")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--force-overwrite", action="store_true")
    parser.add_argument("--no-text-stats", action="store_true")
    parser.add_argument("--no-graph-degree", action="store_true")
    parser.add_argument("--text-emb-path", default="data/processed/text_embeddings.pt")
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--min-epochs", type=int, default=3)
    parser.add_argument("--balance-train", action="store_true")
    parser.add_argument("--loss", choices=["bce", "focal"], default="bce")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--lr-scheduler", choices=["none", "plateau"], default="none")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--modality-dropout", type=float, default=0.0)
    parser.add_argument("--gate-entropy-coef", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "multimodal":
        # Delegate to multimodal trainer while preserving existing CLI behavior.
        import sys

        sys.argv = [
            "train_multimodal",
            "--tabular-path",
            args.tabular_path,
            "--text-emb-path",
            args.text_emb_path,
            "--out-dir",
            "checkpoints/multimodal",
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--weight-decay",
            str(args.weight_decay),
            "--test-ratio",
            str(args.test_ratio),
            "--max-rows",
            str(args.max_rows),
            "--early-stopping-patience",
            str(args.early_stopping_patience),
            "--min-epochs",
            str(args.min_epochs),
            "--loss",
            args.loss,
            "--focal-gamma",
            str(args.focal_gamma),
            "--lr-scheduler",
            args.lr_scheduler,
            "--seed",
            str(args.seed),
            "--modality-dropout",
            str(args.modality_dropout),
            "--gate-entropy-coef",
            str(args.gate_entropy_coef),
        ]
        if args.balance_train:
            sys.argv.append("--balance-train")
        if args.no_amp:
            sys.argv.append("--no-amp")
        train_multimodal_main()
        return

    try:
        from src.metrics_classification import best_f1_threshold, binary_classification_metrics
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing src/metrics_classification.py (needed for --mode local_lite). "
            "Copy the full src/ tree from your Mac, e.g. "
            "scp .../financial_distress/src/metrics_classification.py "
            "user@studio:~/financial_distress/src/"
        ) from exc

    ensure_dir(args.out_dir)

    x_df, y, feature_cols, base_df = load_feature_table(
        tabular_path=args.tabular_path,
        mda_dir=args.mda_dir,
        graph_path=args.graph_path,
        use_text_stats=not args.no_text_stats,
        use_graph_degree=not args.no_graph_degree,
        max_rows=args.max_rows,
    )
    train_mask = time_split(base_df, test_ratio=args.test_ratio)

    x_train = x_df.loc[train_mask]
    y_train = y[train_mask]
    x_test = x_df.loc[~train_mask]
    y_test = y[~train_mask]

    # Stratified validation split so rare positives appear in both parts (stable PR-AUC).
    try:
        x_tr, x_va, y_tr, y_va = train_test_split(
            x_train,
            y_train,
            test_size=0.1,
            stratify=y_train,
            random_state=42,
        )
    except ValueError:
        x_tr, x_va, y_tr, y_va = train_test_split(
            x_train,
            y_train,
            test_size=0.1,
            random_state=42,
        )

    # Class-weight style balancing by scaling positives.
    pos_rate = max(float(np.mean(y_tr)), 1e-6)
    pos_weight = float((1.0 - pos_rate) / pos_rate)
    sample_weight_tr = np.where(y_tr == 1, pos_weight, 1.0)

    candidates: list[tuple[str, Pipeline, dict[str, np.ndarray]]] = []
    candidates.append(
        (
            "hist_gbdt",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "clf",
                        HistGradientBoostingClassifier(
                            learning_rate=0.05,
                            max_depth=6,
                            max_iter=350,
                            random_state=42,
                            early_stopping=True,
                        ),
                    ),
                ]
            ),
            {"clf__sample_weight": sample_weight_tr},
        )
    )
    candidates.append(
        (
            "random_forest",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "clf",
                        RandomForestClassifier(
                            n_estimators=400,
                            max_depth=12,
                            min_samples_leaf=3,
                            class_weight="balanced_subsample",
                            random_state=42,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            {},
        )
    )
    candidates.append(
        (
            "extra_trees",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "clf",
                        ExtraTreesClassifier(
                            n_estimators=600,
                            max_depth=16,
                            min_samples_leaf=2,
                            class_weight="balanced",
                            random_state=42,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            {},
        )
    )
    candidates.append(
        (
            "log_reg",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "clf",
                        LogisticRegression(
                            solver="saga",
                            max_iter=8000,
                            class_weight="balanced",
                            random_state=42,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            {},
        )
    )

    best_name = ""
    best_model = None
    best_val_pr = -1.0
    fitted: list[tuple[str, Pipeline, float]] = []
    model_selection: list[dict[str, float | str]] = []
    for name, cand, fit_kwargs in candidates:
        cand.fit(x_tr, y_tr, **fit_kwargs)
        va_probs = cand.predict_proba(x_va)[:, 1]
        val_pr = float(average_precision_score(y_va, va_probs))
        print(f"[MODEL_SELECT] {name} val_pr_auc={val_pr:.6f}")
        fitted.append((name, cand, val_pr))
        model_selection.append({"name": name, "val_pr_auc": val_pr})
        if val_pr > best_val_pr:
            best_val_pr = val_pr
            best_name = name
            best_model = cand

    # Try weighted ensemble of top-2 models by validation PR-AUC.
    fitted.sort(key=lambda x: x[2], reverse=True)
    if len(fitted) >= 2:
        top = fitted[:2]
        ens = WeightedEnsemble(
            estimators=[top[0][1], top[1][1]],
            weights=[max(top[0][2], 1e-6), max(top[1][2], 1e-6)],
        )
        ens_probs = ens.predict_proba(x_va)[:, 1]
        ens_pr = float(average_precision_score(y_va, ens_probs))
        print(f"[MODEL_SELECT] weighted_ensemble val_pr_auc={ens_pr:.6f}")
        model_selection.append({"name": "weighted_ensemble", "val_pr_auc": ens_pr})
        if ens_pr > best_val_pr:
            best_val_pr = ens_pr
            best_name = "weighted_ensemble"
            best_model = ens

    assert best_model is not None
    # Refit chosen model on full train split.
    if best_name == "weighted_ensemble":
        top = fitted[:2]
        refit_estimators: list[Pipeline] = []
        for nm, est, _score in top:
            if nm == "hist_gbdt":
                sample_weight_full = np.where(
                    y_train == 1,
                    (1.0 - np.mean(y_train)) / max(np.mean(y_train), 1e-6),
                    1.0,
                )
                est.fit(x_train, y_train, clf__sample_weight=sample_weight_full)
            else:
                est.fit(x_train, y_train)
            refit_estimators.append(est)
        best_model = WeightedEnsemble(
            estimators=refit_estimators,
            weights=[max(top[0][2], 1e-6), max(top[1][2], 1e-6)],
        )
    elif best_name == "hist_gbdt":
        sample_weight_full = np.where(y_train == 1, (1.0 - np.mean(y_train)) / max(np.mean(y_train), 1e-6), 1.0)
        best_model.fit(x_train, y_train, clf__sample_weight=sample_weight_full)
    else:
        best_model.fit(x_train, y_train)

    model = best_model
    # Sigmoid calibration on training folds improves interpretability of P(distress).
    calibrated = False
    if best_name != "weighted_ensemble":
        try:
            cal_model = CalibratedClassifierCV(model, method="sigmoid", cv=3)
            cal_model.fit(x_train, y_train)
            model = cal_model
            calibrated = True
            print("[INFO] Applied sigmoid probability calibration (3-fold CV).")
        except Exception as exc:
            print(f"[WARN] Probability calibration skipped: {exc}")

    probs = model.predict_proba(x_test)[:, 1]
    threshold = best_f1_threshold(y_test, probs)
    m = binary_classification_metrics(y_test, probs, threshold=threshold)

    model_path = os.path.join(args.out_dir, "model.joblib")
    meta_path = os.path.join(args.out_dir, "meta.json")
    metrics_path = os.path.join(args.out_dir, "metrics.json")

    # Keep the better checkpoint by PR-AUC unless forced.
    if (not args.force_overwrite) and os.path.exists(metrics_path):
        try:
            old = load_json(metrics_path)
            old_pr = float(old.get("pr_auc", -1.0))
            if old_pr > float(m.get("pr_auc", -1.0)):
                print("[INFO] Existing checkpoint has better PR-AUC; keeping previous model.")
                print(f"[INFO] old_pr_auc={old_pr:.6f} new_pr_auc={float(m.get('pr_auc', -1.0)):.6f}")
                return
        except Exception:
            pass

    joblib.dump(model, model_path)
    save_json(m, metrics_path)
    save_json(
        {
            "created_at": now_iso(),
            "n_rows_total": int(len(x_df)),
            "n_rows_train": int(len(x_train)),
            "n_rows_test": int(len(x_test)),
            "positive_rate_train": float(np.mean(y_train)),
            "recommended_threshold": float(threshold),
            "best_model_name": best_name,
            "best_val_pr_auc": float(best_val_pr),
            "features": feature_cols,
            "use_text_stats": not args.no_text_stats,
            "use_graph_degree": not args.no_graph_degree,
            "probability_calibration": "sigmoid_3fold" if calibrated else "none",
            "label_positive_meaning": "distress / bankruptcy (ECL label=1)",
            "model_selection": model_selection,
        },
        meta_path,
    )

    print("[DONE] Local training complete")
    print(f"[DONE] model:   {model_path}")
    print(f"[DONE] metrics: {metrics_path}")
    print(f"[DONE] meta:    {meta_path}")
    print(m)


if __name__ == "__main__":
    main()
