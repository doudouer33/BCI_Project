"""Train and evaluate CSP + LDA and CSP + SVM on each subject."""

import argparse
import json
import platform
import shlex
import sys
from importlib.metadata import version
from itertools import product
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from src.load_data import ROOT, get_split_indices, load_config, load_subject
from src.plot_results import plot_baseline_accuracy, plot_baseline_confusion


def make_pipeline(model, params, seed):
    csp = CSP(n_components=params["n_components"], log=True)
    if model == "CSP_LDA":
        return Pipeline([("csp", csp), ("lda", LinearDiscriminantAnalysis())])
    return Pipeline([
        ("csp", csp),
        ("scale", StandardScaler()),
        ("svm", SVC(C=params["C"], kernel=params["kernel"], gamma="scale", random_state=seed)),
    ])


def select_params(model, candidates, train, val, labels, seed):
    X_val, y_val = val
    scores = []
    for params in candidates:
        pipeline = make_pipeline(model, params, seed).fit(*train)
        score = f1_score(y_val, pipeline.predict(X_val), labels=labels, average="macro", zero_division=0)
        scores.append({"params": params, "val_macro_f1": float(score)})
    # max keeps the first candidate on a tie, following the order in the config.
    return max(scores, key=lambda result: result["val_macro_f1"]), scores


def run_subject(subject, config, config_id, candidates):
    X, y, metadata = load_subject(subject, config)
    train_idx, val_idx, test_idx = get_split_indices(metadata, config)
    seed = config["random_seed"]
    labels = sorted(config["label_mapping"].values())
    print(f"Subject {subject}: {X.shape}, train={len(train_idx)}, "
          f"val={len(val_idx)}, test={len(test_idx)}", flush=True)

    train = X[train_idx], y[train_idx]
    val = X[val_idx], y[val_idx]
    selections = {
        model: select_params(model, grid, train, val, labels, seed)
        for model, grid in candidates.items()
    }

    development_idx = np.sort(np.concatenate((train_idx, val_idx)))
    development = X[development_idx], y[development_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    trials = metadata.iloc[test_idx][["trial_id"]].reset_index(drop=True)
    for model, (best, scores) in selections.items():
        params = best["params"]
        pipeline = make_pipeline(model, params, seed).fit(*development)
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, labels=labels, average="macro", zero_division=0)
        matrix = confusion_matrix(y_test, y_pred, labels=labels)
        row = {
            "subject": subject, "model": model,
            "train_session": config["train_session"], "test_session": config["test_session"],
            "accuracy": accuracy, "macro_f1": macro_f1, "seed": seed, "config_id": config_id,
            **params,
        }
        log = {
            "subject": subject, "model": model,
            "selected": best, "validation_scores": scores, "confusion_matrix": matrix.tolist(),
        }
        print(f"  {model}: accuracy={accuracy:.4f}, macro_f1={macro_f1:.4f}, {params}", flush=True)
        yield row, trials.assign(y_true=y_test, y_pred=y_pred), log


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--subjects", type=int, nargs="+", help="Default: all configured subjects")
    parser.add_argument("--data-dir", type=Path, help="Override the MOABB data cache path")
    parser.add_argument("--output-dir", type=Path, help="Default: results for all subjects")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.data_dir is not None:
        config["data_dir"] = str(args.data_dir.expanduser().resolve())
    subjects = args.subjects or config["subjects"]
    if len(set(subjects)) != len(subjects) or not set(subjects) <= set(config["subjects"]):
        parser.error("--subjects must contain distinct IDs from the experiment config")
    output = args.output_dir
    if output is None:
        output = ROOT / "results"
        if set(subjects) != set(config["subjects"]):
            output /= "subject_" + "_".join(map(str, subjects))

    csp = config["csp"]
    candidates = {
        "CSP_LDA": [{"n_components": n} for n in csp["n_components"]],
        "CSP_SVM": [
            {"n_components": n, "kernel": kernel, "C": c}
            for n, kernel, c in product(csp["n_components"], csp["svm_kernels"], csp["svm_c"])
        ],
    }
    rows, predictions, records = [], {}, []
    with threadpool_limits(limits=1, user_api="blas"), mne.use_log_level("WARNING"):
        for subject in subjects:
            for row, trials, log in run_subject(subject, config, str(args.config), candidates):
                rows.append(row)
                predictions[f"{row['model']}_s{subject}.csv"] = trials
                records.append(log)

    figure_dir = output / "figures"
    prediction_dir = output / "predictions"
    log_dir = output / "logs"
    for directory in (figure_dir, prediction_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)
    for filename, trials in predictions.items():
        trials.to_csv(prediction_dir / filename, index=False)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "metrics.csv", index=False)
    summary = metrics.groupby("model", sort=False).agg(
        n_subjects=("subject", "size"),
        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", lambda values: values.std(ddof=0)),
        macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", lambda values: values.std(ddof=0)),
    )
    summary.to_csv(output / "csp_summary.csv")
    class_order = sorted(config["label_mapping"], key=config["label_mapping"].get)
    log = {
        "command": shlex.join(sys.orig_argv), "config": config,
        "subjects": subjects, "confusion_matrix_class_order": class_order,
        "versions": {"python": platform.python_version(), **{
            package: version(package) for package in (
                "moabb", "mne", "numpy", "scipy", "pandas", "scikit-learn", "matplotlib", "pyyaml", "threadpoolctl",
            )
        }},
        "results": records,
    }
    (log_dir / "csp_run.json").write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")
    plot_baseline_accuracy(metrics, figure_dir)
    example = {row["model"]: row["confusion_matrix"] for row in records if row["subject"] == subjects[0]}
    plot_baseline_confusion(subjects[0], example, class_order, figure_dir)
    print(f"\n{summary.to_string(float_format='{:.4f}'.format)}\n\nSaved to {output}")


if __name__ == "__main__":
    main()
