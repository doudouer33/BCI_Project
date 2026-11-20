"""Train and evaluate Braindecode EEGNet independently for each subject."""

import argparse
import json
import os
import platform
import random
import shlex
import sys
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import mne
import numpy as np
import pandas as pd
import torch
from braindecode.models import EEGNet
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn

from src.load_data import ROOT, get_split_indices, load_config, load_subject
from src.plot_results import plot_eegnet_training, plot_model_comparison, plot_test_confusions
from src.torch_training import VOLTS_TO_MICROVOLTS, fit, make_loader, run_epoch


def run_subject(subject, config, config_id, device, output, plot_example):
    started = perf_counter()
    X, y, metadata = load_subject(subject, config)
    train_idx, val_idx, test_idx = get_split_indices(metadata, config)
    counts = tuple(map(len, (train_idx, val_idx, test_idx)))
    n_trials, n_chans, n_times = X.shape
    sfreq = metadata.attrs["sfreq"]
    if (n_chans, n_times, sfreq) != (22, 1000, 250) or counts != (230, 58, 288):
        raise ValueError(f"Subject {subject}: unexpected signal shape {X.shape}, {sfreq:g} Hz or splits {counts}")
    if metadata.attrs["units"] != "V":
        raise ValueError("Expected the shared EEG loader to return signals in volts")
    print(f"Subject {subject}: {X.shape}, train={counts[0]}, val={counts[1]}, test={counts[2]}", flush=True)

    seed = config["random_seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    settings = config["eegnet"]
    batch_size = settings["batch_size"]
    class_order = sorted(config["label_mapping"], key=config["label_mapping"].get)
    labels = [config["label_mapping"][name] for name in class_order]
    model_args = {
        "n_chans": n_chans, "n_times": n_times, "sfreq": sfreq, "n_outputs": len(labels),
        "final_layer_with_constraint": True,
    }
    model = EEGNet(**model_args).to(device)
    train = make_loader(X, y, train_idx, batch_size, shuffle_seed=seed)
    val = make_loader(X, y, val_idx, batch_size)
    name = f"EEGNet_s{subject}"
    checkpoint = output / "models" / f"{name}.pt"
    history, selection = fit(model, train, val, settings, device, checkpoint)

    # Construct the test loader only after validation has selected and restored the weights.
    test = make_loader(X, y, test_idx, batch_size)
    _, _, y_pred = run_epoch(model, test, nn.CrossEntropyLoss(), device)
    y_true = y[test_idx]
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    trials = metadata.iloc[test_idx][["trial_id"]].assign(y_true=y_true, y_pred=y_pred)
    trials.to_csv(output / "predictions" / f"{name}.csv", index=False)
    history.to_csv(output / "logs" / f"{name}_epochs.csv", index=False)
    if plot_example:
        figure_dir = output / "figures"
        plot_eegnet_training(subject, history, selection["best_epoch"], figure_dir)
        plot_test_confusions(subject, {"EEGNet": matrix}, class_order,
                             figure_dir / f"eegnet_confusion_s{subject}.png")

    row = {
        "subject": subject, "model": "EEGNet", "train_session": config["train_session"],
        "test_session": config["test_session"], "accuracy": accuracy, "macro_f1": macro_f1,
        "seed": seed, "config_id": config_id,
    }
    log = {
        **row, "input_shape": [n_trials, n_chans, n_times], "model_args": model_args,
        "n_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "split_counts": dict(zip(("train", "val", "test"), counts)),
        **selection, "checkpoint": str(checkpoint.relative_to(output)),
        "epoch_log": f"logs/{name}_epochs.csv", "confusion_matrix": matrix.tolist(),
        "elapsed_seconds": round(perf_counter() - started, 2),
    }
    print(f"  Best epoch {log['best_epoch']}/{log['epochs_run']}, early stopping={log['stopped_early']}; "
          f"test accuracy={accuracy:.4f}, macro-F1={macro_f1:.4f}", flush=True)
    return row, log


def compare_with_csp(metrics, config, output):
    baseline_dir = ROOT / "results"
    baseline_csv = baseline_dir / "metrics.csv"
    baseline_log = baseline_dir / "logs" / "csp_run.json"
    if not baseline_csv.is_file() or not baseline_log.is_file():
        print("CSP reference results unavailable; saved EEGNet metrics only.")
        return metrics

    csp_config = json.loads(baseline_log.read_text(encoding="utf-8"))["config"]
    protocol_keys = (
        "dataset", "label_mapping", "sfreq", "fmin", "fmax", "filter", "cue_offset_from_trial_start_s",
        "tmin", "tmax", "train_session", "test_session", "val_ratio", "random_seed",
    )
    if any(config[key] != csp_config[key] for key in protocol_keys):
        print("CSP reference protocol differs; saved EEGNet metrics only.")
        return metrics

    csp = pd.read_csv(baseline_csv)
    subjects = set(metrics["subject"])
    keys = ["subject", "train_session", "test_session", "seed"]
    expected = metrics[keys].sort_values("subject").reset_index(drop=True)
    for model in ("CSP_LDA", "CSP_SVM"):
        rows = csp.loc[csp["model"].eq(model) & csp["subject"].isin(subjects), keys]
        if not rows.sort_values("subject").reset_index(drop=True).equals(expected):
            print("CSP reference does not cover the EEGNet test protocol; saved EEGNet metrics only.")
            return metrics

    csp = csp.loc[csp["model"].isin(["CSP_LDA", "CSP_SVM"]) & csp["subject"].isin(subjects), metrics.columns]
    comparison = pd.concat([csp, metrics], ignore_index=True).sort_values("subject", kind="stable")
    comparison.to_csv(output / "model_metrics.csv", index=False)
    plot_model_comparison(comparison, output / "figures")
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--subjects", nargs="+", type=int, help="Default: all configured subjects")
    parser.add_argument("--data-dir", type=Path, help="Override the MOABB data cache path")
    parser.add_argument("--output-dir", type=Path, help="Default: results, with a subdirectory for subsets")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.data_dir is not None:
        config["data_dir"] = str(args.data_dir.expanduser().resolve())
    subjects = args.subjects or config["subjects"]
    if len(set(subjects)) != len(subjects) or not set(subjects) <= set(config["subjects"]):
        parser.error("--subjects must contain distinct IDs from the experiment config")
    settings = config["eegnet"]
    if min(settings["batch_size"], settings["max_epochs"], settings["patience"], settings["learning_rate"]) <= 0:
        parser.error("EEGNet training settings must be positive")

    all_subjects = set(subjects) == set(config["subjects"])
    output = args.output_dir or ROOT / "results"
    if args.output_dir is None and not all_subjects:
        output /= "subject_" + "_".join(map(str, subjects))
    for directory in ("figures", "logs", "models", "predictions"):
        (output / directory).mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable to PyTorch")
    device = torch.device(device_name)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    print(f"Device: {device}", flush=True)

    rows, records = [], []
    example = 1 if 1 in subjects else subjects[0]
    with mne.use_log_level("WARNING"):
        for subject in subjects:
            row, record = run_subject(subject, config, str(args.config), device, output, subject == example)
            rows.append(row)
            records.append(record)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "eegnet_metrics.csv", index=False)
    comparison = compare_with_csp(metrics, config, output) if all_subjects else metrics
    summary = comparison.groupby("model", sort=False).agg(
        n_subjects=("subject", "size"),
        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", lambda scores: scores.std(ddof=0)),
        macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", lambda scores: scores.std(ddof=0)),
    )
    summary.to_csv(output / "model_summary.csv")
    log = {
        "command": shlex.join(sys.orig_argv), "config": config, "subjects": subjects,
        "versions": {"python": platform.python_version(), **{
            package: version(package) for package in (
                "torch", "torchaudio", "braindecode", "moabb", "mne", "numpy", "scipy",
                "pandas", "scikit-learn", "matplotlib", "pyyaml",
            )
        }},
        "device": device.type,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda": torch.version.cuda if device.type == "cuda" else None,
        "cudnn": torch.backends.cudnn.version() if device.type == "cuda" else None,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "torch_cpu_threads": torch.get_num_threads(),
        "volts_to_microvolts": VOLTS_TO_MICROVOLTS,
        "optimizer": "Adam", "loss": "CrossEntropyLoss",
        "checkpoint_selection": "lowest validation loss, earliest epoch on ties",
        "refit_on_train_plus_val": False,
        "confusion_matrix_class_order": sorted(config["label_mapping"], key=config["label_mapping"].get),
        "results": records,
    }
    (output / "logs" / "eegnet_run.json").write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")
    print(f"\n{summary.to_string(float_format='{:.4f}'.format)}\n\nSaved to {output}")


if __name__ == "__main__":
    main()
