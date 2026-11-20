"""Train and evaluate Light Transformer on the fixed BNCI2014_001 splits."""

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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn

from src.load_data import ROOT, get_split_indices, load_config, load_subject
from src.models.light_transformer import LightTransformer
from src.plot_results import plot_model_comparison, plot_test_confusions, plot_training_curves
from src.torch_training import VOLTS_TO_MICROVOLTS, fit, make_loader, run_epoch


RESULTS = ROOT / "results"
MODEL = "LightTransformer"
MODELS = ("CSP_LDA", "CSP_SVM", "EEGNet", MODEL)
PROTOCOL_KEYS = (
    "dataset", "label_mapping", "sfreq", "fmin", "fmax", "filter", "cue_offset_from_trial_start_s",
    "tmin", "tmax", "train_session", "test_session", "val_ratio", "random_seed",
)
TRAINING_KEYS = ("batch_size", "learning_rate", "max_epochs", "patience")


def read_log(filename):
    return json.loads((RESULTS / "logs" / filename).read_text(encoding="utf-8"))


def verify_splits(y, metadata, indices, manifest):
    for split, split_idx in zip(("train", "val", "test"), indices):
        saved = manifest.loc[manifest["split"].eq(split)]
        if (
            not np.array_equal(split_idx, saved["index"])
            or not np.array_equal(metadata.iloc[split_idx]["trial_id"], saved["trial_id"])
            or not np.array_equal(y[split_idx], saved["label"])
        ):
            raise ValueError(f"{split} trials differ from the Stage A split manifest")


def run_subject(subject, config, config_id, manifest, device, hardware, output, plot_example):
    started = perf_counter()
    X, y, metadata = load_subject(subject, config)
    indices = get_split_indices(metadata, config)
    counts = tuple(map(len, indices))
    n_trials, n_chans, n_times = X.shape
    sfreq, units = metadata.attrs["sfreq"], metadata.attrs["units"]
    if (n_chans, n_times, sfreq, units) != (22, 1000, 250, "V") or counts != (230, 58, 288):
        raise ValueError(f"Subject {subject}: unexpected EEG shape {X.shape}, {sfreq:g} Hz, {units}, splits {counts}")
    verify_splits(y, metadata, indices, manifest)
    train_idx, val_idx, test_idx = indices
    print(f"Subject {subject}: {X.shape}, train={counts[0]}, val={counts[1]}, test={counts[2]}", flush=True)

    seed = config["random_seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    settings = config["transformer"]
    batch_size = settings["batch_size"]
    class_order = sorted(config["label_mapping"], key=config["label_mapping"].get)
    labels = [config["label_mapping"][name] for name in class_order]
    model_args = {
        "n_chans": n_chans, "n_times": n_times, "n_outputs": len(labels),
        **{key: value for key, value in settings.items() if key not in TRAINING_KEYS},
    }
    model = LightTransformer(**model_args).to(device)
    train = make_loader(X, y, train_idx, batch_size, shuffle_seed=seed)
    val = make_loader(X, y, val_idx, batch_size)
    name = f"{MODEL}_s{subject}"
    checkpoint = output / "models" / f"{name}.pt"

    train_started = perf_counter()
    history, selection = fit(model, train, val, settings, device, checkpoint)
    train_seconds = round(perf_counter() - train_started, 2)

    # Build the test loader only after the best validation checkpoint is restored.
    test = make_loader(X, y, test_idx, batch_size)
    _, _, y_pred = run_epoch(model, test, nn.CrossEntropyLoss(), device)
    y_true = y[test_idx]
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    prediction_path = f"predictions/{name}.csv"
    epoch_path = f"logs/{name}_epochs.csv"
    trials = metadata.iloc[test_idx][["trial_id"]].assign(y_true=y_true, y_pred=y_pred)
    trials.to_csv(output / prediction_path, index=False)
    history.to_csv(output / epoch_path, index=False)
    if plot_example:
        plot_training_curves(subject, "Light Transformer", history, selection["best_epoch"],
                             output / "figures" / f"transformer_training_s{subject}.png")
        plot_test_confusions(subject, {"Light Transformer": matrix}, class_order,
                             output / "figures" / f"transformer_confusion_s{subject}.png")

    row = {
        "subject": subject, "model": MODEL, "train_session": config["train_session"],
        "test_session": config["test_session"], "accuracy": accuracy, "macro_f1": macro_f1,
        "seed": seed, "config_id": config_id,
        "best_epoch": selection["best_epoch"], "epochs_run": selection["epochs_run"],
        "n_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "train_seconds": train_seconds, "device": device.type, "hardware": hardware,
        "elapsed_seconds": round(perf_counter() - started, 2),
    }
    record = {
        **row, **selection, "input_shape": [n_trials, n_chans, n_times], "model_args": model_args,
        "tokens_per_trial": model.n_tokens, "split_counts": dict(zip(("train", "val", "test"), counts)),
        "checkpoint": str(checkpoint.relative_to(output)), "epoch_log": epoch_path,
        "predictions": prediction_path, "confusion_matrix": matrix.tolist(),
    }
    print(f"  Best epoch {row['best_epoch']}/{row['epochs_run']}; train time={train_seconds:.2f}s; "
          f"test accuracy={accuracy:.4f}, macro-F1={macro_f1:.4f}", flush=True)
    return row, record


def summarize(metrics):
    return metrics.groupby("model", sort=False).agg(
        n_subjects=("subject", "size"),
        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", lambda values: values.std(ddof=0)),
        macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", lambda values: values.std(ddof=0)),
    )


def save_comparison(metrics, baseline, eegnet_log, output, plot_all_subjects):
    subjects = set(metrics["subject"])
    baseline = baseline.loc[baseline["subject"].isin(subjects) & baseline["model"].isin(MODELS[:-1])]
    keys = ["subject", "train_session", "test_session", "seed"]
    expected = metrics[keys].sort_values("subject").reset_index(drop=True)
    for model in MODELS[:-1]:
        reference = baseline.loc[baseline["model"].eq(model), keys]
        if not reference.sort_values("subject").reset_index(drop=True).equals(expected):
            raise ValueError(f"{model} reference rows do not match the evaluated subjects and protocol")
    columns = ["subject", "model", *keys[1:], "accuracy", "macro_f1", "config_id"]
    comparison = pd.concat([baseline[columns], metrics[columns]], ignore_index=True)
    comparison.sort_values("subject", kind="stable").to_csv(output / "four_model_metrics.csv", index=False)
    summarize(comparison).to_csv(output / "four_model_summary.csv")
    if plot_all_subjects:
        plot_model_comparison(comparison, output / "figures", MODELS, prefix="four_model")

    eegnet = pd.DataFrame(eegnet_log["results"])
    eegnet = eegnet.loc[eegnet["subject"].isin(subjects)].assign(
        hardware=eegnet_log["gpu"] or eegnet_log["device"], train_seconds=np.nan,
    )
    runtime = pd.concat([eegnet, metrics], ignore_index=True).groupby(["model", "hardware"], sort=False).agg(
        n_subjects=("subject", "size"), n_parameters=("n_parameters", "first"),
        mean_epochs=("epochs_run", "mean"), mean_train_seconds=("train_seconds", "mean"),
        mean_elapsed_seconds=("elapsed_seconds", "mean"), total_elapsed_seconds=("elapsed_seconds", "sum"),
    )
    runtime.to_csv(output / "transformer_runtime.csv", float_format="%.2f")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--subjects", type=int, nargs="+", help="Default: all configured subjects")
    parser.add_argument("--data-dir", type=Path, help="Override the MOABB data cache path")
    parser.add_argument("--output-dir", type=Path, help="Default: results/transformer, with subdirectories for subsets")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.data_dir is not None:
        config["data_dir"] = str(args.data_dir.expanduser().resolve())
    subjects = sorted(args.subjects or config["subjects"])
    if len(subjects) != len(set(subjects)) or not set(subjects) <= set(config["subjects"]):
        parser.error("--subjects must contain distinct IDs from the experiment config")
    if not all(np.isfinite(config["transformer"][key]) and config["transformer"][key] > 0
               for key in TRAINING_KEYS):
        parser.error("Transformer training settings must be finite and positive")

    eegnet_log = read_log("eegnet_run.json")
    csp_config = read_log("csp_run.json")["config"]
    if any(config[key] != eegnet_log["config"][key] or config[key] != csp_config[key] for key in PROTOCOL_KEYS):
        parser.error("The data protocol must match the fixed CSP and EEGNet reference runs")
    baseline = pd.read_csv(RESULTS / "model_metrics.csv")
    manifest = pd.read_csv(RESULTS / "trial_splits.csv")
    all_subjects = subjects == list(range(1, 10))
    output = args.output_dir or RESULTS / "transformer"
    if args.output_dir is None and not all_subjects:
        output /= "subject_" + "_".join(map(str, subjects))

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable to PyTorch")
    device = torch.device(device_name)
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    hardware = gpu or platform.processor() or platform.machine()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    print(f"Device: {device} ({hardware})", flush=True)

    for directory in ("figures", "logs", "models", "predictions"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    rows, records = [], []
    with mne.use_log_level("WARNING"):
        for subject in subjects:
            row, record = run_subject(
                subject, config, str(args.config), manifest.loc[manifest["subject"].eq(subject)],
                device, hardware, output, subject == subjects[0],
            )
            rows.append(row)
            records.append(record)

    metrics = pd.DataFrame(rows)
    summary = summarize(metrics)
    metrics.to_csv(output / "transformer_metrics.csv", index=False)
    summary.to_csv(output / "transformer_summary.csv")
    log = {
        "command": shlex.join(sys.orig_argv), "config": config, "subjects": subjects,
        "split_manifest": "results/trial_splits.csv", "baseline_metrics": "results/model_metrics.csv",
        "eegnet_run": "results/logs/eegnet_run.json",
        "versions": {"python": platform.python_version(), **{
            package: version(package) for package in (
                "torch", "moabb", "mne", "numpy", "scipy", "pandas", "scikit-learn", "matplotlib", "pyyaml",
            )
        }},
        "platform": platform.platform(), "device": device.type, "hardware": hardware,
        "gpu": gpu, "cuda": torch.version.cuda if gpu else None,
        "cudnn": torch.backends.cudnn.version() if gpu else None,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "torch_cpu_threads": torch.get_num_threads(), "volts_to_microvolts": VOLTS_TO_MICROVOLTS,
        "optimizer": "Adam", "loss": "CrossEntropyLoss",
        "checkpoint_selection": "lowest validation loss, earliest epoch on ties",
        "refit_on_train_plus_val": False,
        "train_seconds_scope": "training and validation epochs, checkpoint writes and best checkpoint restore",
        "elapsed_seconds_scope": "subject data loading through test evaluation, logging and example plots",
        "confusion_matrix_class_order": sorted(config["label_mapping"], key=config["label_mapping"].get),
        "results": records,
    }
    (output / "logs" / "transformer_run.json").write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")
    save_comparison(metrics, baseline, eegnet_log, output, all_subjects)
    print(f"\n{summary.to_string(float_format='{:.4f}'.format)}\n\nSaved to {output}")


if __name__ == "__main__":
    main()
