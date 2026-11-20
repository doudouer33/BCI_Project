"""Compare EEGNet channel-wise Z-score and full-session refitting."""

import argparse
import json
import os
import platform
import random
import shlex
import sys
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import mne
import numpy as np
import pandas as pd
import torch
from braindecode.models import EEGNet
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.load_data import ROOT, get_split_indices, load_config, load_subject
from src.plot_results import COLORS, PercentFormatter, plot_eegnet_training, plt
from src.torch_training import VOLTS_TO_MICROVOLTS, fit, make_loader, run_epoch


RESULTS = ROOT / "results"
GROUPS = ("Baseline", "Z-score", "Refit", "Z-score + Refit")


def fit_channel_stats(X, indices, epsilon_uv):
    signals_uv = X[indices] * VOLTS_TO_MICROVOLTS
    return {
        "n_trials": len(signals_uv),
        "mean_uv": signals_uv.mean(axis=(0, 2)).tolist(),
        "std_uv": signals_uv.std(axis=(0, 2), ddof=0).tolist(),
        "epsilon_uv": epsilon_uv,
    }


def normalized_loader(X, y, indices, batch_size, stats, shuffle_seed=None):
    mean = np.asarray(stats["mean_uv"])[None, :, None]
    scale = np.asarray(stats["std_uv"])[None, :, None] + stats["epsilon_uv"]
    signals = X[indices] * VOLTS_TO_MICROVOLTS
    signals -= mean
    signals /= scale
    signals = signals.astype(np.float32)
    if not np.isfinite(signals).all():
        raise FloatingPointError("Z-score input contains NaN or Inf")
    dataset = TensorDataset(torch.from_numpy(signals), torch.as_tensor(y[indices], dtype=torch.long))
    generator = None if shuffle_seed is None else torch.Generator().manual_seed(shuffle_seed)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=generator is not None, generator=generator,
    )


def new_model(model_args, seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return EEGNet(**model_args).to(device)


def refit(model_args, seed, loader, epochs, settings, device):
    model = new_model(model_args, seed, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    history, batch_check = [], {}
    for epoch in range(1, epochs + 1):
        loss, accuracy, _ = run_epoch(
            model, loader, criterion, device, optimizer, batch_check if epoch == 1 else None,
        )
        history.append({"epoch": epoch, "train_loss": loss, "train_accuracy": accuracy})
    return model, pd.DataFrame(history), batch_check


def evaluate(model, loader, trials, device, path):
    _, _, prediction = run_epoch(model, loader, nn.CrossEntropyLoss(), device)
    trials.assign(y_pred=prediction).to_csv(path, index=False)
    return {
        "accuracy": accuracy_score(trials["y_true"], prediction),
        "macro_f1": f1_score(trials["y_true"], prediction, labels=range(4),
                             average="macro", zero_division=0),
    }


def verify_splits(X, y, metadata, indices, manifest, baseline):
    if list(X.shape) != baseline["input_shape"] or metadata.attrs["units"] != "V":
        raise ValueError("Data shape or units differ from the baseline")
    for split, split_idx in zip(("train", "val", "test"), indices):
        saved = manifest.loc[manifest["split"].eq(split)]
        if (
            len(split_idx) != baseline["split_counts"][split]
            or not np.array_equal(split_idx, saved["index"])
            or not np.array_equal(metadata.iloc[split_idx]["trial_id"], saved["trial_id"])
            or not np.array_equal(y[split_idx], saved["label"])
        ):
            raise ValueError(f"{split} trials differ from the original split manifest")


def run_subject(subject, config, epsilon_uv, manifest, baseline, device, output, checkpoint):
    print(f"Subject {subject}: fitting Z-score on the original train split", flush=True)
    X, y, metadata = load_subject(subject, config)
    indices = get_split_indices(metadata, config)
    verify_splits(X, y, metadata, indices, manifest, baseline)
    train_idx, val_idx, test_idx = indices
    full_train_idx = np.sort(np.concatenate([train_idx, val_idx]))
    full_session = np.flatnonzero(metadata["session"].eq(config["train_session"]))
    if len(full_train_idx) != 288 or not np.array_equal(full_train_idx, full_session):
        raise ValueError("Refit indices must cover the complete training session")

    settings = config["eegnet"]
    seed = config["random_seed"]
    batch_size = settings["batch_size"]
    model_args = baseline["model_args"]
    train_stats = fit_channel_stats(X, train_idx, epsilon_uv)
    z_train = normalized_loader(X, y, train_idx, batch_size, train_stats, shuffle_seed=seed)
    z_val = normalized_loader(X, y, val_idx, batch_size, train_stats)
    z_model = new_model(model_args, seed, device)
    z_history, selection = fit(z_model, z_train, z_val, settings, device, checkpoint)
    baseline_epoch, z_epoch = baseline["best_epoch"], selection["best_epoch"]

    print(f"Subject {subject}: refitting for {baseline_epoch} raw and {z_epoch} Z-score epochs",
          flush=True)
    full_train_stats = fit_channel_stats(X, full_train_idx, epsilon_uv)
    raw_train = make_loader(X, y, full_train_idx, batch_size, shuffle_seed=seed)
    z_full_train = normalized_loader(
        X, y, full_train_idx, batch_size, full_train_stats, shuffle_seed=seed,
    )
    raw_model, raw_history, raw_batch = refit(
        model_args, seed, raw_train, baseline_epoch, settings, device,
    )
    z_refit_model, z_refit_history, z_refit_batch = refit(
        model_args, seed, z_full_train, z_epoch, settings, device,
    )

    # All training and epoch selection for this subject finish before test loaders are built.
    trials = metadata.iloc[test_idx][["trial_id"]].assign(y_true=y[test_idx])
    variants = (
        (GROUPS[1], "zscore", z_model, z_history, train_stats, z_epoch, len(train_idx), None),
        (GROUPS[2], "refit", raw_model, raw_history, None, baseline_epoch, len(full_train_idx), raw_batch),
        (GROUPS[3], "zscore_refit", z_refit_model, z_refit_history, full_train_stats,
         z_epoch, len(full_train_idx), z_refit_batch),
    )
    rows, refit_batch_checks = [], {}
    for group, slug, model, history, stats, best_epoch, n_train, refit_batch in variants:
        test = (make_loader(X, y, test_idx, batch_size) if stats is None else
                normalized_loader(X, y, test_idx, batch_size, stats))
        name = f"{slug}_s{subject}"
        scores = evaluate(model, test, trials, device, output / "predictions" / f"{name}.csv")
        history.to_csv(output / "logs" / f"{name}_epochs.csv", index=False)
        rows.append({
            "subject": subject, "experiment": group, "seed": seed, "best_epoch": best_epoch,
            "selection_epochs": baseline["epochs_run"] if group == "Refit" else selection["epochs_run"],
            "refit_epochs": len(history) if refit_batch is not None else 0,
            "n_train_trials": n_train, **scores,
        })
        if refit_batch is not None:
            refit_batch_checks[group] = refit_batch
        print(f"  {group}: Accuracy {scores['accuracy']:.2%}, Macro-F1 {scores['macro_f1']:.2%}",
              flush=True)

    plot_eegnet_training(subject, z_history, z_epoch, output / "figures")
    return rows, {
        "subject": subject,
        "model_args": model_args,
        "split_counts": baseline["split_counts"],
        "channel_names": metadata.attrs["channel_names"],
        "zscore_train": train_stats,
        "zscore_full_train": full_train_stats,
        "raw_best_epoch": baseline_epoch,
        "zscore_selection": selection,
        "refit_first_batches": refit_batch_checks,
    }


def plot_mean_accuracy(summary, path):
    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    means = summary["accuracy_mean"]
    bars = ax.bar(summary.index, means, color=COLORS)
    ax.bar_label(bars, labels=[f"{mean:.1%}" for mean in means], padding=4)
    ax.set(title=f"EEGNet mean test accuracy · {summary['n_subjects'].iloc[0]} subjects",
           ylabel="Accuracy", ylim=(0, 1))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.grid(axis="y", alpha=0.22)
    ax.set_axisbelow(True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/eegnet_ablation.yaml")
    parser.add_argument("--subjects", type=int, nargs="+", help="Default: all nine subjects")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    ablation = load_config(args.config)
    config = load_config(args.config.parent / ablation["experiment_config"])
    epsilon_uv = float(ablation["zscore_epsilon_uv"])
    if not np.isfinite(epsilon_uv) or epsilon_uv <= 0:
        parser.error("zscore_epsilon_uv must be finite and positive")
    baseline_log = json.loads((RESULTS / "logs/eegnet_run.json").read_text(encoding="utf-8"))
    baseline_config = baseline_log["config"]
    common_config = {key: value for key, value in config.items() if key != "transformer"}
    if common_config != {key: value for key, value in baseline_config.items() if key != "transformer"}:
        parser.error("Shared experiment configuration must match the saved EEGNet baseline")
    subjects = sorted(args.subjects or config["subjects"])
    if len(subjects) != len(set(subjects)) or not set(subjects) <= set(config["subjects"]):
        parser.error("--subjects must contain distinct IDs from the experiment config")

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

    output = RESULTS / "eegnet_ablation"
    if subjects != sorted(config["subjects"]):
        output /= "subject_" + "_".join(map(str, subjects))
    for directory in ("figures", "logs", "predictions"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(RESULTS / "trial_splits.csv")
    baseline_records = {record["subject"]: record for record in baseline_log["results"]}
    rows, records = [], []
    with TemporaryDirectory() as temp_dir, mne.use_log_level("WARNING"):
        for subject in subjects:
            subject_rows, record = run_subject(
                subject, config, epsilon_uv, manifest.loc[manifest["subject"].eq(subject)],
                baseline_records[subject], device, output, Path(temp_dir) / "zscore_best.pt",
            )
            rows.extend(subject_rows)
            records.append(record)

    baseline_metrics = pd.read_csv(RESULTS / "eegnet_metrics.csv").set_index("subject")
    for subject in subjects:
        baseline = baseline_records[subject]
        rows.append({
            "subject": subject, "experiment": "Baseline", "seed": baseline["seed"],
            "best_epoch": baseline["best_epoch"], "selection_epochs": baseline["epochs_run"],
            "refit_epochs": 0, "n_train_trials": baseline["split_counts"]["train"],
            **baseline_metrics.loc[subject, ["accuracy", "macro_f1"]].to_dict(),
        })

    metrics = pd.DataFrame(rows)
    metrics["experiment"] = pd.Categorical(metrics["experiment"], categories=GROUPS, ordered=True)
    metrics = metrics.sort_values(["experiment", "subject"])
    metrics.to_csv(output / "metrics.csv", index=False)
    summary = metrics.groupby("experiment", observed=True).agg(
        n_subjects=("subject", "size"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", lambda value: value.std(ddof=0)),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", lambda value: value.std(ddof=0)),
    )
    summary.to_csv(output / "summary.csv")
    plot_mean_accuracy(summary, output / "figures" / "mean_accuracy.png")

    log = {
        "command": shlex.join(sys.orig_argv),
        "config": config,
        "ablation": ablation,
        "subjects": subjects,
        "baseline_metrics": "results/eegnet_metrics.csv",
        "baseline_run": "results/logs/eegnet_run.json",
        "split_manifest": "results/trial_splits.csv",
        "versions": {"python": platform.python_version(), **{
            package: version(package) for package in baseline_log["versions"] if package != "python"
        }},
        "device": device.type,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda": torch.version.cuda if device.type == "cuda" else None,
        "cudnn": torch.backends.cudnn.version() if device.type == "cuda" else None,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "torch_cpu_threads": torch.get_num_threads(),
        "optimizer": "Adam",
        "loss": "CrossEntropyLoss",
        "selection": "minimum validation loss, earliest epoch on ties",
        "refit": "fresh model and Adam at the original seed, fixed epochs on all of 0train",
        "class_order": baseline_log["confusion_matrix_class_order"],
        "results": records,
    }
    (output / "logs" / "run.json").write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")
    print(f"\n{summary.to_string(float_format='{:.4f}'.format)}\n\nSaved to {output}")


if __name__ == "__main__":
    main()
