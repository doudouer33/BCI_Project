"""Read, epoch and split BNCI2014_001 with the shared experiment protocol."""

import argparse
import json
import os
import platform
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

import mne
import numpy as np
import pandas as pd
import yaml
from moabb.datasets import BNCI2014_001
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]


def load_config(path=ROOT / "configs/experiment.yaml"):
    with Path(path).open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def count_nonfinite(data):
    return {"nan": int(np.isnan(data).sum()), "inf": int(np.isinf(data).sum())}


def sample_offsets(seconds, sfreq):
    offsets = np.asarray(seconds) * sfreq
    samples = np.rint(offsets).astype(int)
    if not np.allclose(offsets, samples, rtol=0, atol=1e-8):
        raise ValueError(f"Times {seconds} do not fall on the {sfreq:g} Hz sample grid")
    return samples


def load_subject(subject_id, config):
    """Return (trials × channels × times in V, integer labels, trial DataFrame)."""
    dataset = BNCI2014_001(artifact_handling="annotate")
    mapping = config["label_mapping"]
    if config["dataset"] != "BNCI2014_001" or set(mapping) != set(dataset.event_id):
        raise ValueError("Expected BNCI2014_001 and its four motor imagery labels")
    if set(mapping.values()) != set(range(4)):
        raise ValueError("Use four distinct class codes from 0 to 3")

    sfreq = config["sfreq"]
    cue_offset = config["cue_offset_from_trial_start_s"]
    tmin, tmax = config["tmin"], config["tmax"]
    task_start, task_stop = dataset.interval
    if task_start != cue_offset or not 0 <= tmin < tmax <= task_stop - task_start:
        raise ValueError(f"Configured cue/window disagrees with MOABB interval {dataset.interval}")
    cue_samples, start_offset, stop_offset = sample_offsets([cue_offset, tmin, tmax], sfreq)

    cache = (ROOT / Path(config["data_dir"]).expanduser()).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    # The pinned MOABB version reads these settings during get_data().
    env = {
        "MNE_DATA": str(cache), "MNE_DATASETS_BNCI_PATH": str(cache),
        "MOABB_DOWNLOAD_PROVIDER": os.environ.get("MOABB_DOWNLOAD_PROVIDER", "upstream"),
    }
    with patch.dict(os.environ, env):
        sessions = dataset.get_data(subjects=[subject_id])[subject_id]
    session_names = [config["train_session"], config["test_session"]]
    if len(set(session_names)) != 2 or set(sessions) != set(session_names):
        raise ValueError(f"Subject {subject_id}: found sessions {list(sessions)}, expected {session_names}")

    filter_options = dict(config["filter"])
    pad = filter_options.pop("pad")
    if filter_options["method"] != "fir" or filter_options["phase"] != "zero":
        raise ValueError("The shared protocol requires a zero-phase FIR filter")
    band = (config["fmin"], config["fmax"])
    coefficients = mne.filter.create_filter(None, sfreq, *band, **filter_options, verbose=False)
    # A zero-phase FIR uses only this many samples on each side.
    half_length = len(coefficients) // 2
    label_names = {code: name for name, code in dataset.event_id.items()}
    arrays, rows, run_stats = [], [], []
    channel_names = None

    for session in session_names:
        for run, raw in sorted(sessions[session].items(), key=lambda pair: int(pair[0])):
            context = f"Subject {subject_id}, session {session}, run {run}"
            types = dict(zip(raw.ch_names, raw.get_channel_types()))
            eeg_names = [name for name, kind in types.items() if kind == "eeg"]
            if channel_names is None:
                channel_names = eeg_names
            if len(eeg_names) != 22 or eeg_names != channel_names or raw.info["sfreq"] != sfreq:
                raise ValueError(f"{context}: unexpected EEG channels or sampling rate")

            triggers = mne.find_events(raw, stim_channel="STI", shortest_event=1, verbose=False)
            cues, _ = mne.events_from_annotations(raw, event_id=dataset.event_id, verbose=False)
            if (
                not len(triggers)
                or not np.array_equal(triggers[:, 2], cues[:, 2])
                or not np.array_equal(triggers[:, 0] + cue_samples, cues[:, 0])
            ):
                raise ValueError(f"{context}: MOABB cue annotations disagree with the trial-start triggers")
            artifact_cues = set()
            if "bnci_artifact" in raw.annotations.description:
                flagged, _ = mne.events_from_annotations(
                    raw, event_id={"bnci_artifact": 1}, verbose=False,
                )
                artifact_cues = set(flagged[:, 0])
            if not artifact_cues.issubset(cues[:, 0]):
                raise ValueError(f"{context}: an artifact annotation does not align with a cue")

            starts = cues[:, 0] - raw.first_samp + start_offset
            stops = cues[:, 0] - raw.first_samp + stop_offset
            if starts[0] < half_length or stops[-1] + half_length > raw.n_times:
                raise ValueError(f"{context}: a filtered epoch would touch the recording boundary")
            gaps = starts[1:] - stops[:-1]
            if np.any(gaps < 2 * half_length):
                raise ValueError(f"{context}: FIR input would overlap between trial windows")

            eeg = raw.get_data(picks=eeg_names)
            raw_quality = count_nonfinite(eeg)
            if any(raw_quality.values()):
                raise ValueError(f"{context}: non-finite raw EEG {raw_quality}")
            filtered = mne.filter.filter_data(
                eeg, sfreq, *band, **filter_options, pad=pad,
                filter_length=len(coefficients), copy=False, verbose=False,
            )
            epochs = np.stack([filtered[:, start:stop] for start, stop in zip(starts, stops)])
            epoch_quality = count_nonfinite(epochs)
            if any(epoch_quality.values()):
                raise ValueError(f"{context}: non-finite epoched EEG {epoch_quality}")
            arrays.append(epochs)

            for trigger, cue in zip(triggers, cues):
                event_sample, _, event_code = map(int, trigger)
                cue_sample = int(cue[0])
                label_name = label_names[event_code]
                rows.append({
                    "subject": subject_id, "session": session, "run": run,
                    "trial_id": f"s{subject_id:02d}/{session}/{run}/{event_sample}",
                    "event_sample": event_sample, "cue_sample": cue_sample, "event_code": event_code,
                    "label_name": label_name, "label": mapping[label_name],
                    "artifact_flag": cue_sample in artifact_cues,
                })
            run_stats.append({
                "session": session, "run": run, "raw_samples": int(raw.n_times),
                "trials": len(epochs), "artifact_flags": len(artifact_cues),
                "raw_eeg_nonfinite": raw_quality, "epoch_nonfinite": epoch_quality,
                "non_eeg_channels": {name: kind for name, kind in types.items() if kind != "eeg"},
                "cue_offset_samples": int(cue_samples),
                "min_gap_between_epochs_samples": int(np.min(gaps)) if gaps.size else None,
            })

    metadata = pd.DataFrame(rows).rename_axis("index")
    if metadata["trial_id"].duplicated().any():
        raise ValueError(f"Subject {subject_id}: duplicate trial IDs")
    metadata.attrs.update({
        "channel_names": channel_names, "sfreq": sfreq, "units": "V",
        "moabb_event_codes": dataset.event_id, "moabb_interval_s": list(dataset.interval),
        "first_sample_from_cue_s": start_offset / sfreq,
        "last_sample_from_cue_s": (stop_offset - 1) / sfreq,
        "fir_length_samples": len(coefficients), "runs": run_stats,
    })
    return np.concatenate(arrays), metadata["label"].to_numpy(copy=True), metadata


def get_split_indices(metadata, config):
    """Return positional train/val/test indices for exactly one subject."""
    source = config["train_session"]
    target = config["test_session"]
    if (
        source == target
        or set(metadata["session"]) != {source, target}
        or metadata["subject"].nunique() != 1
    ):
        raise ValueError("Expected exactly one subject with the configured train and test sessions")
    development = np.flatnonzero(metadata["session"].eq(source))
    test = np.flatnonzero(metadata["session"].eq(target))
    train, val = train_test_split(
        development, test_size=config["val_ratio"], random_state=config["random_seed"],
        stratify=metadata["label"].iloc[development],
    )
    splits = (np.sort(train), np.sort(val), test)
    expected = set(config["label_mapping"].values())
    if any(set(metadata["label"].iloc[indices]) != expected for indices in splits):
        raise ValueError("Every split must contain all four classes")
    return splits


def summarize_subject(subject_id, X, metadata, config):
    splits = get_split_indices(metadata, config)
    split_names = np.empty(len(metadata), dtype="U5")
    for name, indices in zip(("train", "val", "test"), splits):
        split_names[indices] = name
    trials = metadata.assign(split=split_names)

    def counts(labels):
        return {name: int(labels.eq(name).sum()) for name in config["label_mapping"]}

    statistics = {
        "subject": subject_id, "shape": list(X.shape), "dtype": str(X.dtype),
        "class_counts": counts(trials["label_name"]),
        "session_counts": trials["session"].value_counts().to_dict(),
        "split_counts": trials["split"].value_counts().to_dict(),
        "split_class_counts": {
            split: counts(group["label_name"]) for split, group in trials.groupby("split")
        },
        "nan": sum(run["epoch_nonfinite"]["nan"] for run in metadata.attrs["runs"]),
        "inf": sum(run["epoch_nonfinite"]["inf"] for run in metadata.attrs["runs"]),
        **metadata.attrs,
    }
    trials.attrs = {}
    return statistics, trials, splits[0]


def main():
    from .plot_results import plot_class_distribution, plot_subject_eeg

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiment.yaml")
    parser.add_argument("--subjects", nargs="+", type=int, help="Default: all subjects in the configuration")
    parser.add_argument("--data-dir", type=Path, help="Override the MOABB data cache path")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.data_dir is not None:
        config["data_dir"] = str(args.data_dir.expanduser().resolve())
    subjects = args.subjects or config["subjects"]
    if len(set(subjects)) != len(subjects):
        parser.error("--subjects must contain distinct subject IDs")

    statistics, manifests = [], []
    for subject_id in subjects:
        X, y, metadata = load_subject(subject_id, config)
        if statistics and metadata.attrs["channel_names"] != statistics[0]["channel_names"]:
            raise ValueError(f"Subject {subject_id}: EEG channel order differs across subjects")
        stats, trials, train_idx = summarize_subject(subject_id, X, metadata, config)
        statistics.append(stats)
        manifests.append(trials)
        print(f"Subject {subject_id}: {X.shape} {stats['units']}, {stats['sfreq']:g} Hz, "
              f"classes={stats['class_counts']}, sessions={stats['session_counts']}, "
              f"splits={stats['split_counts']}, NaN={stats['nan']}, Inf={stats['inf']}", flush=True)
        if subject_id == subjects[0]:
            example = (X[train_idx], y[train_idx], trials.iloc[train_idx], metadata.attrs)
        del X

    manifest = pd.concat(manifests)
    figure_dir = args.output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    train_X, train_y, train_trials, signal_info = example
    plot_subject_eeg(train_X, train_y, train_trials, config, signal_info, figure_dir)
    plot_class_distribution(manifest, config, figure_dir)
    manifest.to_csv(args.output_dir / "trial_splits.csv")
    output = {
        "config": config,
        "versions": {"python": platform.python_version(), **{
            package: version(package) for package in (
                "moabb", "mne", "numpy", "scipy", "pandas", "scikit-learn", "matplotlib", "pyyaml",
            )
        }},
        "n_subjects": len(statistics), "n_trials": len(manifest),
        "class_counts": manifest["label_name"].value_counts().to_dict(),
        "split_counts": manifest["split"].value_counts().to_dict(),
        "subjects": statistics,
    }
    (args.output_dir / "data_statistics.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )


if __name__ == "__main__":
    main()
