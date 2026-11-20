"""Generate the Phase A waveform, class-count and Welch PSD figures."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch


CLASS_NAMES = {
    "left_hand": "Left hand", "right_hand": "Right hand", "feet": "Both feet", "tongue": "Tongue",
}
COLORS = ["#007c83", "#954588", "#ce601b", "#3d61a7"]


def plot_subject_eeg(X, y, metadata, config, signal_info, figure_dir):
    sfreq = signal_info["sfreq"]
    subject = int(metadata["subject"].iloc[0])
    channels = ["C3", "Cz", "C4"]
    selected = [signal_info["channel_names"].index(name) for name in channels]
    signals = X[:, selected] * 1e6
    t = signal_info["first_sample_from_cue_s"] + np.arange(X.shape[-1]) / sfreq
    band = f"{config['fmin']:g}–{config['fmax']:g} Hz"
    first = metadata.iloc[0]

    fig, axes = plt.subplots(3, 1, figsize=(9, 5), sharex=True, layout="constrained")
    fig.suptitle(f"Subject {subject:02d} · Training trial · {CLASS_NAMES[first['label_name']]}",
                 fontweight="bold", x=0.02, ha="left")
    for ax, name, signal in zip(axes, channels, signals[0]):
        ax.plot(t, signal, color=COLORS[0], linewidth=0.85)
        ax.set_ylabel(f"{name} (µV)")
        ax.grid(alpha=0.22)
    axes[-1].set(xlabel="Time from imagery cue (s)", xlim=(config["tmin"], config["tmax"]))
    fig.text(0.02, 0.005, f"{first['trial_id']}  ·  {band} FIR  ·  {sfreq:g} Hz sampling",
             fontsize=9, color="#48515c")
    fig.get_layout_engine().set(rect=(0, 0.04, 1, 0.96))
    fig.savefig(figure_dir / "eeg_waveform.png", dpi=150)
    plt.close(fig)

    nperseg = min(round(2 * sfreq), signals.shape[-1])
    overlap = nperseg // 2
    frequencies, psd = welch(signals, fs=sfreq, window="hann", nperseg=nperseg,
                             noverlap=overlap, scaling="density", axis=-1)
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    ax.axvspan(config["fmin"], config["fmax"], color="#dfe7ef", alpha=0.72, label="FIR passband")
    for (name, code), color in zip(config["label_mapping"].items(), COLORS):
        density = psd[y == code].mean(axis=(0, 1))
        ax.plot(frequencies, 10 * np.log10(np.maximum(density, np.finfo(float).tiny)), color=color,
                label=f"{CLASS_NAMES[name]} (n={np.count_nonzero(y == code)})")
    ax.set(title=f"Subject {subject:02d} · Mean training EEG power spectrum",
           xlabel="Frequency (Hz)", ylabel="Power spectral density (dB re 1 µV²/Hz)", xlim=(0, 60))
    ax.legend(loc="upper right", frameon=False)
    ax.grid(alpha=0.22)
    fig.text(0.02, 0.005,
             f"C3, Cz and C4 · Welch Hann window ({nperseg / sfreq:g} s, {overlap} samples overlap)",
             fontsize=9, color="#48515c")
    fig.get_layout_engine().set(rect=(0, 0.04, 1, 0.96))
    fig.savefig(figure_dir / "eeg_psd.png", dpi=150)
    plt.close(fig)


def plot_class_distribution(manifest, config, figure_dir):
    names = list(config["label_mapping"])
    counts = manifest.groupby(["label_name", "split"]).size().unstack(fill_value=0).reindex(names)
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    base = np.zeros(len(names), dtype=int)
    for split, label, color in zip(("train", "val", "test"), ("Train", "Validation", "Test"), COLORS):
        heights = counts[split].to_numpy()
        ax.bar([CLASS_NAMES[name] for name in names], heights, bottom=base, label=label, color=color)
        base += heights
    for i, total in enumerate(base):
        ax.annotate(f"{total:,}", (i, total), xytext=(0, 5), textcoords="offset points", ha="center")
    n_subjects = manifest["subject"].nunique()
    subject_label = "subject" if n_subjects == 1 else "subjects"
    ax.set(title=f"Motor imagery trials by class and split · {n_subjects} {subject_label}",
           xlabel="Motor imagery class", ylabel="Number of trials", ylim=(0, base.max() * 1.14))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncols=3, frameon=False)
    ax.grid(axis="y", alpha=0.22)
    ax.set_axisbelow(True)
    fig.savefig(figure_dir / "class_distribution.png", dpi=150)
    plt.close(fig)
