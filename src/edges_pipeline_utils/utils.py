"""Various utilities for use in notebooks."""

import re
from importlib.metadata import version
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from edges import modeling as mdl
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from pygsdata import GSData


def yday_to_alanday(year: int, day: int):
    year = int(year)
    day = int(day)
    if year == 2015:
        return day
    if year == 2016:
        return 366 + day
    if year == 2017:
        return 732 + day
    raise ValueError(f"Year must be 2015, 2016 or 2017, got '{year}'")


def plot_single_spectrum(data: GSData, alanspec=None, attribute="data"):
    if alanspec is not None:
        fig, ax = plt.subplots(
            2,
            1,
            sharex=True,
            gridspec_kw={"hspace": 0, "wspace": 0},
            constrained_layout=True,
            squeeze=False,
        )
    else:
        fig, ax = plt.subplots(
            1,
            1,
            sharex=True,
            gridspec_kw={"hspace": 0, "wspace": 0},
            constrained_layout=True,
            squeeze=False,
        )

    flags = data.flagged_nsamples[0, 0, 0] == 0
    attr = getattr(data, attribute)[0, 0, 0]
    ax[0, 0].plot(data.freqs, np.where(flags, np.nan, attr), label="edges-analysis")

    ax[0, 0].set_ylabel("Temperature [K]")
    if alanspec is not None:
        ax[0, 0].plot(data.freqs, np.where(flags, np.nan, alanspec), label="C-code")
        ax[1, 0].plot(
            data.freqs,
            1000 * np.where(flags, np.nan, attr - alanspec),
            label="Difference",
            color="k",
        )
        ax[1, 0].set_ylabel("Difference [mK]")
        ax[1, 0].legend(frameon=False)

    ax[-1, 0].set_xlabel("Frequency [MHz]")

    ax[0, 0].legend(frameon=False)


def print_versions():
    print("Versions: ")
    for pkg in ["read_acq", "pygsdata", "edges-analysis"]:
        print(f"{pkg:>20}: {version(pkg)}")


def plot_rms_lst(gsdata_obj, figsave=None):
    """
    Plot residuals with LST color mapping and RMS vs. LST in subplots.

    Parameters
    ----------
        gsdata_obj : GSData
            The data object containing frequency, LST, and residuals.
        figsave : str, optional
            Path to save the figure. If None, the figure is not saved.

    Returns
    -------
        fig : matplotlib.figure.Figure
            The created figure object.
    """
    offset = 0
    rms = []
    model_fit_freqs = gsdata_obj.freqs
    thermal_noise = get_thermal_noise(gsdata_obj)

    # Create subplots
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={"height_ratios": [2, 1]})

    # Plot residuals
    ax1 = axs[0]
    for i in range(len(gsdata_obj.lsts)):
        model = mdl.LinLog(n_terms=5)
        LL = model.at(x=model_fit_freqs)
        res_5terms = LL.fit(ydata=gsdata_obj.data[0, 0, i, :], xdata=model_fit_freqs)

        # Normalize LSTs for color mapping
        norm = Normalize(vmin=0, vmax=24)
        color = plt.cm.viridis(norm(gsdata_obj.lsts[i]))
        ax1.plot(
            model_fit_freqs,
            offset + res_5terms.residual,
            color=color,
            label=str(gsdata_obj.lsts[i]),
        )

        # Append RMS
        rms.append(calculate_rms(res_5terms.residual))

    ax1.set_xlabel("Frequency (MHz)")
    ax1.set_ylabel("Residuals (K)")
    ax1.set_title("5 term linlog, Averaged")
    ax1.grid()

    # Add colorbar to the first plot
    scalar_map = ScalarMappable(norm=norm, cmap="viridis")
    scalar_map.set_array([])
    cbar = fig.colorbar(scalar_map, ax=ax1)
    cbar.set_label("LST (hr)")

    # Plot RMS vs LST
    ax2 = axs[1]
    rms = np.array(rms)
    ax2.errorbar(
        gsdata_obj.lsts,
        rms,
        yerr=thermal_noise,
        marker="o",
        mfc="red",
        c="k",
        alpha=0.8,
    )
    ax2.set_ylabel("RMS (K)")
    ax2.set_xlabel("LST (hr)")
    ax2.grid()

    plt.tight_layout()

    if figsave:
        plt.savefig(figsave)

    plt.show()
    return fig


def get_thermal_noise(gsdata_obj: GSData, n_terms=20):
    thermal_noise = []

    for i in range(len(gsdata_obj.lsts)):
        model = mdl.LinLog(n_terms=n_terms)
        model_fit_freqs = gsdata_obj.freqs

        LL = model.at(x=model_fit_freqs)
        res = LL.fit(ydata=gsdata_obj.data[0, 0, i, :], xdata=model_fit_freqs)

        thermal_noise.append(calculate_rms(res.residual))

    thermal_noise = np.array(thermal_noise)

    return thermal_noise


def calculate_rms(array, digits=3):
    rms = np.sqrt(np.mean(array**2))
    return round(rms, digits)


def find_closest_s11date(datadir: str, specyear: int, specday: int) -> str:
    """Find s11 date stem (year_day_run) in datadir whose (year, day) is closest to specyear, specday."""
    data_path = Path(datadir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {datadir}")
    stem_to_yd = {}
    for f in data_path.glob("*_O.s1p"):
        m = re.match(r"(\d{4})_(\d{3})_(\d+)_O\.s1p", f.name)
        if not m:
            continue
        y, d, run = int(m.group(1)), int(m.group(2)), m.group(3)
        stem = f"{y}_{d:03d}_{run}"
        s_file = data_path / f"{stem}_S.s1p"
        l_file = data_path / f"{stem}_L.s1p"
        if s_file.exists() and l_file.exists():
            stem_to_yd[stem] = (y, d)
    if not stem_to_yd:
        raise FileNotFoundError(
            f"No s11 calibration files (O/S/L.s1p) found in {datadir} for any day near specyear={specyear}, specday={specday}"
        )

    def day_offset(stem):
        y, d = stem_to_yd[stem]
        return (y - specyear) * 365 + (d - specday)

    closest_stem = min(stem_to_yd, key=lambda s: abs(day_offset(s)))
    return closest_stem


def find_closest_calkit_stem(data_dir: str, year: int, day: int) -> str | None:
    """Find calkit file stem (year_day_run) whose (year, day) is closest to Antenna S11 date"""
    import re

    data_path = Path(data_dir)
    if not data_path.exists():
        return None
    stem_to_yd = {}
    for f in data_path.glob("*_O.s1p"):
        m = re.match(r"(\d{4})_(\d{3})_(\d+)_O\.s1p", f.name)
        if not m:
            continue
        y, d, run = int(m.group(1)), int(m.group(2)), m.group(3)
        stem = f"{y}_{d:03d}_{run}"
        s_file = data_path / f"{stem}_S.s1p"
        l_file = data_path / f"{stem}_L.s1p"
        ant_file = data_path / f"{stem}_ant.s1p"
        if s_file.exists() and l_file.exists() and ant_file.exists():
            stem_to_yd[stem] = (y, d)
    if not stem_to_yd:
        return None

    def day_offset(stem):
        y, d = stem_to_yd[stem]
        return (y - year) * 365 + (d - day)

    closest_stem = min(stem_to_yd, key=lambda s: abs(day_offset(s)))
    return closest_stem
