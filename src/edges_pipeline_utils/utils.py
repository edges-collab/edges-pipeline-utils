"""Various utilities for use in notebooks."""

import re
import sys
import sys
from importlib.metadata import version
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from edges import modeling as mdl
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from pygsdata import GSData


def yday_to_alanday(year: int, day: int) -> int:
    """Convert (year, day-of-year) to Alan's continuous day index."""
    year = int(year)
    day = int(day)
    if year == 2015:
        return day
    if year == 2016:
        return 366 + day
    if year == 2017:
        return 732 + day
    raise ValueError(f"Year must be 2015, 2016 or 2017, got '{year}'")


def plot_single_spectrum(
    data: GSData, alanspec=None, attribute: str = "data"
) -> None:
    """Plot a single spectrum from GSData, optionally with Alan's spectrum."""
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


def print_versions() -> None:
    """Print versions of key EDGES packages."""
    sys.stdout.write("Versions: \n")
    for pkg in ["read_acq", "pygsdata", "edges-analysis"]:
        sys.stdout.write(f"{pkg:>20}: {version(pkg)}\n")


def find_closest_s11date(datadir: str, specyear: int, specday: int) -> str:
    """Find s11 date stem (year_day_run) in datadir closest to specyear, specday."""
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
            f"No s11 calibration files (O/S/L.s1p) found in {datadir} "
            f"for any day near specyear={specyear}, specday={specday}"
        )

    def day_offset(stem):
        y, d = stem_to_yd[stem]
        return (y - specyear) * 365 + (d - specday)

    closest_stem = min(stem_to_yd, key=lambda s: abs(day_offset(s)))
    return closest_stem


def find_closest_calkit_stem(data_dir: str, year: int, day: int) -> str | None:
    """Find calkit file stem closest to Antenna S11 date."""
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
