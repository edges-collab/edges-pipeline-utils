"""Various utilities for use in notebooks."""

from importlib.metadata import version

import matplotlib.pyplot as plt
import numpy as np
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
