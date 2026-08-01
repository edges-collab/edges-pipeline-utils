"""Functions for testing Antenna S11 variabilities."""

from pathlib import Path

import numpy as np
from astropy import units as un
from edges import modeling as mdl
from edges.alanmode.alanmode import reads1p1
from edges.cal import ReflectionCoefficient, S11ModelParams
from edges.frequencies import get_mask

from edges_pipeline_utils import utils

root_dir: Path = Path(
    "/data5/edges/data/EDGES3_data/MRO"
)  # this is where the raw files are
alan_dir: Path = Path("/data4/vydula/edges/edges3_files/scripts/alan_300_310_tests/")
datadir: Path = Path("/data4/vydula/edges/packages/edges3-data-analysis/data/")


def calculate_rms(array, digits=3):
    """Return the RMS of ``array``, rounded to ``digits``."""
    rms = np.sqrt(np.nanmean(array**2))
    return round(rms, digits)


def _as_mhz_float(x: float | un.Quantity) -> float:
    """Band edges in MHz as a plain float.

    Accepts a number in MHz or any :class:`~astropy.units.Quantity` with frequency
    units, so callers are not double-converted with ``f * un.MHz``.
    """
    if isinstance(x, un.Quantity):
        if not x.unit.is_equivalent(un.MHz):
            raise TypeError(
                f"Expected a frequency for f_low/f_high (e.g. MHz); got unit {x.unit}. "
                "Pass a float in MHz, or a single frequency Quantity, "
                "not f*u.MHz twice."
            )
        return float(x.to(un.MHz).value)
    if hasattr(x, "to_value"):
        return float(x.to_value(un.MHz))
    return float(np.asarray(x, dtype=float))


def get_ant_s11(
    year, day, f_low=40, f_high=190, raw=False, n_terms=12, return_model=False
):
    """Read and optionally smooth antenna S11 for a given year/day."""
    try:
        pattern = f"{year}_{day:03d}_*"
        tf_open = next(root_dir.glob(f"{pattern}O.s1p"))
        tf_short = next(root_dir.glob(f"{pattern}S.s1p"))
        tf_load = next(root_dir.glob(f"{pattern}L.s1p"))
        tf_ant = next(root_dir.glob(f"{pattern}ant.s1p"))
    except StopIteration:
        # If any file is missing, return None
        return None, None, None

    try:
        # reads1p1 returns a calibrated ReflectionCoefficient (not a (freq, s11) tuple).
        gamma_ant = reads1p1(
            res=49.930,
            s11_file_open=str(tf_open),
            s11_file_short=str(tf_short),
            s11_file_load=str(tf_load),
            s11_file_ant=str(tf_ant),
        )
        raw_freq = gamma_ant.freqs
        ea_ant_s11 = gamma_ant.reflection_coefficient

    except ValueError:
        # if the files are inconsistent
        return None, None, None

    # get antenna temperature from temperature logger
    temperature = utils.extract_temperature(tf_ant.name)

    f_low_mhz = _as_mhz_float(f_low)
    f_high_mhz = _as_mhz_float(f_high)
    mask = get_mask(raw_freq, f_low_mhz * un.MHz, f_high_mhz * un.MHz)

    # Alan sets wfstart/stop to 54-104 MHz but also hardcodes +/- 1 MHz
    ea_freq = raw_freq[mask]

    ants11_raw = ReflectionCoefficient(
        reflection_coefficient=ea_ant_s11[mask],
        freqs=ea_freq,
    )

    # B18 resolution is 390 kHz
    mod_freq = ea_freq if raw else np.arange(f_low_mhz, f_high_mhz, 0.390) * un.MHz

    ants11 = ants11_raw.smoothed(
        params=S11ModelParams(
            model=mdl.Polynomial(
                n_terms=n_terms,
                transform=mdl.Log10Transform(scale=(f_low_mhz + f_high_mhz) / 2),
            ),
            complex_model_type=mdl.ComplexRealImagModel,
            set_transform_range=True,
            fit_method="alan-qrd",
            find_model_delay=True,
        ),
        freqs=mod_freq,
    )

    if return_model:
        return ants11_raw, ants11
    return temperature.value, mod_freq, ants11.reflection_coefficient


def get_cut_off_freq_range(cut_off=-10, f_low=50, f_high=120, n_terms=16):
    """Return the frequency range based on an S11 magnitude cut-off in dB."""
    _, ref_freq, ref_ants11 = get_ant_s11(
        year=2023, day=154, f_low=f_low, f_high=f_high, n_terms=n_terms
    )

    s11_dB = 20 * np.log10(np.abs(ref_ants11))

    # Closest-to-zero values are where S11 crosses the cut-off
    crossings = np.where(np.diff(np.sign(s11_dB - cut_off)))[0]

    if len(crossings) >= 2:
        first_index = crossings[0]
        last_index = crossings[-1]

    return ref_freq[first_index], ref_freq[last_index]
