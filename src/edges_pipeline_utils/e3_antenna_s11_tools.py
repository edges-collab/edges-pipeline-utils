"""
Functions for testing Antenna S11 variabilities.
"""

import glob
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
    rms = np.sqrt(np.nanmean(array**2))
    return round(rms, digits)


def get_ant_s11(
    year, day, f_low=40, f_high=190, raw=False, n_terms=12, return_model=False
):
    try:
        pattern_base = f"{root_dir}/{year}_{day:03d}_*"
        Tfopen = glob.glob(f"{pattern_base}O.s1p")[0]
        Tfshort = glob.glob(f"{pattern_base}S.s1p")[0]
        Tfload = glob.glob(f"{pattern_base}L.s1p")[0]
        Tfant = glob.glob(f"{pattern_base}ant.s1p")[0]
    except IndexError:
        # If any file is missing, return None
        print(f"Missing one or more required files for year {year}, day {day}")
        return None, None, None

    try:
        # reads1p1 returns a calibrated ReflectionCoefficient (not a (freq, s11) tuple).
        gamma_ant = reads1p1(
            res=49.930,
            s11_file_open=Tfopen,
            s11_file_short=Tfshort,
            s11_file_load=Tfload,
            s11_file_ant=Tfant,
        )
        raw_freq = gamma_ant.freqs
        ea_ant_s11 = gamma_ant.reflection_coefficient

    except ValueError:
        # if the files are inconsistent
        print("Cal files are inconsistent in frequencies")
        return None, None, None

    # get antenna temperature from temperature logger
    file_name = Tfant.split("/")[-1]
    print(file_name)
    temperature = utils.extract_temperature(file_name)

    mask = get_mask(raw_freq, f_low * un.MHz, f_high * un.MHz)

    # use 53-105 MHz --> Alan sets wfstart/stop to 54-104 MHz but also hardcodes /pm 1 to it
    ea_freq = raw_freq[mask]

    ants11_raw = ReflectionCoefficient(
        reflection_coefficient=ea_ant_s11[mask],
        freqs=ea_freq,
    )

    if raw:
        mod_freq = ea_freq
    else:
        mod_freq = np.arange(f_low, f_high, 0.390) * un.MHz  # B18 resolution is 390 kHz

    ants11 = ants11_raw.smoothed(
        params=S11ModelParams(
            model=mdl.Polynomial(
                n_terms=n_terms,
                transform=mdl.Log10Transform(scale=(f_low + f_high) / 2),
            ),
            complex_model_type=mdl.ComplexRealImagModel,
            set_transform_range=True,
            fit_method="alan-qrd",
            find_model_delay=True,
        ),
        freqs=mod_freq,
    )

    if return_model:
        return ants11
    return temperature.value, mod_freq, ants11.reflection_coefficient


def get_cut_off_freq_range(cut_off=-10, f_low=50, f_high=120, n_terms=16):
    _, ref_freq, ref_ants11 = get_ant_s11(
        year=2023, day=154, f_low=f_low, f_high=f_high, n_terms=n_terms
    )
    """
    Returns the frequency range based on S11 cut-off
    """

    s11_dB = 20 * np.log10(np.abs(ref_ants11))

    # this makes all the values close to zero, closest to zero will be the crossing point of the cut-off value
    crossings = np.where(np.diff(np.sign(s11_dB - cut_off)))[0]

    # # Find indices of the closest values
    if len(crossings) >= 2:
        first_index = crossings[0]
        last_index = crossings[-1]

    return ref_freq[first_index], ref_freq[last_index]
