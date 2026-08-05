"""Various utilities for use in notebooks."""

import re
import sys
from datetime import UTC, datetime, time, timedelta
from importlib.metadata import version
from pathlib import Path

import astropy.units as un
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pygsdata import GSData

root_dir: Path = Path(
    "/data5/edges/data/EDGES3_data/MRO/"
)  # this is where the raw files are
alan_dir: Path = Path("/data4/vydula/edges/edges3_files/scripts/alan_300_310_tests/")
datadir: Path = Path("/data4/vydula/edges/packages/edges3-data-analysis/data/")

_DEFAULT_TEMP_LOG = (
    "/data5/edges/data/EDGES3_data/MRO/temperature_logger/temperature.log"
)


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


def plot_single_spectrum(data: GSData, alanspec=None, attribute: str = "data") -> None:
    """Plot a single spectrum from GSData, optionally with Alan's spectrum."""
    if alanspec is not None:
        _fig, ax = plt.subplots(
            2,
            1,
            sharex=True,
            gridspec_kw={"hspace": 0, "wspace": 0},
            constrained_layout=True,
            squeeze=False,
        )
    else:
        _fig, ax = plt.subplots(
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
    # Papermill/YAML often inject these as strings
    specyear = int(specyear)
    specday = int(specday)
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

    return min(stem_to_yd, key=lambda s: abs(day_offset(s)))


def find_closest_calkit_stem(data_dir: str, year: int, day: int) -> str | None:
    """Find calkit file stem closest to Antenna S11 date."""
    year = int(year)
    day = int(day)
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

    return min(stem_to_yd, key=lambda s: abs(day_offset(s)))


def extract_temperature(
    file_name,
    load="box",
    extract_log=False,
    temperature_file=datadir / "temperature_data.csv",
):
    """Return the load temperature nearest the time encoded in ``file_name``.

    For hot load, register 102 (hot-load sensor) is used. For amb / open / short,
    register 101 (ambient load) is used; otherwise the front-end temperature is
    returned.
    """
    year = int(file_name[0:4])
    doy = int(file_name[5:8])
    hour = int(file_name[9:11])

    start_time = datetime(year, 1, 1, tzinfo=UTC) + timedelta(days=doy - 1, hours=hour)

    if extract_log:
        # Re-parse the logger only when requested to avoid repeated file parsing.
        temperature_file = extract_temp_values_from_logger(_DEFAULT_TEMP_LOG)

    df = pd.read_csv(temperature_file)
    df["Time"] = pd.to_datetime(df["Time"], utc=True)

    closest_row = df.loc[(df["Time"] - start_time).abs().idxmin()]

    front_end_temp = closest_row["Front End temperature"]
    amb_load_temp = closest_row["Amb load temperature"]
    hot_load_temp = closest_row["Hot load temperature"]

    # hot -> hot_load; amb -> amb_load; open/short/box -> front_end
    if load == "hot":
        temperature = hot_load_temp
    elif load == "amb":
        temperature = amb_load_temp
    else:
        temperature = front_end_temp

    temperature_deg_C = temperature * un.deg_C
    temperature_K = temperature_deg_C.to(un.K, equivalencies=un.temperature())

    return temperature_K.mean()


def filter_nighttime(datetimes, start_date, end_date, start_night=19, end_night=7):
    """Filter datetimes to nighttime observations within a date range.

    Parameters
    ----------
    datetimes
        List of datetime objects.
    start_night
        Start of nighttime in hours (default 19).
    end_night
        End of nighttime in hours (default 7).

    Returns
    -------
    list
        Datetime objects during nighttime.
    """
    nighttime = []
    for dt in datetimes:
        in_range = start_date <= dt <= end_date
        is_night = dt.time() >= time(start_night, 0) or dt.time() < time(end_night, 0)
        if in_range and is_night:
            nighttime.append(dt)
    return nighttime


def extract_dates(anc_obj):
    """Return start/end times from ACQ ancillary data as datetime objects."""
    start_time = anc_obj.data["times"][0][0].decode("utf-8")
    end_time = anc_obj.data["times"][-1][0].decode("utf-8")

    date_format = "%Y:%j:%H:%M:%S"
    start_time = datetime.strptime(start_time, date_format)
    end_time = datetime.strptime(end_time, date_format)

    return (start_time, end_time)


def _try_sensor_float(line: str, prefix: str) -> float | None:
    """Parse ``'<sensor> <value>'`` lines; return float or None."""
    if prefix not in line:
        return None
    values = line.split(" ")
    if len(values) != 2:
        return None
    try:
        return float(values[1].strip())
    except ValueError:
        return None


def extract_temp_values_from_logger(temperature_file=_DEFAULT_TEMP_LOG):
    """Parse the temperature logger into a CSV and return its path.

    Reads one line at a time and keeps only complete sensor records. Incomplete
    records reset the field counter and are discarded.
    """
    df = pd.DataFrame(
        {
            "Time": [],
            "Front End temperature": [],
            "Amb load temperature": [],
            "Hot load temperature": [],
            "Inner box temperature": [],
            "Thermal Control": [],
            "Battery Voltage": [],
            "PR59 Current": [],
        }
    )

    year = doy = 0
    date_object = None
    front_end_box_temp = amb_load_temp = hot_load_temp = None
    innerbox_temp = therm_control = battery_voltage = pr59_current = None
    count = 0

    with Path(temperature_file).open() as file:
        for line in file:
            this_line = line

            if "_" in this_line:
                values = this_line.split("_")  # e.g. 2022_318_03
                if len(values) == 3:
                    count = 0
                    try:
                        year = int(values[0])
                        doy = int(values[1])
                        count += 1
                    except ValueError:
                        pass

            if "UTC" in this_line:
                # Line looks like: "Mon Nov 14 03:20:26 UTC 2022"
                values = this_line.split(" ")
                if len(values) == 6:
                    clock = values[3].split(":")
                    try:
                        hh, mm, ss = int(clock[0]), int(clock[1]), int(clock[2])
                        date_object = datetime(
                            year, 1, 1, hh, mm, ss, tzinfo=UTC
                        ) + timedelta(days=doy - 1)
                        count += 1
                    except ValueError:
                        pass

            # "0 +3.000000e+01" lines are ignored but counted historically
            if "0 " in this_line:
                count += 1

            sensors = {
                "100 ": "front_end_box_temp",
                "101 ": "amb_load_temp",
                "102 ": "hot_load_temp",
                "103 ": "innerbox_temp",
                "106 ": "therm_control",
                "150 ": "battery_voltage",
                "152 ": "pr59_current",
            }
            current = {
                "front_end_box_temp": front_end_box_temp,
                "amb_load_temp": amb_load_temp,
                "hot_load_temp": hot_load_temp,
                "innerbox_temp": innerbox_temp,
                "therm_control": therm_control,
                "battery_voltage": battery_voltage,
                "pr59_current": pr59_current,
            }

            for prefix, name in sensors.items():
                value = _try_sensor_float(this_line, prefix)
                if value is not None:
                    current[name] = value
                    count += 1

            front_end_box_temp = current["front_end_box_temp"]
            amb_load_temp = current["amb_load_temp"]
            hot_load_temp = current["hot_load_temp"]
            innerbox_temp = current["innerbox_temp"]
            therm_control = current["therm_control"]
            battery_voltage = current["battery_voltage"]
            pr59_current = current["pr59_current"]

            if count == 10:
                temp_df = pd.DataFrame(
                    {
                        "Time": [date_object],
                        "Front End temperature": [front_end_box_temp],
                        "Amb load temperature": [amb_load_temp],
                        "Hot load temperature": [hot_load_temp],
                        "Inner box temperature": [innerbox_temp],
                        "Thermal Control": [therm_control],
                        "Battery Voltage": [battery_voltage],
                        "PR59 Current": [pr59_current],
                    }
                )
                df = pd.concat([df, temp_df], ignore_index=True)
                count = 0

    out_path = datadir / "temperature_data.csv"
    df.to_csv(out_path)
    return str(out_path)


def downselect_weatherlog(
    start_date, end_date, path="/data5/edges/data/2014_February_Boolardy/"
):
    """Filter weather-log entries by date range and save them to a text file."""
    input_file_path = Path(path) / "weather2.txt"
    output_path = datadir / f"weather_log_{start_date}_to_{end_date}.txt"
    start_datetime = datetime.strptime(start_date, "%Y_%j")
    end_datetime = datetime.strptime(end_date, "%Y_%j")

    header = (
        "Datetime Rack_Temp(K) Ambient_Temp(K) Ambient_Hum(%) "
        "Frontend_Temp(K) RCV3_LNA_Temp(K)\n"
    )
    with input_file_path.open() as txt_file, output_path.open("w") as out_file:
        out_file.write(header)

        for line in txt_file:
            parts = line.split()
            try:
                entry_datetime = datetime.strptime(parts[0], "%Y:%j:%H:%M:%S")
            except ValueError:
                continue

            if start_datetime <= entry_datetime <= end_datetime:
                out_file.write(
                    f"{parts[0]} {float(parts[2])} {float(parts[5])} "
                    f"{float(parts[8])} {float(parts[11])} {float(parts[14])}\n"
                )

    return output_path


def get_weatherlog_closest_to_time(file_path, input_time):
    """Retrieve the log entry closest to the given input time.

    Parameters
    ----------
    file_path
        Path to the weather log file.
    input_time
        Input time in the format ``%Y:%j:%H:%M:%S``.

    Returns
    -------
    dict or None
        Closest weather log entry, or None if no entries exist.
    """
    input_datetime = datetime.strptime(input_time, "%Y:%j:%H:%M:%S")
    closest_entry = None
    min_time_diff = float("inf")

    with Path(file_path).open() as file:
        headers = file.readline().strip().split()

        for line in file:
            parts = line.strip().split()

            if len(parts) != len(headers):
                continue

            entry_datetime = datetime.strptime(parts[0], "%Y:%j:%H:%M:%S")
            time_diff = abs((entry_datetime - input_datetime).total_seconds())

            if time_diff < min_time_diff:
                min_time_diff = time_diff
                closest_entry = parts

    if closest_entry:
        return {
            headers[i]: float(closest_entry[i]) if i > 1 else closest_entry[i]
            for i in range(len(headers))
        }
    return None
