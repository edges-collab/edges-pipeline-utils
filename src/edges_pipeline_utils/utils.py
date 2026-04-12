"""Various utilities for use in notebooks."""

import re
import sys
from importlib.metadata import version
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time
from pygsdata import GSData
import astropy.units as un



root_dir: Path = Path('/data5/edges/data/EDGES3_data/MRO/') # this is where the raw files are
alan_dir: Path = Path("/data4/vydula/edges/edges3_files/scripts/alan_300_310_tests/")
datadir: Path = Path("/data4/vydula/edges/packages/edges3-data-analysis/data/")

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



def extract_temperature(
        file_name,
        load="box",
        extract_log=False,
        temperature_file=datadir / "temperature_data.csv",
):
    """
    Take start and end time from the ancillary data and return the average temperature in that time range
    --  For hot load, temperature from hot load temperature sensor is used (register 102) that
    sits directly on the hot load at the end of 8 position switch
    --  For amb, long cable open and short, the register 101 is used which is the temperature of the ambient load
    """

    # get the datetime object using file name

    year = int(file_name[0:4])
    doy = int(file_name[5:8])
    hour = int(file_name[9:11])

    start_time = datetime(year, 1, 1) + timedelta(days=doy - 1, hours=hour)

    if extract_log == True:
        """
        extract the data from log only if required.
        Default is False, meaning the pre-extracted file will be used
        This is don to avoid repeted file parsing

        """
        temperature_file = extract_temp_values_from_logger(
            "/data5/edges/data/EDGES3_data/MRO/temperature_logger/temperature.log"
        )

        # default file is temperature_data.csv

    df = pd.read_csv(temperature_file)

    df["Time"] = pd.to_datetime(df["Time"])

    closest_row = df.loc[(df["Time"] - start_time).abs().idxmin()]

    front_end_temp = closest_row["Front End temperature"]
    amb_load_temp = closest_row["Amb load temperature"]
    hot_load_temp = closest_row["Hot load temperature"]
    inner_box_temp = closest_row["Inner box temperature"]
    therm_control = closest_row["Thermal Control"]
    battery_voltage = closest_row["Battery Voltage"]
    battery_current = closest_row["PR59 Current"]

    if load == "hot":
        temperature = hot_load_temp
        # print('hot', temperature)

    elif load == "open" or "short" or "box" or "amb":
        temperature = amb_load_temp  # this is default if no load is specified

    if load == "amb":
        temperature = amb_load_temp
        # print('amb', temperature)

    elif load == "hot":
        temperature = hot_load_temp
        # print('hot', temperature)

    elif load == "open" or "short" or "box":
        temperature = front_end_temp  # this is default if no load is specified
        # print('default', temperature)

    temperature_deg_C = temperature * un.deg_C
    temperature_K = temperature_deg_C.to(un.K, equivalencies=un.temperature())

    return temperature_K.mean()


def filter_nighttime(datetimes, start_date, end_date, start_night=19, end_night=7):
    """
    Filters a list of datetime objects to include only those corresponding to night time observations

    Args:
        datetimes (list): List of datetime objects.
        start_night (float): start of the night time, default is 19 hr
        end_night (float): start of the night time, default is 7 hr

    Returns:
        list: List of datetime objects during nighttime.
    """
    nighttime = []
    for dt in datetimes:

        if dt >= start_date and dt <= end_date:
            if dt.time() >= time(start_night, 0) or dt.time() < time(end_night, 0):
                nighttime.append(dt)
    return nighttime


def extract_dates(anc_obj):
    """
    Take ancilliary data from acq file and return the start and end time as datetime objects.

    """

    start_time = anc_obj.data["times"][0][0].decode("utf-8")  # first instance of time
    end_time = anc_obj.data["times"][-1][0].decode("utf-8")  # second instance of time

    date_format = "%Y:%j:%H:%M:%S"

    # Parse the string into a datetime object
    start_time = datetime.strptime(start_time, date_format)

    end_time = datetime.strptime(end_time, date_format)

    return (start_time, end_time)


def extract_temp_values_from_logger(
        temperature_file="/data5/edges/data/EDGES3_data/MRO/temperature_logger/temperature.log",
):
    """
    Easiest way (i think) is to read one line at a time, check conditions in each line,
    extract a datetime object and temperature readouts.
    I am throwing away the readouts that don't have complete information.
    If such read out is encountered, count is 'reset' meaning that readout won't be appended to the dataframe

    """

    # lets first create a panda dataframe

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

    with open(temperature_file) as file:
        for line in file:
            this_line = line
            # print(this_line)

            if "_" in this_line:
                # print(this_line)

                values = this_line.split(
                    "_"
                )  # this is the line that looks like 2022_318_03
                if len(values) == 3:
                    count = 0  # reset everytime the first line is encountered

                    try:
                        year = int(values[0])
                        doy = int(values[1])
                        count += 1
                    except:
                        # skip this data record
                        count = count  # do nothing

            if "UTC" in this_line:
                values = this_line.split(
                    " "
                )  # this_line is alredy split with single space

                # this is the line that looks like "Mon Nov 14 03:20:26 UTC 2022"
                # we only want time from this
                if len(values) == 6:
                    time = values[3]
                    values = time.split(
                        ":"
                    )  # we take the time that looks like "03:20:26"

                    try:
                        hh = int(values[0])
                        mm = int(values[1])
                        ss = int(values[2])

                        # create datetime object using year, day of the year, hh,mm,ss information.

                        date_object = datetime(year, 1, 1, hh, mm, ss) + timedelta(
                            days=doy - 1
                        )
                        count += 1

                    except:
                        # skip this data record
                        count = count  # do nothing

            if (
                    "0 " in this_line
            ):  # this is the line that looks like "0 +3.000000e+01" --> we don't need this now
                count += 1  # do nothing

            if "100 " in this_line:
                values = this_line.split(" ")  # this line is for sensor 100
                if len(values) == 2:
                    # check if there are two values in that line -- one for the sensor and one for the value

                    try:
                        front_end_box_temp = float(values[1].strip())
                        count += 1
                    except:
                        # skip this data record
                        count = count  # do nothing

            if "101 " in this_line:
                values = this_line.split(" ")  # this line is for sensor 101
                if (
                        len(values) == 2
                ):  # check if there are two values in that line -- one for the sensor and one for the value
                    try:
                        amb_load_temp = float(values[1].strip())
                        count += 1

                    except ValueError:
                        # skip this data record
                        count = count  # do nothing

            if "102 " in this_line:
                values = this_line.split(" ")  # this line is for sensor 102
                if len(values) == 2:
                    try:
                        hot_load_temp = float(values[1].strip())
                        count += 1

                    except ValueError:
                        # skip this data record
                        count = count  # do nothing

            if "103 " in this_line:
                values = this_line.split(" ")  # this line is for sensor 103
                if len(values) == 2:
                    try:
                        innerbox_temp = float(values[1].strip())
                        count += 1

                    except ValueError:
                        # skip this data record
                        count = count  # do nothing

            if "106 " in this_line:
                values = this_line.split(" ")  # this line is for sensor 106
                if len(values) == 2:
                    try:
                        therm_control = float(values[1].strip())
                        count += 1

                    except ValueError:
                        # skip this data record
                        count = count  # do nothing

            if "150 " in this_line:
                values = this_line.split(" ")  # this is battery voltage 150
                if len(values) == 2:
                    try:
                        battery_voltage = float(values[1].strip())
                        count += 1

                    except ValueError:
                        # skip this data record
                        count = count  # do nothing

            if "152 " in this_line:
                values = this_line.split(" ")  # this is PR59 current 152
                if len(values) == 2:
                    try:
                        pr59_current = float(values[1].strip())
                        count += 1

                    except ValueError:
                        # skip this data record
                        count = count  # do nothing

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

                # Concatenate the this DataFrame with the original DataFrame
                df = pd.concat([df, temp_df], ignore_index=True)
                count = 0  # reset

    # for now I am saving it where the function is called, but we could find a more organized way of saving this file

    df.to_csv(f"{datadir}/temperature_data.csv")

    return f"{datadir}/temperature_data.csv"


def extract_temperature(
        file_name,
        load="box",
        extract_log=False,
        temperature_file=datadir / "temperature_data.csv",
):
    """
    Take start and end time from the ancillary data and return the average temperature in that time range
    --  For hot load, temperature from hot load temperature sensor is used (register 102) that
    sits directly on the hot load at the end of 8 position switch
    --  For amb, long cable open and short, the register 101 is used which is the temperature of the ambient load
    """

    # get the datetime object using file name

    year = int(file_name[0:4])
    doy = int(file_name[5:8])
    hour = int(file_name[9:11])

    start_time = datetime(year, 1, 1) + timedelta(days=doy - 1, hours=hour)

    if extract_log == True:
        """
        extract the data from log only if required.
        Default is False, meaning the pre-extracted file will be used
        This is don to avoid repeted file parsing

        """
        temperature_file = extract_temp_values_from_logger(
            "/data5/edges/data/EDGES3_data/MRO/temperature_logger/temperature.log"
        )

        # default file is temperature_data.csv

    df = pd.read_csv(temperature_file)

    df["Time"] = pd.to_datetime(df["Time"])

    closest_row = df.loc[(df["Time"] - start_time).abs().idxmin()]

    front_end_temp = closest_row["Front End temperature"]
    amb_load_temp = closest_row["Amb load temperature"]
    hot_load_temp = closest_row["Hot load temperature"]
    inner_box_temp = closest_row["Inner box temperature"]
    therm_control = closest_row["Thermal Control"]
    battery_voltage = closest_row["Battery Voltage"]
    battery_current = closest_row["PR59 Current"]

    if load == "hot":
        temperature = hot_load_temp
        # print('hot', temperature)

    elif load == "open" or "short" or "box" or "amb":
        temperature = amb_load_temp  # this is default if no load is specified

    if load == "amb":
        temperature = amb_load_temp
        # print('amb', temperature)

    elif load == "hot":
        temperature = hot_load_temp
        # print('hot', temperature)

    elif load == "open" or "short" or "box":
        temperature = front_end_temp  # this is default if no load is specified
        # print('default', temperature)

    temperature_deg_C = temperature * un.deg_C
    temperature_K = temperature_deg_C.to(un.K, equivalencies=un.temperature())

    return temperature_K.mean()


# weather log is a big file so lets try to downselect to the days that we need


def downselect_weatherlog(start_date, end_date, path='/data5/edges/data/2014_February_Boolardy/'):
    """
    Filters entries in the weather log text file by date range and saves them to a CSV file.

    """

    input_file_path = path + 'weather2.txt'

    output_path = datadir / f"weather_log_{start_date}_to_{end_date}.txt"
    start_datetime = datetime.strptime(start_date, "%Y_%j")
    end_datetime = datetime.strptime(end_date, "%Y_%j")

    with open(input_file_path, 'r') as txt_file, open(output_path, 'w') as out_file:

        out_file.write("Datetime Rack_Temp(K) Ambient_Temp(K) Ambient_Hum(%) Frontend_Temp(K) RCV3_LNA_Temp(K)\n")

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
    """
    Retrieve the log entry closest to the given input time.

    Parameters:
        file_path (str): Path to the weather log file.
        input_time (str): Input time in the format "%Y:%j:%H:%M:%S".

    Returns:
        dict: A dictionary containing the closest weather log entry, or None if no entries exist.
    """
    from datetime import datetime

    input_datetime = datetime.strptime(input_time, "%Y:%j:%H:%M:%S")
    closest_entry = None
    min_time_diff = float('inf')

    with open(file_path, 'r') as file:
        headers = file.readline().strip().split()  # Read headers

        for line in file:
            parts = line.strip().split()

            if len(parts) != len(headers):
                print(f"Skipping malformed line: {line.strip()}")
                continue

            entry_datetime = datetime.strptime(parts[0], "%Y:%j:%H:%M:%S")
            time_diff = abs((entry_datetime - input_datetime).total_seconds())

            if time_diff < min_time_diff:
                min_time_diff = time_diff
                closest_entry = parts

    if closest_entry:
        # print(f"Closest entry: {closest_entry}")
        # Construct dictionary from the closest entry
        return {
            headers[i]: float(closest_entry[i]) if i > 1 else closest_entry[i]
            for i in range(len(headers))
        }
    else:
        return None

