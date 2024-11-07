import functools
from pathlib import Path
import tqdl

import pandas as pd


class WeatherData:
    def __init__():
        pass



# LCDv2: only csv available
def dl_url_LCDv2(station_id, year):
    return "https://www.ncei.noaa.gov/oa/local-climatological-data/v2/" \
        + f"access/{year}/LCD_{station_id}_{year}.csv"

def out_name_LCDv2(station_name, year):
    return f"LCD_{station_name}_{year}.csv"

# GHCNh: only parquet or psv available
def dl_url_GHCNh(station_id, year, ext="parquet"):
    return "https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/" \
        + f"access/by-year/{year}/{ext}/GHCNh_{station_id}_{year}.{ext}"

def out_name_GHCNh(station_name, year, ext="parquet"):
    return f"GHCNh_{station_name}_{year}.{ext}"  


def dl_noaa(dataset, out_base_dir, station_id, station_name=None, 
            start_year=1987, end_year=2023, ext=None, 
            merge_yearly=True, purge_yearly=False):
    # dot data starts on 1987, weather starts way before though
    # also default to ignoring 2024 for now since still incomplete
    # mm there is also bulk period of record i didn't know that existed

    if station_name is None:
        station_name = station_id

    if dataset == "LCDv2":
        ext = "csv"
        dl_url_fn = functools.partial(
            dl_url_LCDv2, station_id)
        out_name_fn = functools.partial(
            out_name_LCDv2, station_name)
        merge_name = out_name_LCDv2(
            station_name, f'{start_year}-{end_year}', 'csv')
    elif dataset == "GHCNh":
        ext = "parquet" if ext is None else ext
        dl_url_fn = functools.partial(
            dl_url_GHCNh, station_id, ext=ext) 
        out_name_fn = functools.partial(
            out_name_GHCNh, station_name, ext=ext)
        merge_name = out_name_GHCNh(
            station_name, f'{start_year}-{end_year}', 'csv')
    else:
        raise ValueError("dataset must be LCDv2 or GHCNh")
    
    if ext == "csv":
        pd_read_fn = pd.read_csv
    if ext == "psv":
        pd_read_fn = functools.partial(pd.read_csv, sep='|')
    elif ext == "parquet":
        pd_read_fn = pd.read_parquet
    else:
        raise ValueError("only csv, psv, parquet")
    
    # this is slightly assuming we're just doing the download and merge once
    out_dir = Path(out_base_dir, f'{station_name}')
    out_dir.mkdir(exist_ok=True) # add parents=True if needed
    out_path_fn = lambda year: Path(out_dir, out_name_fn(year))

    for year in range(start_year, end_year+1):
        out_path = out_path_fn(year)
        if not out_path.is_file():
            # print(f"{year} dataset downloading...")
            tqdl.download(dl_url_fn(year), str(out_path))
        else:
            print(f"{station_id} {year} dataset already downloaded, skipping...")

    if merge_yearly:
        df_yearly_s = [pd_read_fn(out_path_fn(year)) 
                       for year in range(start_year, end_year+1)]
        df = pd.concat(df_yearly_s, ignore_index=True)#.dropna(axis=1, how='all')
        print(df)
        df.to_csv(Path(out_base_dir, merge_name))

    if purge_yearly:    
        for year in range(start_year, end_year+1):
            Path.unlink(out_path, missing_ok=True)
        Path.rmdir(out_dir)
    

dl_noaa_LCDv2 = functools.partial(dl_noaa, "LCDv2")
dl_noaa_GHCNh = functools.partial(dl_noaa, "GHCNh")

# https://www.ncei.noaa.gov/oa/local-climatological-data/v2/doc/lcdv2_DOCUMENTATION.pdf


# https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/doc/ghcnh_DOCUMENTATION.pdf

dtypes_GHCNh_idx = [
    "Station_ID",
    "Station_name",
    "DATE",
    "Latitude",
    "Longitude",
    "Elevation"
]

dtypes_GHCNh_var = [
    "temperature", 
    "dew_point_temperature", 
    "station_level_pressure", 
    "sea_level_pressure", 
    "wind_direction", 
    "wind_speed", 
    "wind_gust", 
    "precipitation", 
    "relative_humidity", 
    "wet_bulb_temperature", 
    "pres_wx_MW1", 
    "pres_wx_MW2", 
    "pres_wx_MW3", 
    "pres_wx_AU1", 
    "pres_wx_AU2", 
    "pres_wx_AU3",
    "pres_wx_AW1", 
    "pres_wx_AW2", 
    "pres_wx_AW3",
    "snow_depth", 
    "visibility", 
    "altimeter", 
    "pressure_3hr_change", 
    "sky_cover_1", 
    "sky_cover_2", 
    "sky_cover_3", 
    "sky_cover_baseht_1", 
    "sky_cover_baseht_2", 
    "sky_cover_baseht_3", 
    "precipitation_3_hour",
    "precipitation_6_hour",
    "precipitation_9_hour",
    "precipitation_12_hour",
    "precipitation_15_hour",
    "precipitation_18_hour",
    "precipitation_21_hour",
    "precipitation_24_hour",
    "remarks"
]

# def clean_noaa():

    
