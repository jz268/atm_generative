import functools
from pathlib import Path
import tqdl

import pandas as pd
import numpy as np
import csv
import sys

import airportsdata as apd

apd_iata = apd.load('IATA')
apd_icao = apd.load('ICAO')


def generate_station_list_csvs():
    for stem in ("ghcnh-station-list", "lcdv2-station-list"):
        input_path = f'data_utils/noaa_lut/{stem}.txt'
        output_path = f'data_utils/noaa_lut/{stem}.csv'
        with open(input_path, 'r') as f_in:
            lines = [line[:80].strip().split(maxsplit=4) + [line[80:].strip()]
                    for line in f_in]
            with open(output_path, 'w') as f_out:
                writer = csv.writer(f_out)
                writer.writerow(('station_id', 'latitude', 'longitude', 'elevation', 'station_name', 'meteostat_id'))
                writer.writerows(lines)

ghcnh_stations = pd.read_csv(Path(Path(__file__).parent, 'noaa_lut/ghcnh-station-list.csv'))
lcdv2_stations = pd.read_csv(Path(Path(__file__).parent, 'noaa_lut/lcdv2-station-list.csv'))

def iata_to_icao(code):
    if code in apd_iata:
        return apd_iata.get(code)['icao']
    raise ValueError("invalid iata code")

def icao_to_iata(code):
    if code in apd_icao:
        return apd_icao.get(code)['iata']
    raise ValueError("invalid icao code")

def loc_to_station_id(dataset, lat, lon):
    stations = ghcnh_stations if dataset == "ghcnh" else lcdv2_stations
    # technically this is wrong but assuming close enough to not matter
    dists = abs(stations['latitude']-lat)+abs(stations['longitude']-lon)
    return stations[dists==dists.min()]['station_id'].values[0]

def iata_to_station_id(dataset, code):
    if code in apd_iata:
        lat = apd_iata.get(code)['lat']
        lon = apd_iata.get(code)['lon']
        return loc_to_station_id(dataset, lat, lon)
    raise ValueError("invalid iata code")

def icao_to_station_id(dataset, code):
    if code in apd_icao:
        lat = apd_icao.get(code)['lat']
        lon = apd_icao.get(code)['lon']
        return loc_to_station_id(dataset, lat, lon)
    raise ValueError("invalid icao code")

loc_to_lcdv2_id = functools.partial(loc_to_station_id, 'lcdv2')
loc_to_ghcnh_id = functools.partial(loc_to_station_id, 'ghcnh')

iata_to_lcdv2_id = functools.partial(iata_to_station_id, 'lcdv2')
iata_to_ghcnh_id = functools.partial(iata_to_station_id, 'ghcnh')

icao_to_lcdv2_id = functools.partial(icao_to_station_id, 'lcdv2')
icao_to_ghcnh_id = functools.partial(icao_to_station_id, 'ghcnh')


class WeatherData:
    def __init__():
        pass



# lcdv2: only csv available
def dl_url_lcdv2(station_id, year):
    return "https://www.ncei.noaa.gov/oa/local-climatological-data/v2/" \
        + f"access/{year}/LCD_{station_id}_{year}.csv"

def out_name_lcdv2(station_name, year):
    return f"LCD_{station_name}_{year}.csv"

# ghcnh: only parquet or psv available
def dl_url_ghcnh(station_id, year, ext="parquet"):
    return "https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/" \
        + f"access/by-year/{year}/{ext}/GHCNh_{station_id}_{year}.{ext}"

def out_name_ghcnh(station_name, year, ext="parquet"):
    return f"GHCNh_{station_name}_{year}.{ext}"  


def dl_noaa(dataset, out_base_dir, station_id, station_name=None, 
            start_year=1987, end_year=2023, ext="parquet",
            merge_yearly=True, purge_yearly=False):
    # dot data starts on 1987, weather starts way before though
    # also default to ignoring 2024 for now since still incomplete
    # mm there is also bulk period of record i didn't know that existed

    if station_name is None:
        station_name = station_id

    if dataset == "lcdv2":
        ext = "csv"
        dl_url_fn = functools.partial(
            dl_url_lcdv2, station_id)
        out_name_fn = functools.partial(
            out_name_lcdv2, station_name)
        merge_name = out_name_lcdv2(
            station_name, f'{start_year}-{end_year}')
    elif dataset == "ghcnh":
        dl_url_fn = functools.partial(
            dl_url_ghcnh, station_id, ext=ext) 
        out_name_fn = functools.partial(
            out_name_ghcnh, station_name, ext=ext)
        merge_name = out_name_ghcnh(
            station_name, f'{start_year}-{end_year}', 'csv')
    else:
        raise ValueError("dataset must be lcdv2 or ghcnh")
    
    if ext == "csv":
        pd_read_fn = pd.read_csv
    elif ext == "psv":
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
    

dl_noaa_lcdv2 = functools.partial(dl_noaa, "lcdv2")
dl_noaa_ghcnh = functools.partial(dl_noaa, "ghcnh")

def dl_noaa_lcdv2_iata(data_dir, code):
    id = iata_to_lcdv2_id(code)
    return dl_noaa_lcdv2(data_dir, id, code)

def dl_noaa_ghcnh_iata(data_dir, code):
    id = iata_to_ghcnh_id(code)
    return dl_noaa_ghcnh(data_dir, id, code)

def dl_noaa_lcdv2_icao(data_dir, code):
    id = icao_to_lcdv2_id(code)
    return dl_noaa_lcdv2(data_dir, id, code)

def dl_noaa_ghcnh_icao(data_dir, code):
    id = iata_to_ghcnh_id(code)
    return dl_noaa_ghcnh(data_dir, id, code)

def dl_noaa_lcdv2_airport(data_dir, code):
    if len(code) == 3:
        return dl_noaa_lcdv2_iata(data_dir, code)
    elif len(code) == 4:
        return dl_noaa_lcdv2_icao(data_dir, code)
    else:
        raise ValueError('invalid airport code?')
    
def dl_noaa_ghcnh_airport(data_dir, code):
    if len(code) == 3:
        return dl_noaa_ghcnh_iata(data_dir, code)
    elif len(code) == 4:
        return dl_noaa_ghcnh_icao(data_dir, code)
    else:
        raise ValueError('invalid airport code?')

# https://www.ncei.noaa.gov/oa/local-climatological-data/v2/doc/lcdv2_DOCUMENTATION.pdf



# uses: https://www.kaggle.com/datasets/zhaodianwen/noaaweatherdatajfkairport/data

def tryconvert(value, dt=None):
    """
    value -> Value to be converted
    dt    -> data type to convert to (redundant for now)
    """
    try:
        return np.float64(value)
    except:
        return np.nan

def clean_noaa_lcdv2_file(data_path, verbose=False):

    data_path = Path(data_path).resolve()

    import_columns = [  
        'DATE',
        'HourlyVisibility',

        'HourlyDryBulbTemperature',
        'HourlyWetBulbTemperature',
        'HourlyDewPointTemperature',

        'HourlyRelativeHumidity',

        'HourlyWindSpeed',
        'HourlyWindDirection',

        'HourlyStationPressure',
        'HourlySeaLevelPressure',
        'HourlyPressureTendency',
        'HourlyAltimeterSetting',

        'HourlyPrecipitation',

        # added (TODO? not sure if relevant)
        # 'HourlyPresentWeatherType',
        # 'HourlySkyConditions',
    ]

# • Snow Depth [cm]
# • Snow Accumulation [cm]

# • Weather Intensity Code
# 1:Light, 2:Moderate, 3:Heavy, 4:Vicinity

# • Weather Descriptor Code
# 1:Shallow, 2:Partial, 3:Patches, 4:Low Drifting, 5:Blowing, 6:Showers, 7:Thunderstorms, 8:Freezing

# • Precipitation Code
# 1:Drizzle, 2:Rain, 3:Snow, 4:Snow Grains, 5:Ice Crystals,
# 6:Ice Pellets, 7:Hail, 8:Small Hail and/or Snow Pellets,
# 9:Unknown Precipitation

# • Obscuration Code
# 1:Mist, 2:Fog, 3:Smoke, 4:Volcanic Ash, 5:Widespread
# Dust, 6:Sand, 7:Haze, 8:Spray

# • Other Weather Code
# 1:Well-Developed Dust/Sand Whirls, 2:Squalls, 3:Funnel
# Cloud, Tornado, Waterspout, 4:Sandstorm, 5:Duststorm

# • Combination Indicator Code
# 1:Not part of combined weather elements, 2:Beginning
# elements of combined weather elements, 3:Combined
# with previous weather element to form a single weather
# report
    
    # Read data and set datetime index
    data_weather = pd.read_csv(data_path, parse_dates=['DATE'], usecols=import_columns)
    data_weather = data_weather.set_index(pd.DatetimeIndex(data_weather['DATE']))
    data_weather.drop(['DATE'], axis=1, inplace=True)
    
    # Replace '*' values with np.nan
    data_weather.replace(to_replace='*', value=np.nan, inplace=True)

    # Replace trace amounts of precipitation with 0
    data_weather['HourlyPrecipitation'].replace(to_replace='T', value='0.00', inplace=True) 
    # Replace rows with tow '.' with np.nan
    data_weather.loc[data_weather['HourlyPrecipitation'].str.count('\.') > 1, 'HourlyPrecipitation'] = np.nan 

    # Convert to float
    for i, _ in enumerate(data_weather.columns):
        data_weather.iloc[:,i] =  data_weather.iloc[:,i].apply(tryconvert)

    # Replace any hourly visibility figure outside these 0-10 bounds
    data_weather.loc[(data_weather['HourlyVisibility'] > 10) | (data_weather['HourlyVisibility'] < 0), 'HourlyVisibility'] = np.nan

    # Downsample to hourly rows 
    data_weather = data_weather.resample('60min').last().shift(periods=1) 

    # Interpolate missing values
    data_weather['HourlyPressureTendency'] = data_weather['HourlyPressureTendency'].fillna(method='ffill') #fill with last valid observation
    data_weather = data_weather.interpolate(method='linear')
    data_weather.drop(data_weather.index[0], inplace=True) #drop first row

    # Transform HourlyWindDirection into a cyclical variable using sin and cos transforms
    data_weather['HourlyWindDirectionSin'] = np.sin(data_weather['HourlyWindDirection'].astype('float')*(2.*np.pi/360))
    data_weather['HourlyWindDirectionCos'] = np.cos(data_weather['HourlyWindDirection'].astype('float')*(2.*np.pi/360))
    # data_weather.drop(['HourlyWindDirection'], axis=1, inplace=True)

    # Transform HourlyPressureTendency into 3 dummy variables based on NOAA documentation
    data_weather['HourlyPressureTendencyIncr'] = [1.0 if x in [0,1,2,3] else 0.0 for x in data_weather['HourlyPressureTendency']] # 0 through 3 indicates an increase in pressure over previous 3 hours
    data_weather['HourlyPressureTendencyDecr'] = [1.0 if x in [5,6,7,8] else 0.0 for x in data_weather['HourlyPressureTendency']] # 5 through 8 indicates a decrease over the previous 3 hours
    data_weather['HourlyPressureTendencyCons'] = [1.0 if x == 4 else 0.0 for x in data_weather['HourlyPressureTendency']] # 4 indicates no change during the previous 3 hours
    # data_weather.drop(['HourlyPressureTendency'], axis=1, inplace=True)
    data_weather['HourlyPressureTendencyIncr'] = data_weather['HourlyPressureTendencyIncr'].astype(('float32'))
    data_weather['HourlyPressureTendencyDecr'] = data_weather['HourlyPressureTendencyDecr'].astype(('float32'))
    data_weather['HourlyPressureTendencyCons'] = data_weather['HourlyPressureTendencyCons'].astype(('float32'))

    # Output csv based on input filename
    data_dir, data_name = data_path.parent, data_path.stem
    data_weather.to_csv(data_dir / Path(data_name + '_CLEANED.csv'), float_format='%g')

    if verbose:
        print("Data successfully cleaned, below are some stats:")
        print('# of megabytes held by dataframe: ' + str(round(sys.getsizeof(data_weather) / 1000000,2)))
        print('# of features: ' + str(data_weather.shape[1])) 
        print('# of observations: ' + str(data_weather.shape[0]))
        print('Start date: ' + str(data_weather.index[0]))
        print('End date: ' + str(data_weather.index[-1]))
        print('# of days: ' + str((data_weather.index[-1] - data_weather.index[0]).days))
        print('# of months: ' + str(round((data_weather.index[-1] - data_weather.index[0]).days/30,2)))
        print('# of years: ' + str(round((data_weather.index[-1] - data_weather.index[0]).days/365,2)))







# perhaps this is not worth dealing with, will look at LCD summaries first

# https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/doc/ghcnh_DOCUMENTATION.pdf

# dtypes_ghcnh_idx = [
#     "Station_ID",
#     "Station_name",
#     "DATE",
#     "Latitude",
#     "Longitude",
#     "Elevation"
# ]

# dtypes_ghcnh_var = [
#     "temperature", 
#     "dew_point_temperature", 
#     "station_level_pressure", 
#     "sea_level_pressure", 
#     "wind_direction", 
#     "wind_speed", 
#     "wind_gust", 
#     "precipitation", 
#     "relative_humidity", 
#     "wet_bulb_temperature", 
#     "pres_wx_MW1", 
#     "pres_wx_MW2", 
#     "pres_wx_MW3", 
#     "pres_wx_AU1", 
#     "pres_wx_AU2", 
#     "pres_wx_AU3",
#     "pres_wx_AW1", 
#     "pres_wx_AW2", 
#     "pres_wx_AW3",
#     "snow_depth", 
#     "visibility", 
#     "altimeter", 
#     "pressure_3hr_change", 
#     "sky_cover_1", 
#     "sky_cover_2", 
#     "sky_cover_3", 
#     "sky_cover_baseht_1", 
#     "sky_cover_baseht_2", 
#     "sky_cover_baseht_3", 
#     "precipitation_3_hour",
#     "precipitation_6_hour",
#     "precipitation_9_hour",
#     "precipitation_12_hour",
#     "precipitation_15_hour",
#     "precipitation_18_hour",
#     "precipitation_21_hour",
#     "precipitation_24_hour",
#     "remarks"
# ]


# dtypes_ghcnh_var_censored = [
#     "temperature", 
#     "dew_point_temperature", 
#     "station_level_pressure", 
#     "sea_level_pressure", 
#     "wind_direction", 
#     "wind_speed", 
#     "wind_gust", 
#     "precipitation", 
#     "relative_humidity", 
#     "wet_bulb_temperature", 
#     # "pres_wx_MW1", # these observations seem to be not reliably there?
#     # "pres_wx_MW2",
#     # "pres_wx_MW3", 
#     # "pres_wx_AU1", 
#     # "pres_wx_AU2", 
#     # "pres_wx_AU3",
#     # "pres_wx_AW1", 
#     # "pres_wx_AW2", 
#     # "pres_wx_AW3",
#     "snow_depth", 
#     "visibility", 
#     "altimeter", 
#     "pressure_3hr_change", 
#     # "sky_cover_1", # probably less relevant going to ignore at least for now
#     # "sky_cover_2", 
#     # "sky_cover_3", 
#     # "sky_cover_baseht_1", 
#     # "sky_cover_baseht_2", 
#     # "sky_cover_baseht_3", 
#     "precipitation_3_hour",
#     "precipitation_6_hour",
#     "precipitation_9_hour",
#     "precipitation_12_hour",
#     "precipitation_15_hour",
#     "precipitation_18_hour",
#     "precipitation_21_hour",
#     "precipitation_24_hour",
#     "remarks"
# ]

# def clean_noaa():

    
