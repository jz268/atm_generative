#!/usr/bin/python3

"""
Usage: python3 clean_data.py -f=jfk_weather.csv -v
"""

# source: https://www.kaggle.com/datasets/zhaodianwen/noaaweatherdatajfkairport/data

import sys
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Cleans up NOAA weather data')
parser.add_argument('-f', '--filepath', default='jfk_weather.csv', help='Filepath to NOAA weather data')
parser.add_argument('-v', '--verbose', action='store_true', help='Print verbose')
args = parser.parse_args()

def tryconvert(value, dt=None):
    """
    value -> Value to be converted
    dt    -> data type to convert to (redundant for now)
    """
    try:
        return np.float64(value)
    except:
        return np.nan

def main():
    DATA_FILEPATH = args.filepath

    import_columns = [  'DATE',
                        'HourlyVisibility',
                        'HourlyDryBulbTemperature',
                        'HourlyWetBulbTemperature',
                        'HourlyDewPointTemperature',
                        'HourlyRelativeHumidity',
                        'HourlyWindSpeed',
                        'HourlyWindDirection',
                        'HourlyStationPressure',
                        'HourlyPressureTendency',
                        'HourlySeaLevelPressure',
                        'HourlyPrecipitation',
                        'HourlyAltimeterSetting']
    
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
    data_weather = pd.read_csv(DATA_FILEPATH, parse_dates=['DATE'], usecols=import_columns)
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
        data_weather.iloc[:,i] =  data_weather.iloc[:,i].apply(lambda x: tryconvert(x))

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
    data_weather.drop(['HourlyWindDirection'], axis=1, inplace=True)

    # Transform HourlyPressureTendency into 3 dummy variables based on NOAA documentation
    data_weather['HourlyPressureTendencyIncr'] = [1.0 if x in [0,1,2,3] else 0.0 for x in data_weather['HourlyPressureTendency']] # 0 through 3 indicates an increase in pressure over previous 3 hours
    data_weather['HourlyPressureTendencyDecr'] = [1.0 if x in [5,6,7,8] else 0.0 for x in data_weather['HourlyPressureTendency']] # 5 through 8 indicates a decrease over the previous 3 hours
    data_weather['HourlyPressureTendencyCons'] = [1.0 if x == 4 else 0.0 for x in data_weather['HourlyPressureTendency']] # 4 indicates no change during the previous 3 hours
    data_weather.drop(['HourlyPressureTendency'], axis=1, inplace=True)
    data_weather['HourlyPressureTendencyIncr'] = data_weather['HourlyPressureTendencyIncr'].astype(('float32'))
    data_weather['HourlyPressureTendencyDecr'] = data_weather['HourlyPressureTendencyDecr'].astype(('float32'))
    data_weather['HourlyPressureTendencyCons'] = data_weather['HourlyPressureTendencyCons'].astype(('float32'))

    # Output csv based on input filename
    file_name, extension = args.filepath.split(".")
    data_weather.to_csv(file_name +'_cleaned.csv', float_format='%g')

    if args.verbose:
        print("Data successfully cleaned, below are some stats:")
        print('# of megabytes held by dataframe: ' + str(round(sys.getsizeof(data_weather) / 1000000,2)))
        print('# of features: ' + str(data_weather.shape[1])) 
        print('# of observations: ' + str(data_weather.shape[0]))
        print('Start date: ' + str(data_weather.index[0]))
        print('End date: ' + str(data_weather.index[-1]))
        print('# of days: ' + str((data_weather.index[-1] - data_weather.index[0]).days))
        print('# of months: ' + str(round((data_weather.index[-1] - data_weather.index[0]).days/30,2)))
        print('# of years: ' + str(round((data_weather.index[-1] - data_weather.index[0]).days/365,2)))

if __name__ == "__main__":
    main()