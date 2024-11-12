import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm


dtype = {
    'Year': 'Int16', 
    'Quarter': 'Int8', 
    'Month': 'Int8', 
    'DayofMonth': 'Int8', 
    'DayOfWeek': 'Int8', 
    'FlightDate': 'str',

    'Reporting_Airline': 'str', 

    'CRSDepTime': 'Int16', 
    'DepTime': 'Int16', 
    'DepDelay': 'Int16', 
    'DepDelayMinutes': 'Int16',

    'CRSArrTime': 'Int16', 
    'ArrTime': 'Int16', 
    'ArrDelay': 'Int16', 
    'ArrDelayMinutes': 'Int16',

    'CarrierDelay': 'Int16', 
    'WeatherDelay': 'Int16', 
    'NASDelay': 'Int16', 
    'SecurityDelay': 'Int16', 
    'LateAircraftDelay': 'Int16',

    'Diverted': 'Int8',
    'DivArrDelay': 'Int16', 
    'DivDistance': 'Int16',
}

abs_time_cols = ['CRSDepTimeAbsolute', 'CRSArrTimeAbsolute', 
                 'DepTimeAbsolute', 'ArrTimeAbsolute']

cols = list(dtype.keys()) + abs_time_cols

schedule = pd.read_csv('data/schedule/lax_to_jfk_full_cleaned.csv', 
                dtype=dtype, usecols=cols, parse_dates=abs_time_cols)
merged = schedule

weather_lax = pd.read_csv('data/noaa_lcdv2/LCD_LAX_1987-2023_CLEANED.csv', 
                          parse_dates=['DATE']).rename(columns=lambda x:f'LAX_{x}')
weather_jfk = pd.read_csv('data/noaa_lcdv2/LCD_JFK_1987-2023_CLEANED.csv', 
                          parse_dates=['DATE']).rename(columns=lambda x:f'JFK_{x}')

merged = pd.merge_asof(
    merged.sort_values('DepTimeAbsolute'), 
    weather_lax.sort_values('LAX_DATE'), 
    # left_by='DestAirportIATA', 
    # right_by='WeatherAirportIATA',
    left_on='DepTimeAbsolute',
    right_on='LAX_DATE',
    allow_exact_matches=True, 
    direction='nearest')

# merged.to_csv('test_0.csv')

merged = pd.merge_asof(
    merged.sort_values('ArrTimeAbsolute'), 
    weather_jfk.sort_values('JFK_DATE'), 
    # left_by='DestAirportIATA', 
    # right_by='WeatherAirportIATA',
    left_on='ArrTimeAbsolute',
    right_on='JFK_DATE',
    allow_exact_matches=True, 
    direction='nearest')

# merged.to_csv('test_1.csv')

merged = merged.fillna(0)

merged_cols = merged.columns.values.tolist()
print(merged_cols)

date_cols = [
    'CRSArrTimeAbsolute', 'ArrTimeAbsolute', 
    'CRSDepTimeAbsolute', 'DepTimeAbsolute',
    'LAX_DATE', 'JFK_DATE', 'FlightDate', 
    'Reporting_Airline' # ignore for now
]

response_cols = [
    'ArrDelay', 'ArrDelayMinutes', 'DepDelay', 'DepDelayMinutes',
    'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay', 
    'Diverted', 'DivDistance', 'DivArrDelay',
]

explanatory_cols = [col for col in merged_cols 
                    if col not in response_cols 
                    and col not in date_cols]

plt.figure()

df = merged

for rcol in tqdm(response_cols, position=0, desc="Y", leave=False):
    for ecol in tqdm(explanatory_cols, position=1, desc="X", leave=False):
        # print(rcol, ecol)
        plt.clf()
        # corr = df[rcol].corr(df[ecol])
        # plt.title(f'{rcol} vs. {ecol}, ρ={corr:.4f}')
        plt.title(f'{rcol} vs. {ecol}')
        plt.xlabel(ecol)
        plt.ylabel(rcol)
        plt.scatter(merged[ecol], merged[rcol])
        plt.savefig(f'media/Y={rcol}_X={ecol}.png')

plt.close()
