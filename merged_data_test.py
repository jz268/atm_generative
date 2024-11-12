import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm


schedule = pd.read_csv('data/schedule/lax_to_jfk_cleaned.csv', 
                 parse_dates=['CRSDepTimeAbsolute', 'CRSArrTimeAbsolute', 
                              'DepTimeAbsolute', 'ArrTimeAbsolute'])
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

# ArrDelay,ArrDelayMinutes,
# CarrierDelay,WeatherDelay,NASDelay,SecurityDelay,LateAircraftDelay,
# DepDelay,DepDelayMinutes,DivDistance,DivArrDelay

# all_cols = schedule.columns.values.tolist() \
#             + weather_lax.columns.values.tolist() \
#             + weather_jfk.columns.values.tolist()

merged_cols = merged.columns.values.tolist()
print(merged_cols)

date_cols = [
    'CRSArrTimeAbsolute', 'ArrTimeAbsolute', 
    'CRSDepTimeAbsolute', 'DepTimeAbsolute',
    'LAX_DATE', 'JFK_DATE', 'FlightDate', 
    'Reporting_Airline' # ignore for now
]

response_cols = [
    'ArrDelay', 'ArrDelayMinutes', 'CarrierDelay', 
    'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay',
    'DepDelay', 'DepDelayMinutes', 'DivDistance', 'DivArrDelay'
]

explanatory_cols = [col for col in merged_cols 
                    if col not in response_cols 
                    and col not in date_cols]

plt.figure()

df = merged

for rcol in tqdm(response_cols, position=0, desc="Y", leave=False):
    for ecol in tqdm(explanatory_cols, position=1, desc="X", leave=False):
        # print(rcol, ecol)
        corr = df[rcol].corr(df[ecol])
        plt.clf()
        plt.title(f'{rcol} vs. {ecol}, ρ={corr:.4f}')
        plt.xlabel(ecol)
        plt.ylabel(rcol)
        plt.scatter(merged[ecol], merged[rcol])
        plt.savefig(f'media/Y={rcol}_X={ecol}.png')

plt.close()
