import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm



# # # clean lax to jfk, deal with tghis later

# dtype = {
#     'CRSArrTime':'Int16',
#     'ArrTime':'Int16',
#     'CRSDepTime':'Int16',
#     'DepTime':'Int16',
#     'ArrDelay':'Int16',
#     'DepDelay':'Int16',
# }

# df = pd.read_csv('data/schedule/lax_to_jfk.csv', dtype=dtype)
# df = df.dropna(subset=['CRSArrTime', 'ArrTime', 'CRSDepTime', 'DepTime'])
# df = df.drop(['Origin','Dest'], axis=1)

# # i think all times are local to time zone? so just need to fix the day when needed
# for time in ['CRSArrTime', 'ArrTime', 'CRSDepTime', 'DepTime']:
#     time_abs = time + 'Absolute'
#     df[time_abs] = df[time].apply(lambda x: min(2359, x)) # roudn down i guess
#     # n = 738
#     # print(df['FlightDate'].iloc[n]+df[time_abs].astype(str).str.zfill(4).iloc[n])
#     df[time_abs] = pd.to_datetime(
#         df['FlightDate']+(df[time_abs].astype(str).str.zfill(4)), format='%Y-%m-%d%H%M')

# # if arr < dep time, then arrival is next day?
# for dep, arr in [('CRSDepTime', 'CRSArrTime'), ('DepTime', 'ArrTime')]:
#     dep_abs, arr_abs = dep + 'Absolute', arr + 'Absolute'
#     df[arr_abs] = df.apply(lambda row: 
#                     row[arr_abs] if row[dep] < row[arr]
#                     else row[arr_abs] + pd.Timedelta(days=1), axis=1)
            
# for time, time_zone in \
#     [('CRSArrTime', 'America/New_York'), ('ArrTime', 'America/New_York'),
#      ('CRSDepTime', 'America/Los_Angeles'), ('DepTime', 'America/Los_Angeles')]:
#     time_abs = time + 'Absolute'
#     print(df[time_abs].dtypes)
#     df[time_abs] = df[time_abs].dt.tz_localize(time_zone, ambiguous='infer').dt.tz_convert("UTC")

# # df = df.drop('FlightDate', axis=1)
# df.reset_index(drop=True, inplace=True)

# print(df)
# print( str(round(sys.getsizeof(df) / 1000000,2)) + ' mb')

# df.to_csv('data/schedule/lax_to_jfk_cleaned.csv', index=False)



# Month,DayOfWeek,Reporting_Airline,
# CRSDepTime,CRSArrTime,DepTime,ArrTime,
# ArrDelay,ArrDelayMinutes,
# CarrierDelay,WeatherDelay,NASDelay,SecurityDelay,LateAircraftDelay,
# DepDelay,DepDelayMinutes,DivDistance,DivArrDelay

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
