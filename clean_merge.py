import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


airports_iata = ['DEN', 'MDW', 'DAL', 'LAS']

# df_w_s = []

# for ap in airports_iata:
#     df = pd.read_csv(f'data/noaa_lcd/NOAA_LCD_{ap}_2014-2023_cleaned.csv')
#     df['WeatherAirportIATA'] = ap
#     df_w_s.append(df)

# df_w = pd.concat(df_w_s)
# print(df_w)

# df_w.to_csv('data/noaa_lcd/merged_2014-2023.csv')

df_w = pd.read_csv('data/noaa_lcd/merged_2014-2023.csv', parse_dates=['DATE'])
df_s = pd.read_csv('data/schedule/airline_2m_cleaned.csv', 
                    parse_dates=['CRSArrTime', 'ArrTime', 'CRSDepTime', 'DepTime'])

# print(df_w)
# print(df_s)

merged = pd.merge_asof(
    df_s.sort_values('ArrTime'), 
    df_w.sort_values('DATE'), 
    left_by='DestAirportIATA', 
    right_by='WeatherAirportIATA',
    left_on='ArrTime',
    right_on='DATE',
    allow_exact_matches=True, 
    direction='nearest').dropna(
        subset=['HourlyDewPointTemperature'])

print(merged)

# ,OriginAirportIATA,DestAirportIATA,CRSDepTime,DepTime,DepDelay,DepDel15,CRSArrTime,ArrTime,ArrDelay,ArrDel15
# ,DATE,HourlyAltimeterSetting,HourlyDewPointTemperature,HourlyDryBulbTemperature,HourlyPrecipitation,HourlyRelativeHumidity,HourlySeaLevelPressure,HourlyStationPressure,HourlyVisibility,HourlyWetBulbTemperature,HourlyWindSpeed,HourlyWindDirectionSin,HourlyWindDirectionCos,HourlyPressureTendencyIncr,HourlyPressureTendencyDecr,HourlyPressureTendencyCons,WeatherAirportIATA


weather_conditions = ['HourlyWindSpeed', 'HourlyVisibility', 'HourlyDewPointTemperature', 'HourlyPrecipitation']
schedule_results = ['WeatherDelay', 'ArrDelay']

df=merged[weather_conditions+schedule_results].fillna(0)
print(df)
scatter = pd.plotting.scatter_matrix(df, alpha=0.2, figsize=(12, 12), diagonal='kde')
plt.savefig('fig.png')

# df.plot(x='col_name_1', y='col_name_2', style='o')

# df=merged

# df.plot(x='HourlyDewPointTemperature', y='ArrDelay', style='o')
# plt.show()