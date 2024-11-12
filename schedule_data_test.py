import dask.dataframe as dd
import pandas as pd
import airportsdata

from tqdm.dask import TqdmCallback

cb = TqdmCallback(desc="global")
cb.register()

from pathlib import Path
from data_utils.schedule_data import *


# TODO: GENERALIZE THIS STUFF 


# # FOR EXTRACTING PART OF FULL DATSET -- TODO: CLEAN THIS UP LATER
# # is origina nd dest in iata?
# ssd_base_dir = Path('/Volumes/SN850X/').resolve()
# airline_path = Path(ssd_base_dir / 'airline.csv')

# columns = [
#     'Year', 'Quarter', 'Month', 'DayofMonth', 'DayOfWeek', 'FlightDate',
#     'Reporting_Airline', 'DOT_ID_Reporting_Airline', 'IATA_CODE_Reporting_Airline',
#     'Tail_Number', 'Flight_Number_Reporting_Airline',
#     'Origin', 'Dest',
#     'CRSDepTime', 'DepTime', 'DepDelay', 'DepDelayMinutes',
#     'TaxiOut', 'TaxiIn',
#     'CRSArrTime', 'ArrTime', 'ArrDelay', 'ArrDelayMinutes',
#     'Cancelled', 'CancellationCode', 'Diverted',
#     'CRSElapsedTime', 'ActualElapsedTime', #'AirTime',
#     'Distance',
#     'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay',
#     'DivReachedDest', 'DivActualElapsedTime', 'DivArrDelay', 'DivDistance',
# ]

# dtypes = {col: 'str' for col in columns}

# # lax_id = airport_to_id('LAX')
# # jfk_id = airport_to_id('JFK')

# ddf = dd.read_csv(airline_path, header=0, 
#             encoding='latin-1', dtype=dtypes)
# ddf = ddf[columns]
# ddf = ddf[(ddf['Origin'] == 'LAX') &
#           (ddf['Dest'] == 'JFK') ]

# df = ddf.compute()
# df.to_csv('data/schedule/lax_to_jfk_full.csv')





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





# # CLEAN FULL LAX TO JFK

dtype = {
    'Year': 'Int16', 
    'Quarter': 'Int8', 
    'Month': 'Int8', 
    'DayofMonth': 'Int8', 
    'DayOfWeek': 'Int8', 
    'FlightDate': 'str',

    'Reporting_Airline': 'str', 
    # 'DOT_ID_Reporting_Airline': 'str', 
    # 'IATA_CODE_Reporting_Airline': 'str',
    # 'Tail_Number': 'str', 
    # 'Flight_Number_Reporting_Airline': 'str',
    # 'Origin': 'str', 
    # 'Dest': 'str',

    'CRSDepTime': 'Int16', 
    'DepTime': 'Int16', 
    'DepDelay': 'Int16', 
    'DepDelayMinutes': 'Int16',

    # 'TaxiOut': 'Int16', 
    # 'TaxiIn': 'Int16',

    'CRSArrTime': 'Int16', 
    'ArrTime': 'Int16', 
    'ArrDelay': 'Int16', 
    'ArrDelayMinutes': 'Int16',

    'Cancelled': 'Int8', 
    'CancellationCode': 'str', 

    'CRSElapsedTime': 'Int16', 
    'ActualElapsedTime': 'Int16', 

    # 'Distance': 'Int16',

    'CarrierDelay': 'Int16', 
    'WeatherDelay': 'Int16', 
    'NASDelay': 'Int16', 
    'SecurityDelay': 'Int16', 
    'LateAircraftDelay': 'Int16',

    'Diverted': 'Int8',
    'DivReachedDest': 'Int16', 
    'DivActualElapsedTime': 'Int16', 
    'DivArrDelay': 'Int16', 
    'DivDistance': 'Int16',
}

df = pd.read_csv('data/schedule/lax_to_jfk_full.csv', dtype=dtype, usecols=list(dtype.keys()))
df = df.dropna(subset=['CRSArrTime', 'ArrTime', 'CRSDepTime', 'DepTime'])
# df = df.drop(['Origin','Dest'], axis=1)

# i think all times are local to time zone? so just need to fix the day when needed
for time in ['CRSArrTime', 'ArrTime', 'CRSDepTime', 'DepTime']:
    time_abs = time + 'Absolute'
    df[time_abs] = df[time].apply(lambda x: min(2359, x)) # roudn down i guess
    # n = 738
    # print(df['FlightDate'].iloc[n]+df[time_abs].astype(str).str.zfill(4).iloc[n])
    df[time_abs] = pd.to_datetime(
        df['FlightDate']+(df[time_abs].astype(str).str.zfill(4)), format='%Y-%m-%d%H%M')

# if arr < dep time, then arrival is next day?
for dep, arr in [('CRSDepTime', 'CRSArrTime'), ('DepTime', 'ArrTime')]:
    dep_abs, arr_abs = dep + 'Absolute', arr + 'Absolute'
    df[arr_abs] = df.apply(lambda row: 
                    row[arr_abs] if row[dep] < row[arr]
                    else row[arr_abs] + pd.Timedelta(days=1), axis=1)
            
for time, time_zone in \
    [('CRSArrTime', 'America/New_York'), ('ArrTime', 'America/New_York'),
     ('CRSDepTime', 'America/Los_Angeles'), ('DepTime', 'America/Los_Angeles')]:
    time_abs = time + 'Absolute'
    print(df[time_abs].dtypes)
    # ugh deal with ambiguous later let's just guess it's forward?
    df[time_abs] = df[time_abs].dt.tz_localize(
        time_zone, ambiguous='NaT', nonexistent='shift_forward').dt.tz_convert("UTC")

# df = df.drop('FlightDate', axis=1)
df.reset_index(drop=True, inplace=True)

print(df)
print( str(round(sys.getsizeof(df) / 1000000,2)) + ' mb')

df.to_csv('data/schedule/lax_to_jfk_full_cleaned.csv', index=False)