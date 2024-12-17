import pandas as pd
import sys
from pathlib import Path
import dask.dataframe as dd

import airportsdata as apd
# import polars as pl

# import timezonefinder as tzf
from data_utils.tz_utils import get_tz_airport_iata

from tqdm.dask import TqdmCallback

cb = TqdmCallback(desc="global")
cb.register()

# apt_ids = pd.read_csv(Path(Path(__file__).parent, 'schedule_lut/L_AIRPORT_ID_IATA_ICAO.csv'))

# def airport_to_id(code):
#     if len(code) == 3:
#         return apt_ids.loc[apt_ids['iata']==code, 'id'].iloc[0]
#     elif len(code) == 4:
#         return apt_ids.loc[apt_ids['icao']==code, 'id'].iloc[0]
#     else:
#         raise ValueError("invalid code?")
    
# # LUT generation stuff?? need to fix the path thing. but also not useful now i think
# def generate_luts():
#     apd_iata = apd.load('IATA')  # key is the IATA location code

#     df_id = pd.read_csv('data_utils/schedule_lut/L_AIRPORT_ID.csv')
#     df_iata = pd.read_csv('data_utils/schedule_lut/L_AIRPORT_IATA.csv')

#     df = pd.merge(df_id, df_iata, how='inner', left_on='Description', right_on='City: Airport')
#     df.drop('Description', inplace=True, axis=1)
#     # df.reset_index(drop=True, inplace=True)

#     df.rename({'Code_x': 'id', 'Code_y': 'iata', 'City: Airport': 'description'}, axis=1, inplace=True)

#     def try_wac_to_icao(code):
#         if code in apd_iata:
#             return apd_iata.get(code)['icao']
#         return None

#     df.insert(loc=2, column='icao', value=df['iata'].apply(try_wac_to_icao))
#     df = df[df.icao.notnull()]

#     print(df)
#     df.to_csv('data_utils/schedule_lut/L_AIRPORT_ID_IATA_ICAO.csv')






# assumes you've downloaded this https://developer.ibm.com/exchanges/data/all/airline/
def ddf_from_ibm(airline_path):

    # # full columns
    # columns = [
    #     'Year', 'Quarter', 'Month', 'DayofMonth', 'DayOfWeek', 'FlightDate',
    #     'Reporting_Airline', 'DOT_ID_Reporting_Airline', 'IATA_CODE_Reporting_Airline',
    #     'Tail_Number', 'Flight_Number_Reporting_Airline',
    #     'Origin', 'Dest',
    #     'CRSDepTime', 'DepTime', 'DepDelay', #'DepDelayMinutes',
    #     'TaxiOut', 'TaxiIn', 
    #     'WheelsOff', 'WheelsOn',
    #     'CRSArrTime', 'ArrTime', 'ArrDelay', #'ArrDelayMinutes',
    #     'Cancelled', 'CancellationCode', 'Diverted',
    #     'CRSElapsedTime', 'ActualElapsedTime', 'AirTime',
    #     'Distance',
    #     'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay',
    #     'DivReachedDest', 'DivActualElapsedTime', 'DivArrDelay', 'DivDistance',
    # ]

    dtype = {
        'Year': 'Int16', 
        'Quarter': 'Int8', 
        'Month': 'Int8', 
        'DayofMonth': 'Int8', 
        'DayOfWeek': 'Int8', 
        'FlightDate': 'str',

        'Reporting_Airline': 'str', # 'category', 
        # 'DOT_ID_Reporting_Airline': 'str', 
        # 'IATA_CODE_Reporting_Airline': 'str',
        'Tail_Number': 'str', 
        'Flight_Number_Reporting_Airline': 'str',

        'Origin': 'str', 
        'Dest': 'str',

        'CRSDepTime': 'Int16', 
        'DepTime': 'Int16', 
        'DepDelay': 'Int16', 
        # 'DepDelayMinutes': 'Int16',

        'TaxiOut': 'Int16', 
        'TaxiIn': 'Int16',
        'WheelsOff': 'str', # 'Int16', bad value, e.g. '0-71'
        'WheelsOn': 'str', # 'Int16', bad value, e.g. '0-55'

        'CRSArrTime': 'Int16', 
        'ArrTime': 'Int16', 
        'ArrDelay': 'Int16', 
        # 'ArrDelayMinutes': 'Int16',

        'Cancelled': 'Int8', # 'boolean', 
        'CancellationCode': 'str', # 'category',

        'CRSElapsedTime': 'Int16', 
        'ActualElapsedTime': 'Int16', 

        'Distance': 'Int16',

        'CarrierDelay': 'Int16', 
        'WeatherDelay': 'Int16', 
        'NASDelay': 'Int16', 
        'SecurityDelay': 'Int16', 
        'LateAircraftDelay': 'Int16',

        'Diverted': 'Int8', # 'boolean', 
        'DivReachedDest': 'Int8', # 'boolean', 
        'DivActualElapsedTime': 'Int16', 
        'DivArrDelay': 'Int16', 
        'DivDistance': 'Int16',
    }

    ddf = dd.read_csv(airline_path, header=0, encoding='latin-1', engine="pyarrow", 
                       dtype=dtype, usecols=list(dtype.keys()))
    
    for col in ('Cancelled', 'Diverted', 'DivReachedDest'):
        ddf[col] = ddf[col].astype('boolean')

    return ddf


def ddf_from_ibm_reduced(airline_path):

    dtype = {
        'Year': 'Int16', 
        'Month': 'Int8', 
        'DayofMonth': 'Int8', 
        'FlightDate': 'str',

        'Reporting_Airline': 'str', # 'category',
        'Tail_Number': 'str', 
        'Flight_Number_Reporting_Airline': 'str',

        'Origin': 'str', 
        'Dest': 'str',

        'CRSDepTime': 'Int16', 
        'DepTime': 'Int16', 

        'WheelsOff': 'str', # 'Int16', bad value, e.g. '0-71'
        'WheelsOn': 'str', # 'Int16', bad value, e.g. '0-55'

        'CRSArrTime': 'Int16', 
        'ArrTime': 'Int16', 

        'Cancelled': 'Int8', # 'boolean', 
        'Diverted': 'Int8', # 'boolean', 
        'DivReachedDest': 'Int8', # 'boolean', 
    }

    ddf = dd.read_csv(airline_path, header=0, encoding='latin-1', engine="pyarrow", 
                       dtype=dtype, usecols=list(dtype.keys()))
    
    for col in ('Cancelled', 'Diverted', 'DivReachedDest'):
        ddf[col] = ddf[col].astype('boolean')

    return ddf


def extract_airport_from_ibm_filter(airline_path, airport_iata, out_dir=None, reduced=True):

    start_year = 1987
    end_year = 2020

    if out_dir is None:
        out_dir = Path(__file__).parent / 'data'
    else:
        out_dir = Path(out_dir).resolve()
    
    if reduced:
        ddf = ddf_from_ibm_reduced(airline_path)
        tag = 'all'
    else:
        ddf = ddf_from_ibm(airline_path)
        tag = 'full'

    ddf = ddf[(ddf['Origin'] == airport_iata) | (ddf['Dest'] == airport_iata)]

    print(f'processing for {tag}...')
    df = ddf.compute()
    df.to_parquet(out_dir / f'{airport_iata.lower()}_{tag}_{start_year}-{end_year}_raw.parquet')


def extract_airport_from_ibm_handle_issues(data_path, start_year=2000, end_year=2019):
    data_path = Path(data_path).resolve()
    df = pd.read_parquet(data_path)

    # filter to year range    
    df = df[df['Year'].between(start_year, end_year, inclusive='both')]

    df = df.loc[~(df['WheelsOff'].str.contains('n')) & ~(df['WheelsOn'].str.contains('n'))]

    for wheel in ('WheelsOff', 'WheelsOn'):
        df[wheel] = df[wheel].astype('float').astype('Int16')

    out_path_stem = data_path.stem[:-4]
    df.to_parquet(data_path.parent / f'{out_path_stem}.parquet')




def clean_airport_extracted(data_path, airport_iata, start_year=2000, end_year=2019):

    data_path = Path(data_path).resolve()

    df = pd.read_parquet(data_path) 

    df[df['Year'].between(start_year, end_year, inclusive='both')]
    df = df.loc[~(df['WheelsOff'].str.contains('n')) & ~(df['WheelsOn'].str.contains('n'))]
    # okay this is stupid but 
    for wheel in ('WheelsOff', 'WheelsOn'):
        df[wheel] = df[wheel].astype('float').astype('Int16')

    df = df.dropna(subset=[
        'CRSArrTime', 'ArrTime', 
        'CRSDepTime', 'DepTime', 
        'WheelsOff', 'WheelsOn'
    ])
    # df = df.drop(['Origin','Dest'], axis=1)

    print("flight date things...")

    time_cols = (
        'CRSArrTime', 'ArrTime', 
        'CRSDepTime', 'DepTime', 
        'WheelsOff', 'WheelsOn'
    )

    # i think all times are local to time zone? so just need to fix the day when needed
    for time in time_cols:
        time_abs = time + 'Absolute'
        df[time_abs] = df[time].clip(upper=2359)
        df.loc[df[time_abs]%100==60,time_abs] += 40
        n = 1945801
        print(df['FlightDate'].iloc[n]+df[time_abs].astype(str).str.zfill(4).iloc[n])
        # print(pd.to_datetime(df['FlightDate'].iloc[n]+df[time_abs].astype(str).str.zfill(4).iloc[n],format='%Y-%m-%d%H%M'))
        df[time_abs] = pd.to_datetime(
            df['FlightDate']+(df[time_abs].astype(str).str.zfill(4)), format='%Y-%m-%d%H%M')
        # df = df.drop(time, axis=1)
        
    df['FlightDate'] = pd.to_datetime(df['FlightDate'], format='%Y-%m-%d')

    print("dealing with time zones...")

    ref_tz = get_tz_airport_iata(airport_iata)

    # adapted from BayesAir dataloader
    airport_codes = pd.concat(
        [
            df["Origin"], 
            df["Dest"]
        ]
    ).unique()

    # i think the airport lookup part is jsut slow as hell lol 
    time_zones = [get_tz_airport_iata(code) for code in airport_codes]

    airport_time_zones = pd.DataFrame(
        {
            "airport_code": airport_codes,
            "time_zone": time_zones
        }
    )

    print("merging origin tz...")

    df = df.merge(
        airport_time_zones, 
        left_on = 'Origin',
        right_on = 'airport_code',
    ).rename(columns={'time_zone': 'OriginTimeZone'})

    print("merging dest tz...")

    df = df.merge(
        airport_time_zones, 
        left_on = 'Dest',
        right_on = 'airport_code',
    ).rename(columns={'time_zone': 'DestTimeZone'})
    
    # at this point, we have df augmented with Origin and Dest time zones

    # first we convert the hhmm to an actual date and time, not adjusted for timezone yet

    time_cols_paired = (
        ('CRSDepTime', 'CRSArrTime'), 
        ('DepTime', 'ArrTime'),
        ('WheelsOff', 'WheelsOn'),
    )

    print("adjusting for day overflows...")
        
    time_cols_with_tzs = (
        ('CRSArrTime', 'DestTimeZone'), ('ArrTime', 'DestTimeZone'),
        ('CRSDepTime', 'OriginTimeZone'), ('DepTime', 'OriginTimeZone'),
        ('WheelsOff', 'OriginTimeZone'), ('WheelsOn', 'DestTimeZone'),
    )
        
    # here, we take the times and make new absolute col where they're standardized to the ref airport tz
    for time, time_zone in time_cols_with_tzs:
        time_abs = time + 'Absolute'
        # print(df[time_abs].dtypes)
        # ugh let's just drop the ambiguous for now
        # df[time_abs] = df[time_abs].dt.tz_localize(
        #     time_zone, ambiguous='NaT', nonexistent='NaT').dt.tz_convert(ref_tz)
        df[time_abs] = df.apply(
            lambda row: row[time_abs].tz_localize(
                row[time_zone], ambiguous='NaT', 
                nonexistent='NaT').tz_convert(ref_tz),
            axis=1
        )
        df = df.dropna(subset=[time_abs])
        
    # make absolute hours, i.e. hours past midnight using absolute time
    # i think we need to use this to process since negatives in the timedeltas are kinda weirdly handled?
    for time in time_cols:
        time_hrs = time + 'AbsoluteHours'
        time_abs = time + 'Absolute'
        df[time_hrs] = ((df[time_abs] - df[time_abs].dt.normalize()) / pd.Timedelta(hours=1)).astype(int)

    # when flight crosses midnight, need to add time (for now just detect as arr < dep?)
    for dep, arr in time_cols_paired:
        dep_hrs = dep + 'AbsoluteHours'
        arr_hrs = arr + 'AbsoluteHours'
        arr_abs = arr + 'Absolute'
        df.loc[df[arr_hrs] < df[dep_hrs], arr_hrs] += 24
        df.loc[df[arr_hrs] < df[dep_hrs], arr_abs] += pd.Timedelta(days=1)

    # adapted from BayesAir dataloader
    dep_delay = df['DepTimeAbsoluteHours'] - df['DepTimeAbsoluteHours']
    for col in ('DepTime', 'ArrTime', 'WheelsOff', 'WheelsOn'):
        col_abs = col + 'Absolute'
        col_hrs = col + 'AbsoluteHours'
        df.loc[dep_delay < -3.0, col_hrs] += 24
        df.loc[dep_delay < -3.0, col_abs] += pd.Timedelta(days=1)

    # df = df.drop('FlightDate', axis=1)
    df.reset_index(drop=True, inplace=True)

    print(df)
    print( str(round(sys.getsizeof(df) / 1000000,2)) + ' mb')

    df.to_parquet(data_path.parent / f'{data_path.stem}_cleaned.parquet', index=False)