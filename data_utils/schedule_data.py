import pandas as pd
import sys
from pathlib import Path


apt_codes = pd.read_csv(Path(Path(__file__).parent, 'schedule_lut/L_AIRPORT_ID_IATA_ICAO'))

# this is not great fix rhis
def clean_airline_2m(data_path):

    data_path = Path(data_path).resolve()

    cols = [
        'FlightDate', 'Year', 'Month', 'DayofMonth', 'Quarter', 'DayOfWeek', 
        'Reporting_Airline', 'OriginAirportID', 'DestAirportID',
        'CRSArrTime', 'ArrTime', 'ArrDelay', 'ArrDel15', 
        'CRSDepTime', 'DepTime', 'DepDelay', 'DepDel15',
        'CRSElapsedTime', 'ActualElapsedTime',
        'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay',
        'Cancelled', 'CancellationCode', 'Diverted',
    ] 

    dtype = {
        'CRSArrTime':'Int64',
        'ArrTime':'Int64',
        'CRSDepTime':'Int64',
        'DepTime':'Int64',
        'ArrDelay':'Int64',
        'DepDelay':'Int64',
        'ArrDel15':'Int8',
        'DepDel15':'Int8'
    }

    las_id = 12889
    mdw_id = 13232
    den_id = 11292
    dal_id = 11259

    airport_ids = [las_id, mdw_id, den_id, dal_id]
    airport_id_iata = {
        las_id: 'LAS',
        mdw_id: 'MDW',
        den_id: 'DEN',
        dal_id: 'DAL',
    }
    airport_iata_id = {
        'LAS': las_id,
        'MDW': mdw_id,
        'DEN': den_id,
        'DAL': dal_id,
    }
    def id_to_iata(id):
        return airport_id_iata[id]

    def clip_time(hhmm):
        return min(2359, hhmm) # round down i guess

    # southwest_id = 'WN'

    df = pd.read_csv(data_path, usecols=cols, dtype=dtype, encoding='latin-1')
    df = df.dropna(subset=['CRSArrTime', 'ArrTime', 'CRSDepTime', 'DepTime'])

    df = df.loc[(df['Year'] >= 2014) & (df['Year'] <= 2023)].drop('Year', axis=1)
    # df = df.loc[df['Reporting_Airline'] == southwest_id].drop('Reporting_Airline', axis=1)
    df = df.loc[(df['OriginAirportID'].isin(airport_ids)) & (df['DestAirportID'].isin(airport_ids))]

    df['OriginAirportID'] = df['OriginAirportID'].apply(id_to_iata)
    df['DestAirportID'] = df['DestAirportID'].apply(id_to_iata)
    df.rename(columns={'OriginAirportID': 'OriginAirportIATA', 'DestAirportID': 'DestAirportIATA'}, inplace=True)

    for time in ['CRSArrTime', 'ArrTime', 'CRSDepTime', 'DepTime']:
        df[time] = df[time].apply(clip_time)
        # print(df[time].astype(str).str.zfill(4).iloc[131])
        df[time] = pd.to_datetime(
            df['FlightDate']+(df[time].astype(str).str.zfill(4)), format='%Y-%m-%d%H%M')
    df = df.drop('FlightDate', axis=1)
    df.reset_index(drop=True, inplace=True)

    print(df)
    print( str(round(sys.getsizeof(df) / 1000000,2)) + ' mb')

    data_dir, data_name = data_path.parent, data_path.stem
    df.to_csv(data_dir / Path(data_name + '_CLEANED.csv'))




