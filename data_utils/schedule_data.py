import pandas as pd
import sys
from pathlib import Path


apt_ids = pd.read_csv(Path(Path(__file__).parent, 'schedule_lut/L_AIRPORT_ID_IATA_ICAO.csv'))

def airport_to_id(code):
    if len(code) == 3:
        return apt_ids.loc[apt_ids['iata']==code, 'id'].iloc[0]
    elif len(code) == 4:
        return apt_ids.loc[apt_ids['icao']==code, 'id'].iloc[0]
    else:
        raise ValueError("invalid code?")
    
# # LUT generation stuff??

# apd_iata = airportsdata.load('IATA')  # key is the IATA location code

# df_id = pd.read_csv('data_utils/schedule_lut/L_AIRPORT_ID.csv')
# df_iata = pd.read_csv('data_utils/schedule_lut/L_AIRPORT_IATA.csv')

# df = pd.merge(df_id, df_iata, how='inner', left_on='Description', right_on='City: Airport')
# df.drop('Description', inplace=True, axis=1)
# # df.reset_index(drop=True, inplace=True)

# df.rename({'Code_x': 'id', 'Code_y': 'iata', 'City: Airport': 'description'}, axis=1, inplace=True)

# def try_wac_to_icao(code):
#     if code in apd_iata:
#         return apd_iata.get(code)['icao']
#     return None

# df.insert(loc=2, column='icao', value=df['iata'].apply(try_wac_to_icao))

# df = df[df.icao.notnull()]

# print(df)

# df.to_csv('data_utils/schedule_lut/L_AIRPORT_ID_IATA_ICAO.csv')



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




