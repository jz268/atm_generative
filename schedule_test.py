import pandas as pd

import airportsdata
apd_iata = airportsdata.load('IATA')  # key is the IATA location code

df_id = pd.read_csv('data_utils/schedule_lut/L_AIRPORT_ID.csv')
df_iata = pd.read_csv('data_utils/schedule_lut/L_AIRPORT_IATA.csv')

df = pd.merge(df_id, df_iata, how='inner', left_on='Description', right_on='City: Airport')
df.drop('Description', inplace=True, axis=1)

df.rename({'Code_x': 'id', 'Code_y': 'iata', 'City: Airport': 'description'}, axis=1, inplace=True)

def try_wac_to_icao(code):
    if code in apd_iata:
        return apd_iata.get(code)['icao']
    return None

df.insert(loc=2, column='icao', value=df['iata'].apply(try_wac_to_icao))

df = df[df.icao.notnull()]

print(df)

df.to_csv('data_utils/schedule_lut/L_AIRPORT_ID_IATA_ICAO.csv')