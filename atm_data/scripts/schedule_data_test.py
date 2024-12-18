import dask.dataframe as dd
import pandas as pd
import airportsdata

from pathlib import Path
from atm_data.data_utils.schedule_data import *


# default
ssd_base_dir = Path('/Volumes/SN850X/').resolve()
airline_path = ssd_base_dir / 'airline.csv'
schedule_data_dir = Path(__file__).resolve().parent.parent / 'data/schedule'
lga_reduced_path = schedule_data_dir / 'lga_reduced_raw.parquet'

start_year = 1995
end_year = 2019
lga_clean_path = schedule_data_dir / f'lga_reduced_{start_year}-{end_year}_clean.parquet'


# extract_airport_from_ibm_filter(airline_path, 'LGA', schedule_data_dir)
# extract_airport_from_ibm_handle_issues(lga_reduced_path, start_year, end_year)
split_by_month(lga_clean_path)

# clean_airport_extracted(schedule_data_dir / 'lga_all_1987-2020.parquet', 'LGA', start_year=2019, end_year=2019)

# df = pd.read_parquet(schedule_data_dir / 'lga_all_1987-2020.parquet')
# df = df.loc[df['Year'].between(2003,2019, inclusive='both')]
# y = len(df)
# # x = len(df[(df['WheelsOff']=='nan')])
# # df = df.replace('nan', pd.nan)
# df = df.loc[(df['WheelsOff'] != 'nan') & (df['WheelsOn'] != 'nan')]
# df = df.dropna(subset=['CRSDepTime', 'CRSArrTime', 'DepTime', 'ArrTime'])
# x = len(df)


# print(f'{x} / {y} = {x/y}')