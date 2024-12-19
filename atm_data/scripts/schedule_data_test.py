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

start_year = 2004
end_year = 2019
lga_clean_path = schedule_data_dir / f'lga_reduced_{start_year}-{end_year}_clean.parquet'

# extract_airport_from_ibm_filter(airline_path, 'LGA', schedule_data_dir)
# extract_airport_from_ibm_handle_issues(lga_reduced_path, start_year, end_year)

split_by_day(lga_clean_path)
split_by_month(lga_clean_path)
split_by_year(lga_clean_path)

# repair_path = schedule_data_dir / f'lga_reduced_{start_year}-{end_year}_clean.parquet'
# repair_targeted(repair_path, '2004-08-21', '5413', 'DepTime', 160, 1600)