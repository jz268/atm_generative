from atm_data.data_utils.weather_metar_data import *
from pathlib import Path

metar_data_dir = Path(__file__).resolve().parent.parent / 'data/iem_metar'

# csv_full_path = metar_data_dir / 'metar_lga_1987-2023.csv'
# csv_to_parquet(csv_full_path)

lga_full_path = metar_data_dir / 'lga_all_1987-2023.parquet'

s = '-RA:02 BR:1 |RA BR |RA'