from data_utils.weather_metar_data import *
from pathlib import Path

metar_data_dir = Path('data/iem_metar').resolve()

csv_full_path = metar_data_dir / 'lga_all_1987-2020.csv'
csv_to_parquet(csv_full_path)

lga_full_path = metar_data_dir / 'lga_all_1987-2020.parquet'