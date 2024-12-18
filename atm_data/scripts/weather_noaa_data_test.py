from atm_data.data_utils.weather_noaa_data import *
from pathlib import Path

data_dir = Path(__file__).resolve().parent.parent / 'data/noaa_lcdv2'

# dl_noaa_lcdv2_airport(data_dir, 'LGA')

# data_path = data_dir / 'LCD_LGA_1987-2023.csv'
# clean_noaa_lcdv2_file(data_path, verbose=True)

cleaned_path = data_dir / 'LCD_LGA_1987-2023_CLEANED.parquet'
import pandas as pd
df = pd.read_parquet(cleaned_path)
print(df)
