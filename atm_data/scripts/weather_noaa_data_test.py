from atm_data.data_utils.weather_noaa_data import *
from pathlib import Path

data_dir = Path(__file__).resolve().parent.parent / 'data/noaa_lcdv2'

# dl_noaa_lcdv2_airport(data_dir, 'LGA')
clean_noaa_lcdv2_file(data_dir / 'LCD_LGA_1987-2023.csv', verbose=True)
