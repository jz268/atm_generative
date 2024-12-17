from data_utils.weather_noaa_data import *
from pathlib import Path

data_dir = Path('../data/noaa_lcdv2').resolve()

# dl_noaa_lcdv2_airport(data_dir, 'LAX')
# dl_noaa_lcdv2_airport(data_dir, 'JFK')

# clean_noaa_lcdv2_file(data_dir / 'LCD_LAX_1987-2023.csv', verbose=True)
# clean_noaa_lcdv2_file(data_dir / 'LCD_JFK_1987-2023.csv', verbose=True)

# dl_noaa_lcdv2_airport(data_dir, 'LGA')
clean_noaa_lcdv2_file(data_dir / 'LCD_LGA_1987-2023.csv', verbose=True)