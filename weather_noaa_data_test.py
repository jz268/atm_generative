from data_utils.weather_noaa_data import *
from pathlib import Path


# data_dir = Path('data/noaa_ghcnh').resolve()

# dl_noaa_ghcnh(data_dir, "USW00023234", "SFO")
# dl_noaa_ghcnh(data_dir, "USW00014739", "BOS")

data_dir = Path('data/noaa_lcdv2').resolve()

# dl_noaa_lcdv2(data_dir, "USW00023234", "SFO")
# dl_noaa_lcdv2(data_dir, "USW00014739", "BOS")

dl_noaa_lcdv2_airport(data_dir, 'LAX')
dl_noaa_lcdv2_airport(data_dir, 'JFK')