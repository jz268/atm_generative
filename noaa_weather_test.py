from data_utils.weather_data import *
from pathlib import Path


# data_dir = Path('data/noaa_ghcnh').resolve()

# dl_noaa_ghcnh(data_dir, "USW00023234", "SFO")
# dl_noaa_ghcnh(data_dir, "USW00014739", "BOS")

# data_dir = Path('data/noaa_lcdv2').resolve()

# dl_noaa_lcdv2(data_dir, "USW00023234", "SFO")
# dl_noaa_lcdv2(data_dir, "USW00014739", "BOS")

print(icao_to_lcdv2_id('KSFO'))
print(icao_to_lcdv2_id('KBOS'))
print(icao_to_lcdv2_id('KLAX'))