from data_utils.weather_data import dl_noaa_GHCNh
from pathlib import Path


data_dir = Path('data/noaa_ghcnh').resolve()

dl_noaa_GHCNh(data_dir, "USW00023234", "SFO")