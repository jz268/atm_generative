import pandas as pd
from pathlib import Path

def csv_to_parquet(data_path):

    data_path = Path(data_path).resolve()

    dtype = {
        'station': 'str',
        # 'valid': ,
        'tmpf': 'float', 
        'dwpf': 'float', 
        'relh': 'float',
        'drct': 'float',
        'sknt': 'float', 
        'p01i': 'float', # precipitation issues, maybe get from noaa instead
        'alti': 'float',
        'mslp': 'float',
        'vsby': 'float',
        'gust': 'float',
        'skyc1': 'category', # sky conditions exist in lcd, but harder to parse?
        'skyc2': 'category',
        'skyc3': 'category',
        'skyc4': 'category',
        'skyl1': 'float',
        'skyl2': 'float',
        'skyl3': 'float',
        'skyl4': 'float',
        'wxcodes': 'str', # space sep list
        'feel': 'float',
        # 'ice_accretion_1hr': 'float', # seems to be not much data for lga
        # 'ice_accretion_3hr': 'float',
        # 'ice_accretion_6hr': 'float',
        'peak_wind_gust': 'float', # limited
        'peak_wind_drct': 'float',
        # 'peak_wind_time': ,
        # 'metar': 'str', # raw str, ignore
        # 'snowdepth': 'float' # not many, ignore
    }

    date_cols = ['valid', 'peak_wind_time']
    cols = list(dtype.keys()) + date_cols

    df = pd.read_csv(data_path, engine="pyarrow",
                     usecols=cols, parse_dates=date_cols)
    
    df = df.convert_dtypes().set_index('valid')

    print(df.dtypes)

    df.to_parquet(data_path.parent / f'{data_path.stem}.parquet')
    