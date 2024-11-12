import dask.dataframe as dd
import pandas as pd
import airportsdata

from tqdm.dask import TqdmCallback

cb = TqdmCallback(desc="global")
cb.register()

from pathlib import Path
from data_utils.schedule_data import *

# is origina nd dest in iata?


ssd_base_dir = Path('/Volumes/SN850X/').resolve()
airline_path = Path(ssd_base_dir / 'airline.csv')


columns = [
    'Year', 'Quarter', 'Month', 'DayofMonth', 'DayOfWeek', 'FlightDate',
    'Reporting_Airline', 'DOT_ID_Reporting_Airline', 'IATA_CODE_Reporting_Airline',
    'Tail_Number', 'Flight_Number_Reporting_Airline',
    'Origin', 'Dest',
    'CRSDepTime', 'DepTime', 'DepDelay', 'DepDelayMinutes',
    'TaxiOut', 'TaxiIn',
    'CRSArrTime', 'ArrTime', 'ArrDelay', 'ArrDelayMinutes',
    'Cancelled', 'CancellationCode', 'Diverted',
    'CRSElapsedTime', 'ActualElapsedTime', #'AirTime',
    'Distance',
    'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay',
    'DivReachedDest', 'DivActualElapsedTime', 'DivArrDelay', 'DivDistance',
]

dtypes = {col: 'str' for col in columns}

# lax_id = airport_to_id('LAX')
# jfk_id = airport_to_id('JFK')

ddf = dd.read_csv(airline_path, header=0, 
            encoding='latin-1', dtype=dtypes)

ddf = ddf[columns]

ddf = ddf[(ddf['Origin'] == 'LAX') &
          (ddf['Dest'] == 'JFK') ]

df = ddf.compute()

df.to_csv('data/schedule/lax_to_jfk_full.csv')