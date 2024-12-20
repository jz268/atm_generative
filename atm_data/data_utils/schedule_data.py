import pandas as pd
import sys
from pathlib import Path
import dask.dataframe as dd
import numpy as np

import airportsdata as apd
# import polars as pl
from calendar import monthrange
import functools
import networkx as nx
import matplotlib.pyplot as plt


# import timezonefinder as tzf
from atm_data.data_utils.tz_utils import get_tz_airport_iata

from tqdm.dask import TqdmCallback
from tqdm import tqdm

cb = TqdmCallback(desc="global")
cb.register()

# apt_ids = pd.read_csv(Path(Path(__file__).parent, 'schedule_lut/L_AIRPORT_ID_IATA_ICAO.csv'))

# def airport_to_id(code):
#     if len(code) == 3:
#         return apt_ids.loc[apt_ids['iata']==code, 'id'].iloc[0]
#     elif len(code) == 4:
#         return apt_ids.loc[apt_ids['icao']==code, 'id'].iloc[0]
#     else:
#         raise ValueError("invalid code?")
    
# # LUT generation stuff?? need to fix the path thing. but also not useful now i think
# def generate_luts():
#     apd_iata = apd.load('IATA')  # key is the IATA location code

#     df_id = pd.read_csv('data_utils/schedule_lut/L_AIRPORT_ID.csv')
#     df_iata = pd.read_csv('data_utils/schedule_lut/L_AIRPORT_IATA.csv')

#     df = pd.merge(df_id, df_iata, how='inner', left_on='Description', right_on='City: Airport')
#     df.drop('Description', inplace=True, axis=1)
#     # df.reset_index(drop=True, inplace=True)

#     df.rename({'Code_x': 'id', 'Code_y': 'iata', 'City: Airport': 'description'}, axis=1, inplace=True)

#     def try_wac_to_icao(code):
#         if code in apd_iata:
#             return apd_iata.get(code)['icao']
#         return None

#     df.insert(loc=2, column='icao', value=df['iata'].apply(try_wac_to_icao))
#     df = df[df.icao.notnull()]

#     print(df)
#     df.to_csv('data_utils/schedule_lut/L_AIRPORT_ID_IATA_ICAO.csv')






# assumes you've downloaded this https://developer.ibm.com/exchanges/data/all/airline/
def ddf_from_ibm(airline_path):

    # # full columns
    # columns = [
    #     'Year', 'Quarter', 'Month', 'DayofMonth', 'DayOfWeek', 'FlightDate',
    #     'Reporting_Airline', 'DOT_ID_Reporting_Airline', 'IATA_CODE_Reporting_Airline',
    #     'Tail_Number', 'Flight_Number_Reporting_Airline',
    #     'Origin', 'Dest',
    #     'CRSDepTime', 'DepTime', 'DepDelay', #'DepDelayMinutes',
    #     'TaxiOut', 'TaxiIn', 
    #     'WheelsOff', 'WheelsOn',
    #     'CRSArrTime', 'ArrTime', 'ArrDelay', #'ArrDelayMinutes',
    #     'Cancelled', 'CancellationCode', 'Diverted',
    #     'CRSElapsedTime', 'ActualElapsedTime', 'AirTime',
    #     'Distance',
    #     'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay',
    #     'DivReachedDest', 'DivActualElapsedTime', 'DivArrDelay', 'DivDistance',
    # ]

    dtype = {
        'Year': 'Int16', 
        'Quarter': 'Int8', 
        'Month': 'Int8', 
        'DayofMonth': 'Int8', 
        'DayOfWeek': 'Int8', 
        'FlightDate': 'str',

        'Reporting_Airline': 'str', # 'category', 
        # 'DOT_ID_Reporting_Airline': 'str', 
        # 'IATA_CODE_Reporting_Airline': 'str',
        'Tail_Number': 'str', 
        'Flight_Number_Reporting_Airline': 'str',

        'Origin': 'str', 
        'Dest': 'str',

        'CRSDepTime': 'Int16', 
        'DepTime': 'Int16', 
        'DepDelay': 'Int16', 
        # 'DepDelayMinutes': 'Int16',

        'TaxiOut': 'Int16', 
        'TaxiIn': 'Int16',
        'WheelsOff': 'str', # 'Int16', bad value, e.g. '0-71'
        'WheelsOn': 'str', # 'Int16', bad value, e.g. '0-55'

        'CRSArrTime': 'Int16', 
        'ArrTime': 'Int16', 
        'ArrDelay': 'Int16', 
        # 'ArrDelayMinutes': 'Int16',

        'Cancelled': 'Int8', # 'boolean', 
        'CancellationCode': 'str', # 'category',

        'CRSElapsedTime': 'Int16', 
        'ActualElapsedTime': 'Int16', 

        'Distance': 'Int16',

        'CarrierDelay': 'Int16', 
        'WeatherDelay': 'Int16', 
        'NASDelay': 'Int16', 
        'SecurityDelay': 'Int16', 
        'LateAircraftDelay': 'Int16',

        'Diverted': 'Int8', # 'boolean', 
        'DivReachedDest': 'Int8', # 'boolean', 
        'DivActualElapsedTime': 'Int16', 
        'DivArrDelay': 'Int16', 
        'DivDistance': 'Int16',
    }

    ddf = dd.read_csv(airline_path, header=0, encoding='latin-1', engine="pyarrow", 
                       dtype=dtype, usecols=list(dtype.keys()))
    
    for col in ('Cancelled', 'Diverted', 'DivReachedDest'):
        ddf[col] = ddf[col].astype('boolean')

    return ddf


def ddf_from_ibm_reduced(airline_path):

    dtype = { 
        'FlightDate': 'string',

        'Origin': 'string', 
        'Dest': 'string',

        'Reporting_Airline': 'string', # 'category',
        'Tail_Number': 'string', 
        'Flight_Number_Reporting_Airline': 'string',

        'CRSDepTime': 'Int16', # 'string', 
        'DepTime': 'Int16', # 'string', 
        'DepDelay': 'Int16',

        'CRSArrTime': 'Int16', # 'string', 
        'ArrTime': 'Int16', # 'string', 
        'ArrDelay': 'Int16',

        'WheelsOff': 'string', # 'Int16', bad value, e.g. '0-71'
        'WheelsOn': 'string', # 'Int16', bad value, e.g. '0-55'

        'CarrierDelay': 'Int16',  # this stuff exists 2003/06 onwards
        'WeatherDelay': 'Int16', 
        'NASDelay': 'Int16', 
        'SecurityDelay': 'Int16', 
        'LateAircraftDelay': 'Int16',

        'Cancelled': 'Int8', # 'boolean', 
        'CancellationCode': 'string', # 'category'

        'Diverted': 'Int8', # 'boolean', 
        # 'DivReachedDest': 'Int8', # 'boolean'
        # 'DivArrDelay': 'Int16',
    }

    ddf = dd.read_csv(airline_path, header=0, encoding='latin-1', engine="pyarrow", 
                       dtype=dtype, usecols=list(dtype.keys()))
    
    for col in ('Cancelled', 'Diverted'):
        ddf[col] = ddf[col].astype('boolean')

    return ddf


def extract_airport_from_ibm_filter(airline_path, airport_iata, out_dir=None, reduced=True):

    # start_year = 1987
    # end_year = 2020

    if out_dir is None:
        out_dir = Path(__file__).parent / 'data'
    else:
        out_dir = Path(out_dir).resolve()
    
    if reduced:
        ddf = ddf_from_ibm_reduced(airline_path)
        tag = 'reduced'
    else:
        ddf = ddf_from_ibm(airline_path)
        tag = 'full' # this is probably not all there rn

    ddf = ddf[(ddf['Origin'] == airport_iata) | (ddf['Dest'] == airport_iata)]

    print(f'processing for {tag}...')
    df = ddf.compute()

    out_stem = f'{airport_iata.lower()}_{tag}_raw'
    df.to_parquet(out_dir / f'{out_stem}.parquet')
    df.to_csv(out_dir / f'{out_stem}.csv', index=False)


# TODO: fix this
def extract_airport_from_ibm_handle_issues(data_path, start_year=1995, end_year=2019):
    data_path = Path(data_path).resolve()
    df = pd.read_parquet(data_path)

    # filter to year range    
    df['FlightDate'] = pd.to_datetime(df['FlightDate'])
    df = df.loc[df['FlightDate'].dt.year.between(start_year, end_year, inclusive='both')]

    # # do somethign about the wheelson wheelsoff issues?
    # df = df.loc[~(df['WheelsOff'].str.contains('n')) & ~(df['WheelsOn'].str.contains('n'))]

    for wheel in ('WheelsOff', 'WheelsOn'):
        df[wheel] = df[wheel].astype('float').astype('Int16')

    # df.dropna(subset=['ArrTime'], inplace=True) # these are probably diverted non-reached, ignore for now?

    delay_cols = ['ArrDelay', 'DepDelay']
    split_delay_cols = ['CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay']
    all_delay_cols = delay_cols + split_delay_cols
    scheduled_cols = ['CRSDepTime', 'CRSArrTime']
    actual_dep_cols = ['DepTime', 'WheelsOff']
    actual_arr_cols = ['ArrTime', 'WheelsOn']
    actual_cols = actual_arr_cols + actual_dep_cols
    time_cols = scheduled_cols + actual_cols

    df.loc[df['ArrDelay'] < 15, split_delay_cols] = 0 # and if all 0 then no sig delay?
    df.loc[df['ArrTime'] == df['CRSArrTime'], all_delay_cols] = 0 # idk why these are NA but ok
    
    zero_delay_mask = (df['ArrDelay'] < 15) | (df['ArrTime'] == df['CRSArrTime'])
    invalid_delay_mask = df['Diverted'] | df['Cancelled']
    nontrivial_split_delay_mask = (~zero_delay_mask) & (~invalid_delay_mask)
    total_split_delays = df[split_delay_cols].sum(axis=1)
    print("\nnontrivial split delays that don't sum to arrival delay:")
    print(df.loc[
        nontrivial_split_delay_mask &
        (df['ArrDelay'] != total_split_delays),
        time_cols + delay_cols + split_delay_cols]
    )
    nsdm_sum = nontrivial_split_delay_mask.sum()
    nsdm_len = len(nontrivial_split_delay_mask)
    print(f'total nontrivial split delays: {nsdm_sum} / {nsdm_len} = {nsdm_sum / nsdm_len}')

    # no actual delay should be this, we treat diverted flights similarly to cancelled?
    # TODO: there is probably a nicer way to handle this
    # one idea is that diverted flights still depart but may or may not make it
    df.loc[df['Diverted'], all_delay_cols] = 9999 
    df.loc[df['Cancelled'], all_delay_cols] = 9999 
    # df.loc[df['Diverted'], actual_arr_cols] = 9999
    df.loc[df['Diverted'] & df['ArrTime'].isna(), actual_arr_cols] = 9999
    df.loc[df['Cancelled'], actual_cols] = 9999

    # print(df.loc[df['WheelsOn'].isna(), time_cols])

    # mostly was just for help in cleaning? the arr dep delays are a tad strange
    # print(df.loc[df['DepDelay'].isna(), time_cols])
    df.drop(['ArrDelay', 'DepDelay'], axis=1, inplace=True)

    # for non-cancelled, not same as missing
    df.loc[~df['Cancelled'], 'CancellationCode'] = 'Z' 

    df = df.sort_values(by=['FlightDate', 'CRSDepTime'])

    string_cols = [
        'Origin', 'Dest', 
        'Tail_Number', 'Flight_Number_Reporting_Airline', 
        'Reporting_Airline', 'CancellationCode'
    ]
    df.replace({col:{'':np.nan} for col in string_cols}, inplace=True)

    print("\nmissing data, before drop:")
    print(pd.concat([df.isna().sum(), df.isna().sum() / len(df)], axis=1))

    ba_cols = [
        'FlightDate', 'Origin', 'Dest', 'Flight_Number_Reporting_Airline', 
        'CRSDepTime', 'DepTime', 'CRSArrTime', 'ArrTime', 
        'WheelsOff', 'WheelsOn', 'Cancelled'
    ]
    # i think the ~0.24% missing ArrTime is maybe diversions that don't make it?
    df.dropna(subset=ba_cols, inplace=True)

    print("\nmissing data, after drop:")
    print(pd.concat([df.isna().sum(), df.isna().sum() / len(df)], axis=1))

    # make categories
    df['Reporting_Airline'] = df['Reporting_Airline'].astype('category')
    df['CancellationCode'] = df['CancellationCode'].astype('category')

    df.reset_index(drop=True, inplace=True)

    print(df.dtypes)

    # manual fixes
    if start_year <= 2004 <= end_year:
        df.loc[
            (df['Flight_Number_Reporting_Airline'] == '5413') &
            (df['FlightDate'] == pd.to_datetime('2004-08-21')),
            'DepTime' 
        ] = 1600
        print(f'DepTime: replacing 160 with 1600 in 2004-08-21 flight 5413 :)')


    out_path_stem = f'{data_path.stem[:-4]}_{start_year}-{end_year}_clean'
    df.to_parquet(data_path.parent / f'{out_path_stem}.parquet')
    df.to_csv(data_path.parent / f'{out_path_stem}.csv', index=False)



def repair_targeted(path, date, flight_number, col, old_value, new_value):
    """
        # repair log (temporary):
        original: 2004-08-21,BNA,LGA,OH,N458CA,5413,1405,160,1720,1955,1637,1946,False
        fix: 160 -> 1600
    """

    df = pd.read_parquet(path)
    target = ((df['FlightDate'] == pd.to_datetime(date)) & 
        (df['Flight_Number_Reporting_Airline'] == flight_number))
    idx_list = df.index[target].tolist()
    assert len(idx_list) == 1, "flight number should be unique in a day"
    idx = idx_list[0]
    
    # safety check
    if old_value == new_value:
        print(f"warning: old_value ({old_value}) same as new_value ({new_value})")

    if df.at[idx, col] == new_value:
        print(f"warning: existing value ({df.at[idx, col]}) same as new_value ({new_value})")

    elif df.at[idx, col] != old_value:
        raise ValueError(
            f"\n   old_value ({old_value}) and existing value ({df.at[idx, col]}) don't match...\n" +
            "   are you sure you're changing the right value?")
    
    # setting value
    df.at[idx, col] = new_value

    print(f'{col}: replacing {old_value} with {new_value} in {date} flight {flight_number} :)')

    df.to_parquet(path)
    df.to_csv(path.with_suffix('.csv'), index=False)





def split_initial_work(data_path, out_dir, time_res):

    if time_res == "yearly":
        split_monthly = False
        split_daily = False
    elif time_res == "monthly":
        split_monthly = True
        split_daily = False
    elif time_res == "daily":
        split_monthly = True
        split_daily = True
    else:
        raise ValueError("time_res must be one of daily, monthly, yearly")

    data_path = Path(data_path).resolve()
    
    out_dir = data_path.parent / f'{data_path.stem}_{time_res}' if out_dir is None else out_dir
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    years = data_path.stem.split('_')[2].split('-')
    start_year = int(years[0])
    end_year = int(years[1])

    out_head = "_".join(data_path.stem.split('_')[:2])
    out_tail = data_path.stem.split('_')[-1]

    out_dir_parquet = out_dir / 'parquet'
    out_dir_csv = out_dir / 'csv'

    out_dir_csv.mkdir(parents=True, exist_ok=True)
    out_dir_parquet.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(data_path)
    df_date = df['FlightDate'].dt

    year_mask = {
        year: (df_date.year == year)
        for year in range(start_year, end_year+1)
    }
    month_mask = {
        month: (df_date.month == month)
        for month in range(1, 13)
    }
    day_mask = {
        day: (df_date.day == day)
        for day in range(1, 32)
    }

    return (
        data_path, out_dir, 
        start_year, end_year, 
        out_head, out_tail, 
        out_dir_parquet, out_dir_csv, 
        df, year_mask, month_mask, day_mask,
        split_monthly, split_daily
    )


def split_by_time_unified(data_path, time_res, out_dir=None): 

    (
        data_path, out_dir, 
        start_year, end_year, 
        out_head, out_tail, 
        out_dir_parquet, out_dir_csv, 
        df, year_mask, month_mask, day_mask,
        split_monthly, split_daily
    ) = \
        split_initial_work(data_path, out_dir, time_res)

    for year in (pbar_year := tqdm(range(start_year, end_year+1), leave=False)):
        pbar_year.set_description(f" year")

        year_df = df.loc[year_mask[year]]

        # split yearly only, don't need to proceed further
        if not split_monthly:
            year_out_stem = f'{out_head}_{year}_{out_tail}'
            year_df.to_parquet(out_dir_parquet / f'{year_out_stem}.parquet')
            year_df.to_csv(out_dir_csv / f'{year_out_stem}.csv', index=False)
            del year_df
            continue

        # split monthly, need to proceed further
        year_out_dir_csv = out_dir_csv / f'{year:04d}'
        year_out_dir_parquet = out_dir_parquet / f'{year:04d}'
        year_out_dir_csv.mkdir(parents=True, exist_ok=True)
        year_out_dir_parquet.mkdir(parents=True, exist_ok=True)
            
        for month in (pbar_month := tqdm(range(1, 13), leave=False)):
            pbar_month.set_description(f"month")

            month_df = year_df.loc[month_mask[month]]

            # split monthly only, don't need to proceed further
            if not split_daily:
                month_out_stem = f'{out_head}_{year}_{month:02d}_{out_tail}'
                month_df.to_parquet(year_out_dir_parquet / f'{month_out_stem}.parquet')
                month_df.to_csv(year_out_dir_csv / f'{month_out_stem}.csv', index=False)
                del month_df
                continue

            # split daily, need to proceed further

            month_out_dir_csv = out_dir_csv / str(year) / f'{month:02d}'
            month_out_dir_parquet = out_dir_parquet / str(year) / f'{month:02d}'
            month_out_dir_csv.mkdir(parents=True, exist_ok=True)
            month_out_dir_parquet.mkdir(parents=True, exist_ok=True)

            _, num_days = monthrange(year, month)

            for day in (pbar_day := tqdm(range(1, num_days+1), leave=False)):
                pbar_day.set_description(f"  day")

                day_df = month_df.loc[day_mask[day]].reset_index(drop=True)

                day_out_stem = f'{out_head}_{year}_{month:02d}_{day:02d}_{out_tail}'
                day_df.to_parquet(month_out_dir_parquet / f'{day_out_stem}.parquet')
                day_df.to_csv(month_out_dir_csv / f'{day_out_stem}.csv', index=False)

                del day_df

            del month_df

        del year_df


split_by_day = functools.partial(split_by_time_unified, time_res='daily')
split_by_month = functools.partial(split_by_time_unified, time_res='monthly')
split_by_year = functools.partial(split_by_time_unified, time_res='yearly')



# we are just assuming the single airport case here, so radial is fine

def visualize_schedule(data_path, out_dir):   
    data_path = Path(data_path).resolve()
    out_dir = Path(out_dir).resolve()
    df = pd.read_parquet(data_path)
    out_path = out_dir / data_path.with_suffix(".png").name
    # print(out_path)
    visualize_schedule_df(df, out_path)

def visualize_schedule_df(df, out_path):
    cols = ['Origin', 'Dest']

    df = df.loc[:,cols]

    multigraph = False

    if multigraph:
        pass
    else:
        df["count"] = 1
        total_flights = len(df)
        df["count"] = df.groupby(cols).transform("sum")
        df = df.drop_duplicates(
                subset=cols, keep="first"
            ).reset_index(drop=True)
        df = df.sort_values(['count'], ascending=False)
        
        G = nx.from_pandas_edgelist(
            df=df, 
            source='Origin', 
            target='Dest', 
            edge_attr='count',
            create_using=nx.DiGraph()
        )        

        plt.clf()
        plt.figure(figsize=(20,16))
        plt.title(f"scheduled flights per route for {out_path.stem}, total: {total_flights}")

        pos=nx.nx_agraph.graphviz_layout(G, prog="twopi", root='LGA') 
        # nx.draw_networkx(G,pos, arrows=True)
        labels = nx.get_edge_attributes(G,'count',)

        d = {}
        for origin, dest, count in list(df.itertuples(index=False, name=None)):
            if origin not in d:
                d[origin] = 0
            if dest not in d:
                d[dest] = 0
            d[origin] += count
            d[dest] += count

        style = 'arc3, rad = 0.03'

        nx.draw(
            G, pos, 
            font_size=10,
            with_labels=True, 
            arrows=True,
            nodelist=[k for k in d], 
            node_size=[100 * d[k]**.9 for k in d],
            connectionstyle=style,
        )

        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=labels, 
            label_pos=.45,
            font_size=10,
            connectionstyle=style,
        )
        
        plt.savefig(out_path)














# # ignore this, let's just adapt to use bayes air dataloader for
# def clean_airport_extracted(data_path, airport_iata, start_year=2000, end_year=2019):

#     data_path = Path(data_path).resolve()

#     df = pd.read_parquet(data_path) 

#     df[df['Year'].between(start_year, end_year, inclusive='both')]
#     df = df.loc[~(df['WheelsOff'].str.contains('n')) & ~(df['WheelsOn'].str.contains('n'))]
#     # okay this is stupid but 
#     for wheel in ('WheelsOff', 'WheelsOn'):
#         df[wheel] = df[wheel].astype('float').astype('Int16')

#     df = df.dropna(subset=[
#         'CRSArrTime', 'ArrTime', 
#         'CRSDepTime', 'DepTime', 
#         'WheelsOff', 'WheelsOn'
#     ])
#     # df = df.drop(['Origin','Dest'], axis=1)

#     print("flight date things...")

#     time_cols = (
#         'CRSArrTime', 'ArrTime', 
#         'CRSDepTime', 'DepTime', 
#         'WheelsOff', 'WheelsOn'
#     )

#     # i think all times are local to time zone? so just need to fix the day when needed
#     for time in time_cols:
#         time_abs = time + 'Absolute'
#         df[time_abs] = df[time].clip(upper=2359)
#         df.loc[df[time_abs]%100==60,time_abs] += 40
#         n = 1945801
#         print(df['FlightDate'].iloc[n]+df[time_abs].astype(str).str.zfill(4).iloc[n])
#         # print(pd.to_datetime(df['FlightDate'].iloc[n]+df[time_abs].astype(str).str.zfill(4).iloc[n],format='%Y-%m-%d%H%M'))
#         df[time_abs] = pd.to_datetime(
#             df['FlightDate']+(df[time_abs].astype(str).str.zfill(4)), format='%Y-%m-%d%H%M')
#         # df = df.drop(time, axis=1)
        
#     df['FlightDate'] = pd.to_datetime(df['FlightDate'], format='%Y-%m-%d')

#     print("dealing with time zones...")

#     ref_tz = get_tz_airport_iata(airport_iata)

#     # adapted from BayesAir dataloader
#     airport_codes = pd.concat(
#         [
#             df["Origin"], 
#             df["Dest"]
#         ]
#     ).unique()

#     # i think the airport lookup part is jsut slow as hell lol 
#     time_zones = [get_tz_airport_iata(code) for code in airport_codes]

#     airport_time_zones = pd.DataFrame(
#         {
#             "airport_code": airport_codes,
#             "time_zone": time_zones
#         }
#     )

#     print("merging origin tz...")

#     df = df.merge(
#         airport_time_zones, 
#         left_on = 'Origin',
#         right_on = 'airport_code',
#     ).rename(columns={'time_zone': 'OriginTimeZone'})

#     print("merging dest tz...")

#     df = df.merge(
#         airport_time_zones, 
#         left_on = 'Dest',
#         right_on = 'airport_code',
#     ).rename(columns={'time_zone': 'DestTimeZone'})
    
#     # at this point, we have df augmented with Origin and Dest time zones

#     # first we convert the hhmm to an actual date and time, not adjusted for timezone yet

#     time_cols_paired = (
#         ('CRSDepTime', 'CRSArrTime'), 
#         ('DepTime', 'ArrTime'),
#         ('WheelsOff', 'WheelsOn'),
#     )

#     print("adjusting for day overflows...")
        
#     time_cols_with_tzs = (
#         ('CRSArrTime', 'DestTimeZone'), ('ArrTime', 'DestTimeZone'),
#         ('CRSDepTime', 'OriginTimeZone'), ('DepTime', 'OriginTimeZone'),
#         ('WheelsOff', 'OriginTimeZone'), ('WheelsOn', 'DestTimeZone'),
#     )
        
#     # here, we take the times and make new absolute col where they're standardized to the ref airport tz
#     for time, time_zone in time_cols_with_tzs:
#         time_abs = time + 'Absolute'
#         # print(df[time_abs].dtypes)
#         # ugh let's just drop the ambiguous for now
#         # df[time_abs] = df[time_abs].dt.tz_localize(
#         #     time_zone, ambiguous='NaT', nonexistent='NaT').dt.tz_convert(ref_tz)
#         df[time_abs] = df.apply(
#             lambda row: row[time_abs].tz_localize(
#                 row[time_zone], ambiguous='NaT', 
#                 nonexistent='NaT').tz_convert(ref_tz),
#             axis=1
#         )
#         df = df.dropna(subset=[time_abs])
        
#     # make absolute hours, i.e. hours past midnight using absolute time
#     # i think we need to use this to process since negatives in the timedeltas are kinda weirdly handled?
#     for time in time_cols:
#         time_hrs = time + 'AbsoluteHours'
#         time_abs = time + 'Absolute'
#         df[time_hrs] = ((df[time_abs] - df[time_abs].dt.normalize()) / pd.Timedelta(hours=1)).astype(int)

#     # when flight crosses midnight, need to add time (for now just detect as arr < dep?)
#     for dep, arr in time_cols_paired:
#         dep_hrs = dep + 'AbsoluteHours'
#         arr_hrs = arr + 'AbsoluteHours'
#         arr_abs = arr + 'Absolute'
#         df.loc[df[arr_hrs] < df[dep_hrs], arr_hrs] += 24
#         df.loc[df[arr_hrs] < df[dep_hrs], arr_abs] += pd.Timedelta(days=1)

#     # adapted from BayesAir dataloader
#     dep_delay = df['DepTimeAbsoluteHours'] - df['DepTimeAbsoluteHours']
#     for col in ('DepTime', 'ArrTime', 'WheelsOff', 'WheelsOn'):
#         col_abs = col + 'Absolute'
#         col_hrs = col + 'AbsoluteHours'
#         df.loc[dep_delay < -3.0, col_hrs] += 24
#         df.loc[dep_delay < -3.0, col_abs] += pd.Timedelta(days=1)

#     # df = df.drop('FlightDate', axis=1)
#     df.reset_index(drop=True, inplace=True)

#     print(df)
#     print( str(round(sys.getsizeof(df) / 1000000,2)) + ' mb')

#     df.to_parquet(data_path.parent / f'{data_path.stem}_cleaned.parquet', index=False)