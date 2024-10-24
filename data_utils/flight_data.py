import pandas as pd
import gzip
import coiled
import dask
import dask.dataframe as dd

from pathlib import Path
import datetime as dt
from datetime import datetime, timezone

# from pyairports.airports import Airports
import airportsdata

import copy


class FlightData:

    def __init__(self, data_dir, base_stem):

        self.data_dir = Path(data_dir)
        self.base_stem = base_stem
        self.file_stem = base_stem + '_LADDfiltered'

        # <facilityName>_<YYYYMMDD>_<StartTimeUTC>_<Durationinsecs>
        stem_list = self.base_stem.split('_')
        self.facility_name = stem_list[0]
        yyyymmdd = stem_list[1]
        year = int(yyyymmdd[:4])
        month = int(yyyymmdd[4:6])
        day = int(yyyymmdd[6:])
        start_time_utc = stem_list[2]
        hour = int(start_time_utc[:2])
        minute = int(start_time_utc[2:4])
        second = int(start_time_utc[4:])
        self.start_datetime = datetime(
            year, month, day, 
            hour, minute, second, 
            tzinfo=timezone.utc)
        self.duration = int(stem_list[3]) # in seconds

        def path_from_prefix(prefix):
            return Path(data_dir, prefix + '_' + self.file_stem + '.csv')

        self.iff_path = path_from_prefix('IFF')
        self.rd_path = path_from_prefix('RD')
        self.ev_path = path_from_prefix('EV')

        # trackpoints data
        self.tp_path = path_from_prefix('TP')
        # merged cleaned data
        self.mc_path = path_from_prefix('MC')
        # resampled data
        self.rs_path = path_from_prefix('RS')

        self.iff_cols = {} # index on recType
        self.iff_cols_idx = {}

        self.iff_cols[3] = [
            'recType',
            'recTime',
            'fltKey',
            'bcnCode',
            'cid',

            'Source',
            'msgType',
            'acId',
            'recTypeCat',
            'coord1',

            'coord2',
            'alt',
            'significance',
            'coord1Accur',
            'coord2Accur',

            'altAccur',
            'groundSpeed',
            'course',
            'rateOfClimb',
            'altQualifier',

            'altIndicator',
            'trackPtStatus',
            'leaderDir',
            'scratchPad',
            'msawInhibitInd',

            'assignedAltString',
            'controllingFac',
            'controllingSec',
            'receivingFac',
            'receivingSec',
            
            'activeContr',
            'primaryContr',
            'kybrdSubset',
            'kybrdSymbol',
            'adsCode',

            'opsType',
            'airportCode',
            'trackNumber',
            'tptReturnType',
            'modeSCode',

            'sensorTrack',
            'spi',
            'dvs',
            'dupM3a',
            'tid',
        ]

        self.iff_cols_idx[3] = {k: v for v, k in enumerate(self.iff_cols[3])}

        self.tp_cols = self.iff_cols[3]
        self.tp_cols_idx = self.iff_cols_idx[3]

        self.mc_dtypes = {
            'recTime':'Float64',
            'fltKey':'Int32',
            'cid':str,
            'acId':str,
            'coord1':'Float64',
            'coord2':'Float64',
            'alt':'Float64',
            'significance':'Int8',
            'groundSpeed':'Int16',
            'course':'Int16',
        }


    # requires IFF
    def make_trackpoints_csv(self, max_significance=6):

        with open(self.iff_path, 'r') as iff_file, open(self.tp_path, 'w+') as tp_file:
            while True:
                lines = iff_file.readlines(50000)
                if not lines:
                    break
                for line in lines:
                    data = line.split(',')
                    if data[0] == '3' and int(data[12]) < max_significance:
                        tp_file.write(line)


    # requires TP and RD
    def make_merged_cleaned_csv(self):

        df_rd = pd.read_csv(self.rd_path, usecols=["Msn", "Orig", "EstOrig", "Dest", "EstDest"])

        # convert all IATA to ICAO... not perfect since somethings inexplicably have numbers attached?
        apd_iata = airportsdata.load('IATA')
        apd_icao = airportsdata.load('ICAO')
        # return same if fail
        ignore_code = '-'
        def standardize_codes(code):
            if len(code) == 3:
                if code in apd_iata:
                    return apd_iata.get(code)['icao']
                return ignore_code
            elif len(code) == 4:
                if code not in apd_icao:
                    return ignore_code
                return code
            elif code == '?':
                return code
            return ignore_code
            
        for col in ['Orig', 'EstOrig', 'Dest', 'EstDest']:
            df_rd[col] = df_rd[col].apply(standardize_codes)

        # for now let's only pick what we know and agree
        def check_code(code):
            return len(code) > 1
        fail_code = 'x'
        def merge_code(code,est_code):
            if (check_code(code) and check_code(est_code)):
                if code == est_code:
                    return code
            if check_code(code):
                return code
            elif check_code(est_code):
                return est_code
            return fail_code
        
        # aggregate estimates and known for orig and dest separately
        for col in ['Orig', 'Dest']:
            est_col = 'Est'+col
            df_rd[col] = df_rd.apply(
                lambda x: merge_code(x[col],x[est_col]),axis=1)
            df_rd.drop(est_col, axis=1,inplace=True)

        total_rd = len(df_rd)
        # throw out the ones we don't know both orig and dest, 
        # also ignore when orig and dest are the same?
        df_rd = df_rd.loc[
            (df_rd['Orig'].str.len() > 1) & 
            (df_rd['Dest'].str.len() > 1) & 
            (df_rd['Orig'] != df_rd['Dest'])]
        accept_rd = len(df_rd)

        df_rd.rename(columns={'Msn':'fltKey','Orig':'orig','Dest':'dest'}, inplace=True)

        # print(df_rd)
        # print(df_rd.loc['KSFO','KBOS'])
        print(f'accepted {accept_rd} / {total_rd}')

        names = [*self.mc_dtypes]
        cleaned_idx = [self.tp_cols_idx[name] for name in names]

        # uhh see what the issue was with the dtypes later
        df_tp = pd.read_csv(self.tp_path, 
                            usecols=cleaned_idx, 
                            names=names,
                            dtype=self.mc_dtypes)
        df_tp.dropna(inplace=True)
        # df_tp = df_tp.astype(clean_dtypes)

        df_tp.set_index(['fltKey'], inplace=True)

        # default inner join -- ?
        df = pd.merge(df_rd, df_tp, on='fltKey')
        print(len(df_tp), len(df))
        # print(df)

        df.set_index(['orig','dest','fltKey'], inplace=True)
        df.to_csv(self.mc_path)


    # requires MC
    def make_resampled_csv(self):
        
        df = pd.read_csv(self.mc_path, dtype=self.mc_dtypes)
        df.set_index(['orig','dest','fltKey'], inplace=True)
        df['recTime'] = pd.to_datetime(df['recTime'], unit='s')

        print(df)