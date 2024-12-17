import requests
import json
from ratelimit import limits, sleep_and_retry
from pprint import pprint
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import functools

# link = "https://simwxdata.avmet.com/scores?object=airport,airspace,area,sector&type=Forecast,NextDay,PostOp&timing=Daily,Hourly,QuarterHourly&date=2017-01-01"

list_str = lambda l: ','.join(l)
date_path = lambda dir, date, ext: Path(dir / f'simwx_{date}.{ext}')

def download_scores(data_dir, start_date, end_date):

    url = "https://simwxdata.avmet.com/scores"
    params = {
        'object': list_str(['region','airport','airspace','area','sector']),
        'type': list_str(['Forecast','NextDay','PostOp']),
        'timing': list_str(['Daily','Hourly','QuarterHourly']),
        'date': start_date
    }
    # default inclusive
    dates = pd.date_range(start_date,end_date,freq='d').strftime("%Y-%m-%d").tolist()

    c, p = 1, 3
    @sleep_and_retry
    @limits(calls=c, period=p)
    def download_score(date):
        params['date'] = date
        response = requests.get(url=url, params=params)
        file_path = date_path(data_dir, date, 'json')
        with open(file_path, "w") as f:
            f.write(response.text)
        # print(response.url.replace('%2C', ','))
    print(f"rate limited to {c} calls per {p} seconds...")

    for date in tqdm(dates):
        download_score(date)
    


def download_videos(data_dir, start_date, end_date):

    dates = pd.date_range(start_date,end_date,freq='d').strftime("%Y-%m-%d").tolist()

    c, p = 1, 1
    @sleep_and_retry
    @limits(calls=c, period=p)
    def download_video(date):
        year, month, day = date[:4], date[5:7], date[8:10]
        url = f'https://data.avmet.com/simwx/static/data/videos/{year}/{month}/{year}{month}{day}.mp4'
        content = requests.get(url).content
        file_path = date_path(data_dir, date, 'mp4')
        with open(file_path, 'wb') as f:
            f.write(content)
    print(f"rate limited to {c} calls per {p} seconds...")

    for date in tqdm(dates):
        download_video(date)




def read_scores(data_dir, start_date, end_date):
    
    dates = pd.date_range(start_date,end_date,freq='d').strftime("%Y-%m-%d").tolist()

    for date in tqdm(dates):
        file_path = date_path(data_dir, date, 'json')
        with open(file_path) as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        """
            so essentially we have

            forecast (including nextday):
                airspace,airports: 
                    hourly: h-F,(H-P),N
                    daily: d-F,D,N
                region: 
                    hourly: HR-F,P,N(alt:FHR-N)
                    daily: DR-F,P,N

            postop:
                airspace,airports: 
                    quarterhourly: QH-P
                    hourly: H-P
                    daily: d-P
                area: 
                    hourly: HA-P
                    daily: DA-P
                sector: 
                    hourly: HS-P
                    daily: DS-P

            so we should do:
                (1) change FHR-N records to HR-N
                (2) combine things for each of the time/space res and score type combos
        """

        l = ['Weather Impact Scores - Daily Region', # should be: 17 (8z-24z) | 01 (actual) | 03 (0z,12z,18z tmrw)
             'Weather Impact Scores - Hourly Region', # should be: 17 (8z-24z) | 01 (actual) | 03 (0z,12z,18z tmrw)
             'Weather Impact Scores - Forecast Hourly Region', # should be 3, NextDay (0z,12z,18z), maybe should be mixed with above
             'Weather Impact Score - Daily', # airspace and airports? nominal should be 17 | 1 | 3
             'Weather Impact Score - Hourly', # airspace and airoprts? nominal should be 17 | 0 | 3
             # so the nominal total for above should be 21*4-1 = 83?

             # below seem to always be 1? for postop -- so this is the "actual" data
             'Weather Impact Scores - Hourly', 
             'Weather Impact Scores - Quarter Hour', 
             'Weather Impact Scores - Daily Area', 
             'Weather Impact Scores - Hourly Area', 
             'Weather Impact Scores - Daily Sector', 
             'Weather Impact Scores - Hourly Sector']
        # if len(df.index) != 89:
        #     print(date, len(df.index), 
        #           len(df.loc[df['scoreType'] == 'Forecast'].index), # 68 nominal
        #           len(df.loc[df['scoreType'] == 'PostOp'].index), # 9 nominal
        #           len(df.loc[df['scoreType'] == 'NextDay'].index), # 12 nominal
        #           len(df['name'].unique()),
        #     )
        #     for name in l:
        #         tmp = df.loc[df['name'] == name]
        #         print(" {:02d} ".format(len(tmp.index)) + name, end='\n')
        #         for s in ['Forecast', 'PostOp', 'NextDay']:
        #             print("   |   {:02d} ".format(len(tmp.loc[tmp['scoreType'] == s].index)) + s, end='')
        #         print('')
        #     print('')
        # if date == '2024-10-26': #len(df['name'].unique()) >= 11:
        #     print(date, [name for name in df['name'].unique()])
        #     for name in l:
        #         tmp = df.loc[df['name'] == name]
        #         print(" {:02d} ".format(len(tmp.index)) + name, end='\n')
        #         for s in ['Forecast', 'PostOp', 'NextDay']:
        #             print("   |   {:02d} ".format(len(tmp.loc[tmp['scoreType'] == s].index)) + s, end='')
        #         print('')
        #     print('')
        if len(df.loc[(df['name'] == 'Weather Impact Scores - Hourly Region') & (df['scoreType'] == 'NextDay')].index) > 0 \
            and len(df.loc[(df['name'] == 'Weather Impact Scores - Forecast Hourly Region') & (df['scoreType'] == 'NextDay')].index) > 0:
            print(date)


        # if len(df['scoreType'].unique()) < 3:
        #     print(date, df['scoreType'].unique())



if __name__ == '__main__':

    # ssd_base_dir = Path('/Volumes/SN850X/').resolve()
    # data_dir = Path(ssd_base_dir / 'avmet_simwx/')

    # data_dir = Path('../data/avmet_simwx/').resolve()
    # start_date = '2022-11-08' #'2017-01-01' # picking up where left off...
    # end_date = '2024-11-19'
    # download_scores(data_dir, start_date, end_date)

    # videos_dir = Path('data/avmet_simwx_videos/').resolve()
    # start_date = '2017-01-01'
    # end_date = '2024-11-19'
    # download_videos(videos_dir, start_date, end_date)

    data_dir = Path('data/avmet_simwx/').resolve()
    start_date = '2017-01-01' #'2017-01-01' # picking up where left off...
    end_date = '2024-11-19'
    read_scores(data_dir, start_date, end_date)