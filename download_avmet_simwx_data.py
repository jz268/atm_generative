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

    for date in dates:
        file_path = date_path(data_dir, date, 'json')
        with open(file_path) as f:
            data = json.load(f)
            print(data)



if __name__ == '__main__':

    # ssd_base_dir = Path('/Volumes/SN850X/').resolve()
    # data_dir = Path(ssd_base_dir / 'avmet_simwx/')

    data_dir = Path('data/avmet_simwx/').resolve()
    start_date = '2022-11-08' #'2017-01-01' # picking up where left off...
    end_date = '2024-11-19'
    download_scores(data_dir, start_date, end_date)
    # read_scores(data_dir, start_date, end_date)

    videos_dir = Path('data/avmet_simwx_videos/').resolve()
    start_date = '2017-01-01'
    end_date = '2024-11-19'
    download_videos(videos_dir, start_date, end_date)