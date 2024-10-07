import pandas as pd
import plotly.express as px

# from shapely.geometry import Point
# import geopandas as gpd
# from geopandas import GeoDataFrame
# import geodatasets

# appears to read around once every ten seconds?? actually idk

from datetime import datetime as dt
import datetime
import time
# date_time = datetime.datetime(2020, 1, 1, 0, 0, 0)
# print("Given Date:",date_time)
# print("UNIX timestamp:",
# (time.mktime(date_time.timetuple())))
# midnight=int((time.mktime(date_time.timetuple())))
# print(midnight)


import data_utils

df = pd.read_csv('merge_clean.csv',index_col=[0,2])

# df = df.loc[df['UAID']=='JBU834']

# df['Time'] - midnight

# df = df[['Time','Longitude','Latitude']]

df = df.drop(['RecTypeCat', 'PointSource', 'CID'], axis=1)
df = df[df['Significance']<=5]

print(df)

print(df.index.get_level_values(0))
print(df.index.get_level_values(1))
# print(df.index.get_level_values(2))



# ts = traces.TimeSeries(df)

# print(ts)

# ts.sample(
#     sampling_period=datetime.timedelta(minutes=15),
#     # start=df.index[0].to_pydatetime(),
#     start=datetime.datetime(2020, 1, 1, 4, 26, 31),
#     # end=df.index[-1].to_pydatetime(),
#     end=datetime.datetime(2020,1,1,10,17,4),
#     interpolate='linear',
# )

# ts_long = traces.TimeSeries.from_csv(
#     'merge_clean.csv',
#     time_column=3,
#     time_transform=lambda x: dt.fromtimestamp(int(x)),
#     value_column=4,
#     value_transform=float,
#     default=0,
# )
# ts_long.compact()

# ts_lat = traces.TimeSeries.from_csv(
#     'merge_clean.csv',
#     time_column=3,
#     time_transform=lambda x: dt.fromtimestamp(int(x)),
#     value_column=5,
#     value_transform=float,
#     default=0,
# )
# ts_lat.compact()

# ts_long = ts_long.sample(
#     sampling_period=datetime.timedelta(minutes=1),
#     # start=datetime(1992, 8, 27, 8),
#     # end=datetime(1992, 8, 27, 9),
#     interpolate='linear',
# )


# geometry = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]
# gdf = GeoDataFrame(df, geometry=geometry)   

# world = gpd.read_file(geodatasets.data.naturalearth.land['url'])
# gdf.plot(ax=world.plot(figsize=(10, 6)), marker='o', color='red', markersize=15);

# fig = px.scatter_geo(df,
#                      lat='Latitude',
#                      lon='Longitude',
#                      hover_data='UAID',
#                      color='UAID',
#                     #  animation_frame=df.index,
#                      )
# fig.update_layout(title = 'World map', title_x=0.5)
# fig.show()