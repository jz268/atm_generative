import subprocess

import matplotlib.pyplot as plt

import geopandas as gpd
import geodatasets
import contextily as cx

import rioxarray as rxr
from rasterio.plot import plotting_extent
import earthpy.plot as ep



source_image = 'data/weather/n0q_202001010000.png'
output_image = 'data/weather/n0q_202001010000.tif'

# subprocess.call(['gdal_translate', '-co', 'compress=lzw', '-of', 'Gtiff', '-a_srs', 'EPSG:4326', source_image, output_image])

# alternative: https://mesonet.agron.iastate.edu/request/gis/n0r2gtiff.php?dstr=202001010000
# # https://stackoverflow.com/questions/64589390/python-georasters-geotiff-image-into-geopandas-dataframe-or-pandas-dataframe


# # https://gis.stackexchange.com/questions/384581/raster-to-geopandas
# rds = rxr.open_rasterio(output_image)
# rds.name = "data"
# df = rds.squeeze().to_dataframe().reset_index()
# geometry = gpd.points_from_xy(df.x, df.y)
# gdf = gpd.GeoDataFrame(df, crs=rds.rio.crs, geometry=geometry)

# print(gdf)

# df = gdf

# ax = df.plot(figsize=(10, 10), alpha=0.5, edgecolor="k")

# print(df.crs)

