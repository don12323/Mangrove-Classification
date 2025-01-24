"""
Script for preparation of the pixel datset from NEO data and labeled polygon files?

Two main inputs are:
    1. a polygon file containing the geo-reference polugons and the labels of each parcel??? Use *.geojson??
    2. folder containing 6 band NEO data data in tif format.

    Not complete!! Working on this next.
"""

import numpy as np
import rasterio as rio
from rasterio.plot import show
from rasterio.mask import mask
import json
import os
import matplotlib.pyplot as plt
from rasterio.windows import Window

plt.style.use('seaborn-v0_8-bright')
plt.rcParams["font.family"] = "serif"

NEO_path = '/mnt/c/Users/Imesh/Desktop/summer_proj/MAPQ3389-EnSTAR'

RGB_path =  os.path.join(NEO_path,'1-21-2022_Ortho_ColorBalance.tif')
sixbands_path = os.path.join(NEO_path,'1-21-2022_Ortho_6Band.tif')


with rio.open(sixbands_path) as src:
    print('>> Opening all bands file')
    print(f' Size of data: {src.count}\n Dataset covers: {src.bounds}\n Dataset CRS: {src.crs}\n dim: {src.width}x{src.height}\n Indexes: {src.indexes}\n Datatype: {src.dtypes}')

    pixel_size_x = abs(src.transform[0])
    pixel_size_y = abs(src.transform[4])
    patch_size = 1000 # All in Meters
    window = Window(6000,8000,patch_size/pixel_size_x, patch_size/pixel_size_y)

    bands = [src.read(i,window=window) for i in range(1,5)]
    band_titles = ['Band1','Band2','Band3','Band4']


# Plotting first 4 bands in a small region
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
for i, ax in enumerate(axes.ravel()):
    ax.imshow(bands[i], cmap='gray',
            vmin=np.percentile(bands[i], 5),
            vmax=np.percentile(bands[i], 95))
    ax.set_title(band_titles[i])
    ax.axis('off')

plt.tight_layout()
plt.show()

with rio.open(RGB_path) as src:
    print('>> Opening RGB data')
    print(f' Size of data: {src.count}\n Dataset covers: {src.bounds}\n Dataset CRS: {src.crs}\n dim: {src.width}x{src.height}\n Indexes: {src.indexes}\n Datatype: {src.dtypes}')
    
    show(src.read(window=window),transform=src.transform)
print(patch_size)


