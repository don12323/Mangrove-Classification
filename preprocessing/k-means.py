"""
Script for labelling the NEO data using k-means clustering using first 4 bands.
Trying to see if we can pick up darker mangroves seperately.

Inputs are just the NEO data.
"""
#from sklearn.cluster import KMeans
#from sklearn import cluster
#from sklearn import metrics

from fast_pytorch_kmeans import KMeans #https://github.com/DeMoriarty/fast_pytorch_kmeans
import torch

import rasterio as rio
import earthpy as et
import earthpy.spatial as es
from rasterio.windows import Window

import numpy as np
import json
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mc

from metrics import GPUSilhouetteScore

plt.style.use('seaborn-v0_8-bright')
plt.rcParams["font.family"] = "serif"

NEO_path = '/mnt/c/Users/Imesh/Desktop/summer_proj/MAPQ3389-EnSTAR'

RGB_path =  os.path.join(NEO_path,'1-21-2022_Ortho_ColorBalance.tif')
sixbands_path = os.path.join(NEO_path,'1-21-2022_Ortho_6Band.tif')

with rio.open(sixbands_path) as src:
    print('>> Opening all bands file')
    print(src.meta)
    pixel_size_x = abs(src.transform[0])
    pixel_size_y = abs(src.transform[4])
    patch_size = 200 # All in Meters
    window = Window(1000,8000,patch_size/pixel_size_x, patch_size/pixel_size_y)

    data_arr3D = np.stack([src.read(i,window=window).astype(np.float32) for i in range(1,5)], axis=-1)
    print('size of bands array', data_arr3D.shape)

"""
Convert to 1D array of observations and train a classifier

how do you test for convexness??
"""
k=4
redb = data_arr3D[:,:,0]
nirb = data_arr3D[:,:,3]

#NDVI = np.where(nirb+redb==0,0,(nirb-redb)/(nirb+redb))
NDVI = es.normalized_diff(nirb,redb)
print(f'NDVI range: [{np.min(NDVI)},{np.max(NDVI)}]')


print(f"NDVI shape: {NDVI.shape}, data_arr3D shape: {data_arr3D.shape}")
NDVI = NDVI[:,:,np.newaxis]
data_arr3D = np.append(data_arr3D,NDVI,axis=2)
height, width,__ = data_arr3D.shape
data_arr2D = data_arr3D.reshape(height*width,5)

random_seed=42 # For reproducibility of results
# sklearn
#cl = cluster.KMeans(n_clusters=k, random_state=random_seed, n_jobs=-1, verbose=1)
#param = cl.fit(data_arr2D)


#img_cl = param.labels_

#sh_score=metrics.silhouette_score(data_arr2D, img_cl, metric='euclidean', n_jobs=-1) # doesnt show how many iterations left
#print(f">> Silhouette Coefficient for k={k}:{sh_score} ")

#sil_score_manual = calc_silhouette_score(data_arr2D, img_cl, n_jobs=1)
#print(f"Manual Silhouette Score: {sil_score_manual}")

#img_cl = img_cl.reshape(data_arr3D[:,:,0].shape)

"""
torch_kmeans
"""
kmeans = KMeans(n_clusters=4, mode='euclidean', init_method="random",verbose=1)
X = torch.from_numpy(data_arr2D).to(torch.float).to('cuda')

img_cl=kmeans.fit_predict(X)
sh_gpu = GPUSilhouetteScore()
gpu_score, individual_scores = sh_gpu.calculate(
        data=data_arr2D,
        labels=img_cl,
        batch_size=512
)

img_cl = img_cl.cpu().numpy()
img_cl = img_cl.reshape(data_arr3D[:,:,0].shape)

print(f"Sh score using gpu: {gpu_score}")

"""
Plotting
"""

band_titles = ['Band1','Band2','Band3','Band4']
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
for i, ax in enumerate(axes.ravel()):
    ax.imshow(data_arr3D[:,:,i], cmap='gray',
            vmin=np.percentile(data_arr3D[i], 5),
            vmax=np.percentile(data_arr3D[i], 95))
    ax.set_title(band_titles[i])
    ax.axis('off')

plt.tight_layout()
plt.show()

with rio.open(RGB_path) as src:
    print('>> Opening RGB data')
    print(src.meta)
    RGBim = np.stack([src.read(i,window=window) for i in range(1,4)],axis=-1)


cmap = mc.LinearSegmentedColormap.from_list("", ["black","red","green","yellow"])#,"blue","pink","cyan"])

mask = img_cl == 3
lab1_mask = np.where(mask, img_cl, np.nan)

plt.figure(figsize=(12, 12))

plt.subplot(2,2,1)
plt.imshow(RGBim)
plt.title("NEO imagery")
plt.axis("off")

plt.subplot(2,2,2)
plt.imshow(img_cl, cmap=cmap)
plt.title("Classified labels using clustering")
plt.axis("off")


plt.subplot(2,2,3)
plt.imshow(RGBim)
plt.imshow(lab1_mask, cmap='Reds', alpha=0.6)
plt.title("NEO imagery + Label overlay")
plt.axis("off")


plt.subplot(2,2,4)
plt.imshow(NDVI, cmap='Greens', vmin=-1, vmax=1)
plt.title("NDVI")
plt.axis("off")

plt.tight_layout()
plt.show()




