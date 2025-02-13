import numpy as np
import argparse
import os
import rasterio as rio
import geojson
import geopandas as gpd # Can't open geojson f with gpd-attribute error with fiona
from rasterio.windows import Window
from rasterio.features import geometry_mask
from shapely.geometry import shape, MultiPolygon, Polygon
import tkinter as tk
from pyproj import Transformer
from GUI import MangroveClassifierGUI


import matplotlib.pyplot as plt
"""
input: GeoJSON file with coordinates in EPSG:4326 (lonlat)
"""


# Paths

NEO_path = '/mnt/c/Users/Imesh/Desktop/summer_proj/MAPQ3389-EnSTAR'
output_path = '/mnt/c/Users/Imesh/Desktop/summer_proj/MAPQ3389-EnSTAR/patches'
geo_path = os.path.join(NEO_path,'testpols.geojson')
sixbands_path = os.path.join(NEO_path, '1-21-2022_Ortho_6Band.tif')

def read_geojson(file_path):
    """Read GeoJSON file and return the polygons"""
    with open(file_path, 'r') as f:
        gj = geojson.load(f)

    polygons = []
    for feature in gj['features']:
        geom = shape(feature['geometry'])
        if isinstance(geom, (Polygon, MultiPolygon)):
            polygons.append(geom)
            gdf = gpd.GeoDataFrame(geometry=polygons, crs="EPSG:4326")
            gdf_utm = gdf.to_crs("EPSG:32750") #TODO change to src.crs
        else:
            print(f"Skipping non-polygon geomL {geom.type}")
    return list(gdf_utm.geometry)

def save_patch_mask(mask, src, window, output_name, output_path):
    """Save the classified patch as a GeoTIFF"""
    path = os.path.join(output_path, output_name)
    transform = rio.windows.transform(window, src.transform)
    print(mask.shape, np.max(mask))
    print(window)
    with rio.open(
        path,
        'w',
        driver='GTiff',
        height=mask.shape[0],
        width=mask.shape[1],
        count=1,
        dtype=mask.dtype,
        crs=src.crs,
        transform=transform,
        nodata=0
    ) as dst:
        dst.write(mask, 1)

def get_polygon_window(src, polygon):
    bounds = list(polygon.bounds)
    try:
        # Convert to pix coord
        bounds = [(bounds[0], bounds[1]), (bounds[2], bounds[3])] 

        print(f"Bounds in geo coord:{bounds}")
        # Convert bounds to pixels and get window dimensions
        (row_start, col_start), (row_stop, col_stop) = [src.index(x, y) for x, y in bounds]
        width, height = col_stop - col_start, row_start - row_stop   # swap col and row:wq
        
        print(f"Bounds in pix coord:({col_start}, {row_start}) ({col_stop}, {row_stop})")
        print(f"Window dimensions: {width}x{height}")
        
        if width <= 0 or height <= 0:
            raise ValueError("Invalid winodow dimensions")
        
        window = Window(col_start,row_stop, width, height)
        print(f"window:{window}")
        # Make mask for the polygon region
        shapes = [polygon] # Needs to be in a list for the geometry_mask func in rasterio
        transform = rio.windows.transform(window, src.transform)
        mask = geometry_mask(shapes, out_shape=(height,width), 
                transform=transform, invert=True)
        
        return window, mask
    except Exception as e:
        print(f"Error creating window and mask: {e}")
        return None, None

def process_polygons():
    """Process all polygons in geojson file and run k-means"""
    try:
        print(f"Reading polygons from {os.path.basename(geo_path)}")
        polygons = read_geojson(geo_path)
        
        if not polygons:
            Print("No valid polygons were found")
            return
        npoly = len(polygons)
        print(f"Found {len(polygons)} polygons")
        
        # Iterate through each polygon
        with rio.open(sixbands_path) as src:
            print(src.crs)
            for i, polygon in enumerate(polygons):
                print(f"\nProcessing polygon {i+1}/{npoly}")
                try:
                    window, poly_mask = get_polygon_window(src, polygon)
                    
                    if window is None or poly_mask is None:
                        print(f"Skipping polygon {i + 1}: Invalid window or mask")
                        continue
                    patch = np.stack([src.read(k, window=window).astype(np.float32) 
                        for k in range(1, 5)], axis=-1)
                    # Apply mask
                    print(f"patch shape:{patch.shape}, mask shape: {poly_mask.shape}")
                    #poly_mask = poly_mask.T
                    patch = patch * poly_mask[:,:,np.newaxis]
                    if np.all(patch == 0):
                        print(f"Skipping polygon {i + 1}: No valid data")
                        continue
                    # Add NDVI
                    redb, nirb = patch[:,:,0], patch[:,:,3]
                    valid_pixels = (redb != 0) & (nirb != 0)
                    ndvi = np.zeros_like(redb)
                    ndvi[valid_pixels] = (nirb[valid_pixels] - redb[valid_pixels]) / (nirb[valid_pixels] + redb[valid_pixels] + 1e-8)
                    ndvi = ndvi[:,:,np.newaxis]
                    patch = np.append(patch, ndvi, axis=2)

                    # GUI stuff
                    root = tk.Tk()
                    centroid = polygon.centroid
                    app = MangroveClassifierGUI(root, patch, coords=(centroid.x, centroid.y))
                    root.mainloop()

                    try:
                        if not app.skip_patch and app.selected_label is not None:
                            # Apply classification only within polygon
                            mask = (app.labels == app.selected_label).astype(np.uint8)
                            mask = mask * poly_mask
                            output_name = f"polygon_{i + 1}_classified.tif"
                            save_patch_mask(mask, src, window, output_name, output_path)
                            print(f"Successfully saved classified polygon to {output_name}")

                    finally:
                        if root.winfo_exists():
                            root.destroy()

                except Exception as e:
                    print(f"Error processing polygon {i+1}/{npoly}: {e}")
                    continue
        
    except Exception as e:
        print(f"Error processing polygons: {e}")
        raise


if __name__=="__main__":
    process_polygons()

