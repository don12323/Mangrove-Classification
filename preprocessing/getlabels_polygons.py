import numpy as np
import argparse
import os
import rasterio as rio
import geojson
from rasterio.windows import Window
from rasterio.features import geometry_mask
from shapely.geometry import shape, MultiPolygon, Polygon
import tkinter as tk
from pyproj import Transformer
from GUI import MangroveClassifierGUI

import matplotlib.pyplot as plt
# Paths

NEO_path = '/mnt/c/Users/Imesh/Desktop/summer_proj/MAPQ3389-EnSTAR'
geo_path = os.path.join(NEO_path,'custom_pol.geojson')
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
        else:
            print(f"Skipping non-polygon geomL {geom.type}")
    return polygons

def get_pixel_coords(src, x, y):
    """Convert WGS84 coordinates to pixel coordinates via UTM projection"""
    source_crs = "EPSG:4326"
    target_crs = src.crs
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    x, y = transformer.transform(x, y)
    return src.index(x, y)

def get_polygon_window(src, polygon):
    bounds = list(polygon.bounds)
    # Bounds[max_col, min_row, min_col, max_row]
    for i in range(0, len(bounds), 2):
        col, row = get_pixel_coords(src, bounds[i], bounds[i+1])
        bounds[i] = int(col)
        bounds[i+1] = int(row)
    
    print(f"New Bounds:{bounds}")
    width = bounds[0] - bounds[2]
    height = bounds[3] - bounds[1]
    
    if width <= 0 or height <= 0:
        raise ValueError("Invalid winodow dimensions")
    window = Window(bounds[1],bounds[2], width, height)
    # Make mask for the polygon region
    shapes = [polygon] # Needs to be in a list for the geometry_mask func in rasterio
    transform = rio.windows.transform(window, src.transform)
    mask = geometry_mask(shapes, out_shape=(width, height),
            transform=transform, invert=False)
    plt.figure(figsize=(10, 10))
    plt.imshow(mask, cmap='gray')
    
    return window, mask

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
                window, poly_mask = get_polygon_window(src, polygon)
                
        plt.show()
        
    except Exception as e:
        print(f"Error processing polygons: {e}")
        raise


if __name__=="__main__":
    process_polygons()

