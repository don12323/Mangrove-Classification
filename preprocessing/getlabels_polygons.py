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
            gdf = gpd.GeoDataFrame(geometry=polygons, crs="EPSG:4326")
            gdf_utm = gdf.to_crs("EPSG:32750")
        else:
            print(f"Skipping non-polygon geomL {geom.type}")
    return list(gdf_utm.geometry)

def get_pixel_coords(src, x, y):
    """Convert WGS84 coordinates to pixel coordinates via UTM projection"""
    source_crs = "EPSG:4326"
    target_crs = src.crs
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    x, y = transformer.transform(x, y)
    return src.index(x, y)

def get_polygon_window(src, polygon):
    bounds = list(polygon.bounds)
    # Convert to pix coord
    bounds = [(bounds[0], bounds[1]), (bounds[2], bounds[3])] 

    print(f"Bounds in geo coord:{bounds}")
    print(polygon)
    # Convert bounds to pixels and get window dimensions
    (col_start, row_start), (col_stop, row_stop) = [src.index(x, y) for x, y in bounds]
    width, height = col_start - col_stop, row_stop - row_start
    print(f"Bounds in pix coord:({col_start}, {row_start}) ({col_stop}, {row_stop})")
    if width <= 0 or height <= 0:
        raise ValueError("Invalid winodow dimensions")
    
    window = Window(row_start,col_stop, width, height)
    # Make mask for the polygon region
    shapes = [polygon] # Needs to be in a list for the geometry_mask func in rasterio
    transform = rio.windows.transform(window, src.transform)
    print(f"Window dimensions: {width}x{height}")
    print(f"Trnasofrm: {transform}")
    mask = geometry_mask(shapes, out_shape=(width, height),  #TODO the issue here is we havent converted polygon vertices to the target crs
            transform=transform, invert=True)
    print(f"mask min: {np.min(mask)}, mask max: {np.max(mask)}")
    plt.figure(figsize=(10, 10))
    plt.imshow(mask, cmap='gray')
    plt.show()
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
                
        #plt.show()
        
    except Exception as e:
        print(f"Error processing polygons: {e}")
        raise


if __name__=="__main__":
    process_polygons()

