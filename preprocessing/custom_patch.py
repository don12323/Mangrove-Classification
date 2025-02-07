import argparse
import os
import numpy as np
import rasterio as rio
from rasterio.windows import Window
import tkinter as tk
from GUI import MangroveClassifierGUI

NEO_path = '/mnt/c/Users/Imesh/Desktop/summer_proj/MAPQ3389-EnSTAR'
sixbands_path = os.path.join(NEO_path, '1-21-2022_Ortho_6Band.tif')

def get_pixel_coords(src, x, y):
    """Convert geographic coordinates to pixel coordinates"""
    return ~src.transform * (x, y)

def save_patch_mask(mask, src, window, output_name):
    """Save the classified patch as a GeoTIFF"""
    output_path = os.path.join(NEO_path, output_name)
    transform = rio.windows.transform(window, src.transform)
    
    with rio.open(
        output_path,
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

def process_single_patch(coords, width, height, output_name):
    """Process a single rectangular patch at the specified coordinates"""
    print(f"Processing patch at coordinates: {coords}")
    print(f"Patch dimensions: {width}x{height}")
    
    with rio.open(sixbands_path) as src:
        # Convert geographic coordinates to pixel coordinates
        col, row = get_pixel_coords(src, coords[0], coords[1])
        col, row = int(col), int(row)
        
        # Calculate window bounds
        half_width = width // 2
        half_height = height // 2
        window = Window(
            col - half_width,
            row - half_height,
            width,
            height
        )
        
        # Read patch data
        patch = np.stack([src.read(k, window=window).astype(np.float32) 
                         for k in range(1, 7)], axis=-1)
        
        # Calculate NDVI
        redb, nirb = patch[:,:,0], patch[:,:,3]
        ndvi = (nirb - redb) / (nirb + redb + 1e-8)
        ndvi = np.nan_to_num(ndvi)
        ndvi = ndvi[:,:,np.newaxis]
        patch = np.append(patch, ndvi, axis=2)
        
        # Create GUI
        root = tk.Tk()
        app = MangroveClassifierGUI(root, patch, coords)
        root.mainloop()
        
        try:
            if not app.skip_patch and app.selected_label is not None:
                mask = (app.labels == app.selected_label).astype(np.uint8)
                save_patch_mask(mask, src, window, output_name)
                print(f"Successfully saved classified patch to {output_name}")
            else:
                print("Patch processing was skipped")
        finally:
            if root.winfo_exists():
                root.destroy()

def main():
    parser = argparse.ArgumentParser(description='Process a single patch for mangrove classification')
    parser.add_argument('--x', type=float, required=True, help='X coordinate (longitude)')
    parser.add_argument('--y', type=float, required=True, help='Y coordinate (latitude)')
    parser.add_argument('--width', type=int, default=500, 
                      help='Width of the patch in pixels (default: 500)')
    parser.add_argument('--height', type=int, default=500, 
                      help='Height of the patch in pixels (default: 500)')
    parser.add_argument('--output', required=True, 
                      help='Output filename (e.g., "patch1_classified.tif")')
    
    args = parser.parse_args()
    
    try:
        process_single_patch(
            (args.x, args.y),
            args.width,
            args.height,
            args.output
        )
    except Exception as e:
        print(f"Error processing patch: {e}")
        raise

if __name__ == "__main__":
    main()
