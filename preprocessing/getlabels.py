import numpy as np
import os
import rasterio as rio
from rasterio.windows import Window

import tkinter as tk
from tqdm import tqdm
from GUI import MangroveClassifierGUI
"""
Script for interactive classification of mangroves using k-means clustering for large datasets.
Iterates through patches of desired size, runs k-means and selects which class is mangrove manually.
Output is a binary map of mangroves in gtiff format that is the original size.

"""



def save_mask(mask, src, output_path):

    """
    Write binary mask to GTiff file.
	
    Args:
        mask: Binary numpy array of where mangroves are.
        src: 
        output_path: Path where the GTiff file will be saved
    """
    print("\nConverting mask to GTiff file...")
    try:
        with rio.open(
            output_path,
            'w',
            driver='GTiff',
            height=mask.shape[0],
            width=mask.shape[1],
            count=1,
            dtype=mask.dtype,
            crs=src.crs,
            transform=src.transform,
            nodata=0
        ) as dst:
            dst.write(mask, 1)

        print(f"Successfully saved mask to {output_path}")
    except Exception as e:
        print(f"Error saving Tiff file: {e}")
        raise


def process_image(src_path, patch_size=3333, initial_clusters=4):
    """ Main part of code fo processing the image with interactive classification."""
    print("\nStarting image processing...")
    
    with rio.open(src_path) as src:
        #meta = src.meta
        width, height = src.width, src.height
        print(f"Opening 6band  file: {os.path.basename(src_path)}") 
        print(f"\nImage dimensions: {width}x{height}")
        print(f"Processing patches of size: {patch_size}x{patch_size}")
        
        # Calculate total number of patches
        total_patches = ((height + patch_size - 1) // patch_size) * \
                       ((width + patch_size - 1) // patch_size)
        
        labeled_data = np.zeros((height, width), dtype=np.uint8)  #TODO change this to float
        processed_patches = 0
        empty_patches = 0
        
        # Create progress bar
        pbar = tqdm(total=total_patches, desc="Processing patches")
        
        for i in range(0, height, patch_size):
            for j in range(0, width, patch_size):
                window = Window(j, i, 
                              min(patch_size, width - j),
                              min(patch_size, height - i))
                
                # Read patch
                patch = np.stack([src.read(k, window=window).astype(np.float32) 
                                for k in range(1, 7)], axis=-1)
                
                # Calculate NDVI and append to patch 
                redb, nirb = patch[:,:,0], patch[:,:,3]
                ndvi = (nirb - redb) / (nirb + redb + 1e-8)
                ndvi = np.nan_to_num(ndvi)
                ndvi = ndvi[:,:,np.newaxis]
                patch = np.append(patch, ndvi, axis=2)
                
                # Skip if patch is empty
                if np.mean(patch[:,:,0] != 0) < 0.1:
                    empty_patches += 1
                    pbar.update(1)
                    continue
                
                # Create GUI for patch
                root = tk.Tk()
                app = MangroveClassifierGUI(root, patch, (i, j))
                root.mainloop()
                
                try:
                    # Process results
                    if not app.skip_patch and app.selected_label is not None:
                        mask = app.labels == app.selected_label
                        labeled_data[i:i + window.height, 
                                   j:j + window.width][mask] = 1
                        processed_patches += 1
                finally:
                    if root.winfo_exists():
                        root.destroy()
                
                pbar.update(1)
        
        pbar.close()
        
        print("\nProcessing complete:")
        print(f"Total patches: {total_patches}")
        print(f"Empty patches skipped: {empty_patches}")
        print(f"Patches with mangroves: {processed_patches}")
        
        return labeled_data, src

def main():
    # Define paths
    NEO_path = '/mnt/c/Users/Imesh/Desktop/summer_proj/MAPQ3389-EnSTAR'
    results_path = '/mnt/c/Users/Imesh/Desktop/summer_proj/results'
    sixbands_path = os.path.join(NEO_path, '1-21-2022_Ortho_6Band.tif')
    output_gtiff_path = os.path.join(results_path, 'mangrove_mask.tif')
    
    print("Starting Mangrove Classification Process for file")
    
    try:
        labeled_data, src = process_image(sixbands_path)
        save_mask(labeled_data, src, output_gtiff_path)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        raise

if __name__ == "__main__":
    main()
