#!/usr/bin/env python3

import os
import argparse
import numpy as np
import rasterio
from tqdm import tqdm
import shutil
import time

"""
    Creates patches with original data with some overlap
    writes data in uint16 (original) and not float32 to save space
"""

def main():
    parser = argparse.ArgumentParser(description='Create image patches for U-Net training')
    parser.add_argument('--size', type=int, default=512, help='Patch size')
    parser.add_argument('--stride', type=int, default=256, help='Stride for sliding window')
    parser.add_argument('--overlap', type=float, default=0.0, help='Overlap percentage (0.0 to 0.9). If specified, overrides stride')
    
    args = parser.parse_args()
    
    # Input dirs
    data_dir = "/mnt/c/Users/Imesh/Desktop/summer_proj/MAPQ3389-EnSTAR"
    image_path = os.path.join(data_dir, "1-21-2022_Ortho_6Band.tif")
    labels_path = os.path.join(data_dir, "labels_converted.tif")
    output_dir = os.path.join(data_dir, "patches")
    
    # Calc stride 
    if args.overlap > 0:
        if args.overlap >= 1.0:
            raise ValueError("Overlap must be between 0.0 and 0.9")
        stride = int(args.size * (1 - args.overlap))
        print(f"Calculated stride from {args.overlap*100}% overlap: {stride}")
    else:
        stride = args.stride
    
    # Output dirs
    images_output_dir = os.path.join(output_dir, 'images')
    labels_output_dir = os.path.join(output_dir, 'labels')

    # Clear existing images #TODO  Need to handle train, val, test folders after being created
    clear_start = time.time()
    if os.path.exists(images_output_dir):
        shutil.rmtree(images_output_dir)
    if os.path.exists(labels_output_dir):
        shutil.rmtree(labels_output_dir)
    
    os.makedirs(images_output_dir, exist_ok=True)
    os.makedirs(labels_output_dir, exist_ok=True)
    clear_time = time.time() - clear_start
    print(f"  - Clear time: {clear_time:.2f} seconds ({clear_time/60:.2f} minutes)")

    print(">> Startinig patch creation...")
    print(f"  Image: {os.path.basename(image_path)}")
    print(f"  Labels: {os.path.basename(labels_path)}")
    print(f"  Output directory: {output_dir}")
    print(f"  Patch size: {args.size}x{args.size}")
    print(f"  Stride: {stride}")

    start_time = time.time()
    with rasterio.open(image_path) as src_img, rasterio.open(labels_path) as src_labels:
        height, width = src_img.shape
        kwargs_img = src_img.meta.copy()
        kwargs_labels = src_labels.meta.copy()
        
        patch_count = 0
        skipped_count = 0
        
        total_patches = ((height - args.size) // stride + 1) * ((width - args.size) // stride + 1)
        
        print(f"  Image dimensions: {width}x{height}")
        print(f"  Total potential patches: {total_patches}")
        
        # Loop over patches
        with tqdm(total=total_patches, desc="Creating patches") as pbar:
            for y in range(0, height - args.size + 1, stride):
                for x in range(0, width - args.size + 1, stride):
                    # Window for current patch
                    window = rasterio.windows.Window(x, y, args.size, args.size)
                    
		    # Calc patch transform and update metadat for new patch
                    patch_transform = rasterio.windows.transform(window, src_img.transform)
			 
                    kwargs_img.update({
                        'height': args.size,
                        'width': args.size,
                        'count': 3,
                        'transform': patch_transform
                    })

                    kwargs_labels.update({
                        'height': args.size,
                        'width': args.size,
                        'count': 1,
                        'transform': patch_transform
                    })
                    # Read only the current patch from disk (This is better than using patchify lib where it reads the whole file in)
                    image_patch = src_img.read([1,2,4], window=window)
                    labels_patch = src_labels.read(1, window=window)
                    
                    # Check if patch empty
                    if np.all(image_patch == 0):
                        skipped_count += 1
                        pbar.update(1)
                        continue
                    
                    # -------Save patches-------
                    patch_name = f"patch_{y:06d}_{x:06d}.tif"
                    mask_name = f"mask_{y:06d}_{x:06d}.tif"
                    
                    # Image patches
                    with rasterio.open(os.path.join(images_output_dir, patch_name), 'w', **kwargs_img) as dst:
                        for band_idx in range(3):
                            dst.write(image_patch[band_idx], band_idx + 1)
                    
                    # labels patches
                    with rasterio.open(os.path.join(labels_output_dir, mask_name), 'w', **kwargs_labels) as dst:
                        dst.write(labels_patch, 1)
                    
                    patch_count += 1
                    pbar.update(1)
    total = time.time() - start_time
    print(f">> Successfully created {patch_count} patches")
    print(f"  Skipped {skipped_count} empty patches")
    print(f"  Total non-empty patches: {patch_count}")
    print(f"  - Patch creation: {total:.2f} seconds ({total/60:.2f} minutes)")

if __name__ == "__main__":
    main()
