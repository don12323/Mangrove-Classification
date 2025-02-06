import numpy as np
import os
import rasterio as rio
from rasterio.windows import Window
import torch
from fast_pytorch_kmeans import KMeans # Adapted from: https://github.com/DeMoriarty/fast_pytorch_kmeans
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tqdm import tqdm
"""
Script for interactive classification of mangroves using k-means clustering for large datasets.
Iterates through patches of desired size and .......

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

class MangroveClassifierGUI:
    
    def __init__(self, master, patch_data, window_coords):
        self.master = master
        self.master.title("Mangrove Classifier")
        self.patch = patch_data
        self.row, self.col = window_coords
        self.current_k = 4
        self.selected_label = None
        self.skip_patch = False
        
        # Add protocol handler for window closing
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Check for empty patch before proceeding
        if self.check_empty_patch():
            self.skip_patch = True
            self.master.after(100, self.master.quit)
            return
            
        self.run_kmeans()
        self.setup_gui()
    
    def on_closing(self):
        """Handle window closing event."""
        if messagebox.askokcancel("Quit", "Do you want to skip this patch?"):
            self.skip_patch = True
            self.master.quit()
    
    def select_mangroves(self):
        """Select current cluster as mangroves and close window."""
        self.selected_label = self.cluster_var.get()
        self.master.quit()
    
    def skip_current(self):
        """Skip current patch and close window."""
        self.skip_patch = True
        self.master.quit()
        
    def check_empty_patch(self):
        """
        Check if patch contains valid data.
        Returns True if patch is mostly empty (>90% zeros or invalid data).
        """
        # Calculate percentage of valid data in the first band
        valid_data_percentage = np.mean(self.patch[:,:,0] != 0)
        return valid_data_percentage < 0.1
        
    def run_kmeans(self):
        """Perform k-means clustering on valid patch data."""
        try:
            # Create mask for valid pixels
            valid_pixels = ~np.all(self.patch == 0, axis=2)
            patch_reshaped = self.patch[valid_pixels].reshape(-1, self.patch.shape[-1])
            
            if patch_reshaped.shape[0] == 0:
                self.skip_patch = True
                return
                
            X = torch.from_numpy(patch_reshaped).to(torch.float).to('cuda')
            
            kmeans = KMeans(n_clusters=self.current_k, mode='euclidean', 
                          init_method="random", verbose=0)
            labels_valid = kmeans.fit_predict(X)
            
            # Initialize labels array with -1 (indicating no cluster)
            self.labels = np.full(self.patch.shape[:2], -1, dtype=int)
            self.labels[valid_pixels] = labels_valid.cpu().numpy()
            
        except Exception as e:
            print(f"Error in k-means clustering: {e}")
            self.skip_patch = True
            
    def setup_gui(self):
        """Set up the GUI components with improved layout."""
        main_frame = ttk.Frame(self.master, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create fig
        self.fig = Figure(figsize=(18, 12))
        self.ax1 = self.fig.add_subplot(121)
        self.ax2 = self.fig.add_subplot(122)
        
        # Canvas to show fig
        self.canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, columnspan=2)
        
        # Create slider frame
        slider_frame = ttk.Frame(main_frame)
        slider_frame.grid(row=1, column=0, columnspan=2, pady=10)
        
        # Cluster selection slider
        ttk.Label(slider_frame, text="Cluster:").pack(side=tk.LEFT)
        self.cluster_var = tk.IntVar(value=0)
        self.cluster_slider = ttk.Scale(
            slider_frame, 
            from_=0, 
            to=self.current_k-1,
            variable=self.cluster_var,
            orient=tk.HORIZONTAL,
            command=self.update_display
        )
        self.cluster_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Coordinates and info frame
        info_frame = ttk.Frame(main_frame)
        info_frame.grid(row=2, column=0, columnspan=2, pady=5)
        
        ttk.Label(info_frame, 
                 text=f"Patch coordinates: Row {self.row}, Col {self.col}").pack()
        ttk.Label(info_frame, 
                 text=f"Current K value: {self.current_k}").pack()
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=10)
        
        ttk.Button(button_frame, text="Select as Mangroves", 
                  command=self.select_mangroves).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Skip Patch", 
                  command=self.skip_current).grid(row=0, column=1, padx=5)
        ttk.Button(button_frame, text="Adjust K", 
                  command=self.adjust_k).grid(row=0, column=2, padx=5)
        
        self.update_display()
        
    def update_display(self, *args):
        """Update the display with current cluster selection."""
        try:
            self.ax1.clear()
            self.ax2.clear()
            
            # Normalize (for some reason it makes some patches dim)
            rgb_img = np.stack([self.patch[:,:,i] for i in range(3)], axis=-1)
            rgb_img = np.clip((rgb_img - rgb_img.min()) / 
                    (rgb_img.max() - rgb_img.min() + 1e-8), 0, 1)
            
            self.ax1.imshow(rgb_img)
            self.ax1.set_title("RGB Image")
            self.ax1.axis('off')
            
            # Display cluster overlay #TODO add option to turn on and off overlay
            self.ax2.imshow(rgb_img)
            mask = self.labels == self.cluster_var.get()
            self.ax2.imshow(mask, alpha=0.5, cmap="Reds")
            self.ax2.set_title(f"Cluster {self.cluster_var.get()} Overlay")
            self.ax2.axis('off')
            
            self.fig.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            print(f"Error updating display: {e}")
            
    def adjust_k(self):
        """Adjust number of clusters with improved UI feedback."""
        dialog = tk.Toplevel(self.master)
        dialog.title("Adjust K")
        dialog.transient(self.master)
        
        ttk.Label(dialog, text="Enter new K value:").pack(pady=5)
        k_entry = ttk.Entry(dialog)
        k_entry.pack(pady=5)
        k_entry.insert(0, str(self.current_k))
        
        def apply_new_k():
            try:
                new_k = int(k_entry.get())
                if 2 <= new_k <= 10:
                    self.current_k = new_k
                    # Update slider configuration
                    self.cluster_slider.configure(to=new_k-1)
                    self.cluster_var.set(0)
                    # Rerun clustering with new k entered
                    self.run_kmeans()
                    self.update_display()
                    dialog.destroy()
                else:
                    messagebox.showwarning("Invalid Input", 
                                         "Please enter a value between 2 and 10")
            except ValueError:
                messagebox.showwarning("Invalid Input", 
                                     "Please enter a valid number")
        
        ttk.Button(dialog, text="Apply", command=apply_new_k).pack(pady=5)
        
        # Center dialog
        dialog.geometry(f"+{self.master.winfo_rootx()+50}+"
                       f"{self.master.winfo_rooty()+50}")

def process_image(src_path, patch_size=2500, initial_clusters=4):
    """ Main part of code fo processing the image with interactive classification."""
    print("\nStarting image processing...")
    
    with rio.open(src_path) as src:
        meta = src.meta
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
        
        print(f"\nProcessing complete:")
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
