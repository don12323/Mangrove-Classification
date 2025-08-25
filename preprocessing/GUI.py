# mangrove_gui.py
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import torch
from fast_pytorch_kmeans import KMeans

class MangroveClassifierGUI:
    def __init__(self, master, patch_data, coords=None):
        self.master = master
        self.master.title("Mangrove Classifier")
        self.patch = patch_data
        self.coords = coords
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
        if messagebox.askokcancel("Quit", "Do you want to skip this patch?"):
            self.skip_patch = True
            self.master.quit()
    
    def select_mangroves(self):
        self.selected_label = self.cluster_var.get()
        self.master.quit()
    
    def skip_current(self):
        self.skip_patch = True
        self.master.quit()
        
    def check_empty_patch(self):
        valid_data_percentage = np.mean(self.patch[:,:,0] != 0)
        return valid_data_percentage < 0.1
        
    def run_kmeans(self):
        try:
            valid_pixels = ~np.all(self.patch == 0, axis=2)
            patch_reshaped = self.patch[valid_pixels].reshape(-1, self.patch.shape[-1])
            
            if patch_reshaped.shape[0] == 0:
                self.skip_patch = True
                return
                
            X = torch.from_numpy(patch_reshaped).to(torch.float).to('cuda')
            
            kmeans = KMeans(n_clusters=self.current_k, mode='euclidean', 
                          init_method="random", verbose=0)
            labels_valid = kmeans.fit_predict(X)
            
            self.labels = np.full(self.patch.shape[:2], -1, dtype=int)
            self.labels[valid_pixels] = labels_valid.cpu().numpy()
            
        except Exception as e:
            print(f"Error in k-means clustering: {e}")
            self.skip_patch = True
            
    def setup_gui(self):
        main_frame = ttk.Frame(self.master, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.fig = Figure(figsize=(13, 7))
        self.ax1 = self.fig.add_subplot(121)
        self.ax2 = self.fig.add_subplot(122)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, columnspan=2)
        
        slider_frame = ttk.Frame(main_frame)
        slider_frame.grid(row=1, column=0, columnspan=2, pady=10)
        
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
        
        info_frame = ttk.Frame(main_frame)
        info_frame.grid(row=2, column=0, columnspan=2, pady=5)
        
        if self.coords:
            ttk.Label(info_frame, 
                     text=f"Coordinates: {self.coords}").pack()
        ttk.Label(info_frame, 
                 text=f"Current K value: {self.current_k}").pack()
        
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
        try:
            self.ax1.clear()
            self.ax2.clear()
            
            rgb_img = np.stack([self.patch[:,:,i] for i in range(3)], axis=-1)
            rgb_img = np.clip((rgb_img - rgb_img.min()) / 
                    (rgb_img.max() - rgb_img.min() + 1e-8), 0, 1)
            
            self.ax1.imshow(rgb_img)
            self.ax1.set_title("RGB Image")
            
            self.ax2.imshow(rgb_img)
            mask = self.labels == self.cluster_var.get()
            self.ax2.imshow(mask, cmap='Reds', alpha=0.5)
            self.ax2.set_title(f"Cluster {self.cluster_var.get()} Overlay")
            
            self.fig.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            print(f"Error updating display: {e}")
            
    def adjust_k(self):
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
                    self.cluster_slider.configure(to=new_k-1)
                    self.cluster_var.set(0)
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
        dialog.geometry(f"+{self.master.winfo_rootx()+50}+"
                       f"{self.master.winfo_rooty()+50}")
