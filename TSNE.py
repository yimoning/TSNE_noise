import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # Use 1 thread for OpenBLAS
os.environ['MKL_NUM_THREADS'] = '1'  # If using MKL
os.environ['OMP_NUM_THREADS'] = '1'  # For other libraries using OpenMP


import torch
from matplotlib.colors import BoundaryNorm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from noise import *
from imutils import *
# from diffusion_learned_only.starter_kit.noise import add_natural_noise
from degradations import simple_deg_simulation

class ImageTSNEVisualizer:
    def __init__(self, path, max_files=20, noise=True):
        self.path = path
        self.images = []
        self.features = []
        self.img_labels = []  # This list will store the labels for each degraded version of each image
        self.feature_labels = []
        self.noise = noise
        self.load_and_normalize_images(max_files)

    def load_and_normalize_images(self, max_files):
        kernel = np.load("kernels.npy", allow_pickle=True)
        """Load image data from .npz files, normalize them, and store labels if available."""
        for i in range(1, max_files + 1):
            file_path = os.path.join(self.path, f'{i}.npz')
            if os.path.exists(file_path):
                data = np.load(file_path)
                raw_img = data['raw']
                max_val = data['max_val']
                normalized_img = (raw_img / max_val).astype(np.float32)
                # new_width, new_height = 64, 64
                # normalized_img = cv2.resize(normalized_img, (new_width, new_height))
                # for _ in range(50):
                self.images.append(downsample_raw(convert_to_tensor(normalized_img)).flatten())
                self.img_labels.append(i - 1)  # Same label for all degraded versions
                # Apply degradation simulation multiple times and keep the same label for all
                for _ in range(50):  # Create 10 degraded versions
                    degraded_img = simple_deg_simulation(normalized_img, kernel)
                    self.features.append(degraded_img.flatten())
                    self.feature_labels.append(i - 1)  # Same label for all degraded versions
            else:
                print(f"File {file_path} not found")
        self.images = np.array(self.images)
        self.features = np.array(self.features)
        if len(self.images) == 0:
            raise ValueError("No images were loaded.")

    def plot_tsne(self):
        """Perform PCA reduction followed by t-SNE and plot the results with discrete labels for images and features."""
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        import numpy as np
        from matplotlib.colors import ListedColormap, BoundaryNorm

        # Check if there's enough data for both images and features
        if len(self.images) > 1 and len(self.features) > 1:
            # Create t-SNE instances and perform transformations
            tsne_images = TSNE(n_components=2, perplexity=min(len(self.images) - 1, len(self.features) - 1), random_state=42)
            tsne_results_images = tsne_images.fit_transform(self.images)

            tsne_features = TSNE(n_components=2, perplexity=min(len(self.features) - 1, len(self.images) - 1), random_state=42)
            tsne_results_features = tsne_features.fit_transform(self.features)

            # Set up a unique label array for both images and features, assuming they share labels
            unique_labels = np.unique(self.img_labels)
            num_classes = len(unique_labels)

            # Define a colormap with enough colors
            cmap = plt.get_cmap('nipy_spectral', num_classes)

            plt.figure(figsize=(12, 10))

            # Plot images
            scatter_images = plt.scatter(tsne_results_images[:, 0], tsne_results_images[:, 1], c=self.img_labels,
                                         cmap=cmap, alpha=0.6, edgecolors='w', linewidths=1, marker='^',
                                         label='Images')

            # Plot features
            scatter_features = plt.scatter(tsne_results_features[:, 0], tsne_results_features[:, 1], c=self.feature_labels,
                                           cmap=cmap, alpha=0.6, edgecolors='w', linewidths=0.5, marker='o',
                                           label='Features')

            # Configure colorbar and legend
            boundaries = np.arange(-0.5, num_classes + 0.5, 1)
            norm = BoundaryNorm(boundaries, ncolors=num_classes, clip=True)
            cbar = plt.colorbar(scatter_images, ticks=np.arange(num_classes), boundaries=boundaries,
                                spacing='proportional')
            cbar.set_ticklabels(unique_labels)

            plt.legend(handles=[scatter_images, scatter_features], labels=['Images', 'Features'])

            plt.title('t-SNE Visualization of Images and Features')
            plt.xlabel('t-SNE Dimension 1')
            plt.ylabel('t-SNE Dimension 2')

            plt.show()
        else:
            print("Not enough data to perform t-SNE.")


# Usage
path = "/home/yimoning/mcmaster/SwinIR/data/val_in/lr"
visualizer = ImageTSNEVisualizer(path)
visualizer.plot_tsne()