import sys
sys.path.append("/work/FAC/FBM/DBC/mrapsoma/prometex/projects/ProtoPNet")
import torch
import os

from settings import base_architecture, num_classes, train_batch_size, train_push_batch_size, img_size, prototype_shape, prototype_activation_function, add_on_layers_type, ablate_prototype_selection
import model
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import pandas as pd

from tqdm import tqdm
import joblib
import cv2

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
from matplotlib import cm

import anndata
from tifffile import imread
from pathlib import Path

nsclc_path = Path('/work/FAC/FBM/DBC/mrapsoma/prometex/data/NSCLC/02_processed')

mask_path = nsclc_path / 'masks/20250430_cell_masks'
anndata_base_path = nsclc_path / 'export/20250512_adata_graphs'
all_cells_anndata_path = nsclc_path / 'sce_objects/sce.h5ad'
metadata_path = nsclc_path / 'metadata/clinical.parquet'
img_path = nsclc_path / 'images/images_mcd'

df_metadata = pd.read_parquet(metadata_path)

from get_posthoc_metrics import *

# 1. load dataloader
train_dir = "/work/FAC/FBM/DBC/mrapsoma/prometex/projects/ProtoPNet/datasets/nsclc/train_normed_cropped"
base = "/work/FAC/FBM/DBC/mrapsoma/prometex/projects/ProtoPNet/saved_models"
# run = "resnet152/electric-deluge-9"
run = "resnet152/azure-dream-77"
checkpoint = "30_18push0.8098"
resolution = 1
# checkpoint = "60_9push0.8037"
artifact_path = os.path.join(base, run, "artifacts")
os.makedirs(artifact_path, exist_ok=True)
train_push_dir = train_dir

train_push_dataset = datasets.ImageFolder(
    train_push_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
    ]))
train_push_loader = torch.utils.data.DataLoader(
    train_push_dataset, batch_size=train_push_batch_size, shuffle=False,
    num_workers=1, pin_memory=False)

# 2. load model 
state_dict_path = f"{base}/{run}/{checkpoint}.pth"
ppnet = torch.load(state_dict_path, weights_only=False, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))  # Add map_location if needed

# 3. find top k images per prototype
save_dir = os.path.join(base, run, "prototype_image_centers", checkpoint)
os.makedirs(save_dir, exist_ok=True)

# new one for new cell masks
def global_cell_id_to_feature_dict(anndata_obj, global_cell_id):
    sample_id = "_".join(global_cell_id.split("_")[:-1])

    expression_values = anndata_obj.X[int(global_cell_id.split("_")[-1]) - 1]
    cell_type_res_0 = anndata_obj.obs.loc[global_cell_id]['cell_category']
    cell_type_res_1 = anndata_obj.obs.loc[global_cell_id]['cell_type']
    cell_type_res_2 = anndata_obj.obs.loc[global_cell_id]['cell_subtype']
    centroid_x = anndata_obj.obs.loc[global_cell_id]['Center_X']
    centroid_y = anndata_obj.obs.loc[global_cell_id]['Center_Y']

    feature_dict = {
        "sample_id": sample_id,
        "global_cell_id": global_cell_id,
        "expression_values": expression_values,
        "cell_type_res_0": cell_type_res_0,
        "cell_type_res_1": cell_type_res_1,
        "cell_type_res_2": cell_type_res_2,
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
    }

    return feature_dict

import numpy as np
from typing import Dict, Any

def compute_cell_features(image: np.ndarray) -> Dict[int, Dict[str, Any]]:
    """
    Compute cell features from a 3-channel image.
    
    Args:
        image: numpy array of shape (h, w, 3) where:
            channel 0: cell IDs
            channel 1: prototype induced activation values
            channel 2: cell type mask (cell type IDs)
    
    Returns:
        Dictionary mapping cell_id -> features dictionary containing:
            - median_activation: median activation value within the cell
            - mean_activation: mean activation value within the cell
            - activation_ranking: ranking of this cell's median activation among all cells
            - activation_percentile: percentile of this cell's median activation among all cells
            - cell_type: the cell type ID
            - pixel_count: number of pixels in the cell
    """
    # Extract channels
    cell_ids = image[:, :, 0]
    activations = image[:, :, 1]
    
    # Get unique cell IDs (excluding background/zero)
    unique_cells = np.unique(cell_ids)
    unique_cells = unique_cells[unique_cells != 0]  # Remove background
    
    # Dictionary to store results
    cell_features = {}
    
    # List to store median activations for all cells for ranking
    all_median_activations = []
    
    # First pass: compute basic features for each cell
    for cell_id in unique_cells:
        # Create mask for this cell
        cell_mask = cell_ids == cell_id
        
        # Get activation values within this cell
        cell_activation_values = activations[cell_mask]
        
        # Compute median and mean activation
        median_activation = np.median(cell_activation_values)
        mean_activation = np.mean(cell_activation_values)
        
        # Store features
        cell_features[int(cell_id)] = {
            'median_activation': float(median_activation),
            'mean_activation': float(mean_activation),
            'activation_values': cell_activation_values  # Store for ranking
        }
        
        all_median_activations.append(median_activation)
    
    # Convert to numpy array for ranking
    all_median_activations = np.array(all_median_activations)
    
    # Second pass: compute rankings and percentiles
    for cell_id, features in cell_features.items():
        median_activation = features['median_activation']
        
        # Compute ranking (higher activation = better rank)
        # argsort gives indices in ascending order, so we reverse for descending
        sorted_indices = np.argsort(all_median_activations)[::-1]  # Descending order
        ranking = np.where(sorted_indices == list(cell_features.keys()).index(cell_id))[0][0] + 1
        
        # Compute percentile (percentage of cells with lower median activation)
        percentile = (np.sum(all_median_activations <= median_activation) / len(all_median_activations)) * 100
        
        # Add ranking and percentile to features
        features['activation_ranking'] = int(ranking)
        features['activation_percentile'] = float(percentile)
        
        # Remove the raw activation values to save memory
        del features['activation_values']
    
    return cell_features


def generate_prototype_specific_cell_matrix(prototype_masks):
    feature_dicts = []

    for prototype_num, index_mask_dict in enumerate(prototype_masks):
        for rank, (index, prototype_induced_mask) in enumerate(index_mask_dict.items()):
            sample_id = train_push_dataset.samples[index][0].split("/")[-1].split(".")[0]
            cell_features = compute_cell_features(prototype_induced_mask)
            # input:
                # h, w, c image where ch 0 is cell ids, ch 1 is prototype induced activation values, ch 2 is cell type mask
            # output:
                # a dict of cell_id -> features
                # for each cell: 
                    # get the median activation value within the cell via channel 1
                    # give the activation value 
                    # give the ranking within the cell
                    # give the percentile over all cells in the image
            # cell_id_mask = get_mask(sample_id)
            # relevant_cell_ids = np.unique(cell_id_mask * prototype_induced_mask)
            # relevant_cell_ids = relevant_cell_ids[relevant_cell_ids > 0]
            anndata_obj = get_anndata(sample_id)
            global_cell_ids = [f"{sample_id}_{cell_id}" for cell_id in cell_features.keys()]

            for global_cell_id in tqdm(global_cell_ids, desc=f"Processing global cell IDs for prototype {prototype_num} and mask {index}"):
                feature_dict = global_cell_id_to_feature_dict(anndata_obj, global_cell_id)
                feature_dict['image_rank'] = rank
                feature_dict['prototype_num'] = prototype_num

                cell_id = int(global_cell_id.split("_")[-1])
                for key in cell_features[cell_id]:
                    feature_dict[key] = cell_features[cell_id][key]

                feature_dicts.append(feature_dict)


    return pd.DataFrame(feature_dicts).set_index('global_cell_id')

obj = joblib.load("/work/FAC/FBM/DBC/mrapsoma/prometex/projects/ProtoPNet/saved_models/resnet152/azure-dream-77/artifacts/cell_prototype_masks_res1.joblib")
prototype_masks = obj[0]
prototype_specific_dataframe = generate_prototype_specific_cell_matrix(prototype_masks)
prototype_specific_dataframe.to_parquet(os.path.join(artifact_path, "prototype_specific_cell_matrix.parquet"))