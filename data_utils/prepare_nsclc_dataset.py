import os
import cv2
import joblib
from skimage.io import imread
import anndata 
from pathlib import Path
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from tifffile import imread
import tifffile
from tqdm import tqdm

import sys
sys.path.append("/work/FAC/FBM/DBC/mrapsoma/prometex/projects/ProtoPNet/GNN_spatial")
sys.path.append("/work/FAC/FBM/DBC/mrapsoma/prometex/projects/ProtoPNet/GNN_spatial/src")
from src.gnn_spatial.NSCLC.gen_pca import IRIDIUM_INDICES, get_mask, get_raw_pixel_matrix, preprocess
from src.gnn_spatial.dataset.NSCLCDataModule import NsclcImageDataset

nsclc_path = Path('/work/FAC/FBM/DBC/mrapsoma/prometex/data/NSCLC/02_processed')

mask_path = nsclc_path / 'masks/20250430_cell_masks'
anndata_base_path = nsclc_path / 'export/20250512_adata_graphs'
all_cells_anndata_path = nsclc_path / 'sce_objects/sce.h5ad'
metadata_path = nsclc_path / 'metadata/clinical.parquet'
img_path = nsclc_path / 'images/images_mcd_masked_normed_43dim'

df_metadata = pd.read_parquet(metadata_path)

"""
Helper script that takes a folder as input and generates a PCA on the dna-free, arcsinh'ed, 
"""
def compute_post_pca_min_max(train_dataset):
    """
    Computes min/max values per PCA component across all images after PCA transformation.
    
    Args:
        train_dataset: Dataset object containing tumor_type and patient_id information
    
    Returns:
        dict: {'min': np.array, 'max': np.array} with shape (3,) for 3 PCA components
    """
    src_dir = Path('/work/FAC/FBM/DBC/mrapsoma/prometex/data/NSCLC/02_processed/images/images_mcd_masked_normed_43dim')
    
    # Initialize with extreme values
    min_values = np.full(43, np.inf)  # 3 PCA components
    max_values = np.full(43, -np.inf)
    
    for tumor_type, patient_id in tqdm(zip(train_dataset.data['tumor_type'], train_dataset.data['patient_id'])):
        patient_id_filename_tiff = patient_id + ".tiff"
        image_path_tiff = src_dir / patient_id_filename_tiff
        
        try:
            # Read PCA-transformed image (shape: [3, height, width])
            img = tifffile.imread(image_path_tiff)
            
            # Update min/max for each component
            for i in range(43):  # For each PCA component
                component_data = img[i]
                min_values[i] = min(min_values[i], np.min(component_data))
                max_values[i] = max(max_values[i], np.max(component_data))
                
        except Exception as e:
            print(f"Error processing {patient_id}: {str(e)}")
            continue
    
    return {
        'min': min_values,
        'max': max_values
    }
    

def prepare_nsclc_files(dataset, min_max_dict, is_train=True):
    src_dir = Path('/work/FAC/FBM/DBC/mrapsoma/prometex/data/NSCLC/02_processed/images/images_mcd_masked_normed_43dim')
    output_dir = Path(f'/work/FAC/FBM/DBC/mrapsoma/prometex/projects/ProtoPNet/datasets/nsclc/{"train_normed_43dim" if is_train else "test_normed_43dim"}')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # with open("/work/FAC/FBM/DBC/mrapsoma/prometex/projects/ProtoPNet/GNN_spatial/src/gnn_spatial/train/splits/train_tumor_typ_20250519.txt", "r") as file:
    #     train_ids = [line.strip().split("\t")[1] for line in file.readlines()]
    # # with open("/work/FAC/FBM/DBC/mrapsoma/prometex/projects/ProtoPNet/GNN_spatial/src/gnn_spatial/train/splits/val_tumor_typ_20250519.txt", "r") as file:
    # #     val_ids = [line.strip().split("\t")[1] for line in file.readlines()]
    # # with open("/work/FAC/FBM/DBC/mrapsoma/prometex/projects/ProtoPNet/GNN_spatial/src/gnn_spatial/train/splits/test_tumor_typ_20250519.txt", "r") as file:
    # #     test_ids = [line.strip().split("\t")[1] for line in file.readlines()]


    # train_dataset = NsclcImageDataset(
    #     patient_ids_array=train_ids,
    #     should_average_mask=False
    # )
    # iterate through the (train, val) ids
    # for each entry in self.data:
    for tumor_type, patient_id in tqdm(zip(dataset.data['tumor_type'], dataset.data['patient_id'])):
        patient_id_filename_tiff = patient_id + ".tiff"
        # read the image in to get (3, m, n) np array
        image_path_tiff = src_dir / patient_id_filename_tiff
        img = tifffile.imread(image_path_tiff)
        # save the image as a png in the appropriate directory
        for dim in range(img.shape[0]):
            min_val, max_val = min_max_dict['min'][dim], min_max_dict['max'][dim]
            img[dim] = (img[dim] - min_val) / (max_val - min_val)

        masked_img = img * get_mask(patient_id).astype(bool)
        # image_cv2_format = np.transpose(masked_img, (1, 2, 0)) * 255.
        #TODO: scale the image by 
        patient_id_filename_png = patient_id + ".tiff"
        type_specific_output_dir = output_dir / tumor_type.replace(" ", "_")
        os.makedirs(type_specific_output_dir, exist_ok=True)
        # if type_specific_output
        image_path_tiff = type_specific_output_dir / patient_id_filename_tiff
        tifffile.imwrite(image_path_tiff, masked_img)
        # cv2.imwrite(image_path_png, cv2.cvtColor(image_cv2_format, cv2.COLOR_BGR2RGB))


if __name__ == "__main__":
    # load train set 
    with open("/work/FAC/FBM/DBC/mrapsoma/prometex/projects/ProtoPNet/GNN_spatial/src/gnn_spatial/train/splits/train_tumor_typ_20250519.txt", "r") as file:
        train_ids = [line.strip().split("\t")[1] for line in file.readlines()]
    with open("/work/FAC/FBM/DBC/mrapsoma/prometex/projects/ProtoPNet/GNN_spatial/src/gnn_spatial/train/splits/val_tumor_typ_20250519.txt", "r") as file:
        val_ids = [line.strip().split("\t")[1] for line in file.readlines()]
    # # with open("/work/FAC/FBM/DBC/mrapsoma/prometex/projects/ProtoPNet/GNN_spatial/src/gnn_spatial/train/splits/test_tumor_typ_20250519.txt", "r") as file:
    # #     test_ids = [line.strip().split("\t")[1] for line in file.readlines()]

    print("constructing train dataset...")
    train_dataset = NsclcImageDataset(
        patient_ids_array=train_ids,
        should_average_mask=False
    )
    print("...done.")

    print("constructing val dataset...")
    val_dataset = NsclcImageDataset(
        patient_ids_array=val_ids,
        should_average_mask=False
    )
    print("...done.")

    print("computing min max...")
    # min_max_dict = compute_post_pca_min_max(train_dataset)
    min_max_dict = {"min": np.array([0.0] * 43), "max": np.array([1.0] * 43)}  # Placeholder for min/max values
    print("...done.")

    print("preparing nsclc dataset...")
    prepare_nsclc_files(train_dataset, min_max_dict, is_train=True)
    prepare_nsclc_files(val_dataset, min_max_dict, is_train=False)
    print("...done")
