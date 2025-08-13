from skimage.io import imread
import anndata 
from pathlib import Path
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from tifffile import imread
from tqdm import tqdm

import sys
sys.path.append("/work/FAC/FBM/DBC/mrapsoma/prometex/projects/ProtoPNet/GNN_spatial")
sys.path.append("/work/FAC/FBM/DBC/mrapsoma/prometex/projects/ProtoPNet/GNN_spatial/src")
from src.gnn_spatial.NSCLC.gen_pca import get_raw_pixel_matrix, preprocess
from data_utils.NSCLCDataModule import NsclcImageDataset

nsclc_path = Path('/work/FAC/FBM/DBC/mrapsoma/prometex/data/NSCLC/02_processed')

mask_path = nsclc_path / 'masks/20250430_cell_masks'
anndata_base_path = nsclc_path / 'export/20250512_adata_graphs'
all_cells_anndata_path = nsclc_path / 'sce_objects/sce.h5ad'
metadata_path = nsclc_path / 'metadata/clinical.parquet'
img_path = nsclc_path / 'images/images_mcd'

df_metadata = pd.read_parquet(metadata_path)

"""
Helper script that takes a folder as input and generates a PCA on the dna-free, arcsinh'ed, 
"""

if __name__ == "__main__":
    print("getting stats input...")
    raw_pixel_dict = get_raw_pixel_matrix(from_aggregated=False)
    _, summary_dict = preprocess(raw_pixel_dict)
    print("...done.")
    
    print("dumping stats...")
    from joblib import dump, load
    # dump(pca, 'pca_model_no_dna_normalized.joblib')
    dump(summary_dict, 'preprocessing_stats.joblib')
    print("...done")