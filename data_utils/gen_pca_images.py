import joblib
from skimage.io import imread
# import anndata 
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
# from src.gnn_spatial.NSCLC.gen_pca import IRIDIUM_INDICES, get_raw_pixel_matrix, preprocess
# from data_utils.NSCLCDataModule import NsclcImageDataset

nsclc_path = Path('/work/FAC/FBM/DBC/mrapsoma/prometex/data/NSCLC/02_processed')
pca_path = Path('/work/FAC/FBM/DBC/mrapsoma/prometex/data/datasets/PCa/02_processed')

mask_path = nsclc_path / 'masks/20250430_cell_masks'
anndata_base_path = nsclc_path / 'export/20250512_adata_graphs'
all_cells_anndata_path = nsclc_path / 'sce_objects/sce.h5ad'
metadata_path = nsclc_path / 'metadata/clinical.parquet'
# img_path = nsclc_path / 'images/images_mcd'

df_metadata = pd.read_parquet(metadata_path)

"""
Helper script that takes a folder as input and generates a PCA on the dna-free, arcsinh'ed, 
"""

def gen_pca_image(patient_id, output_dir, pca, stats):
    patient_id = patient_id + ".tiff"
    img_pth = pca_path / 'images/filtered' / patient_id
    image = tifffile.imread(img_pth)

    # if no dna for pca, no dna for output
    # if pca.n_features_in_ == 41:
    #     image = np.delete(image, IRIDIUM_INDICES, axis=0)

    image = np.arcsinh(image)
    for k in range(image.shape[0]):
        image[k] = np.clip(image[k], stats['clip_lower'][k], stats['clip_upper'][k])
        image[k] = (image[k] - stats['min'][k]) / (stats['max'][k] - stats['min'][k])

    original_image_shape = image.shape
    image = image.reshape(image.shape[0], -1)
    new_image = pca.transform(image.T).T
    new_image = new_image.reshape((3, *original_image_shape[1:]))
    output_path = Path(output_dir) / Path(patient_id)
    tifffile.imwrite(output_path, new_image)

    return new_image

if __name__ == "__main__":
    print("computing pca images...")
    pca = joblib.load("/work/FAC/FBM/DBC/mrapsoma/prometex/projects/ProtoPNet/pca_pca_model.joblib")
    stats = joblib.load("/work/FAC/FBM/DBC/mrapsoma/prometex/projects/ProtoPNet/pca_preprocessing_stats_43dim.joblib")
    for img_pth in tqdm(Path('/work/FAC/FBM/DBC/mrapsoma/prometex/data/datasets/PCa/02_processed/images/filtered').glob("*.tiff")):
        patient_id = str(img_pth).split("/")[-1].split(".")[0]
        gen_pca_image(patient_id, '/work/FAC/FBM/DBC/mrapsoma/prometex/projects/ProtoPNet/datasets/pca_pca', pca, stats)
    print("...done.")
    
    # print("dumping stats...")
    # from joblib import dump, load
    # # dump(pca, 'pca_model_no_dna_normalized.joblib')
    # dump(summary_dict, 'preprocessing_stats.joblib')
    # print("...done")