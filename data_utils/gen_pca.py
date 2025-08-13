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
        
def get_mask(patient_id):
    patient_mask_path = mask_path / f"{patient_id}.tiff"
    return imread(patient_mask_path)

def get_anndata(patient_id): 
    return anndata.read_h5ad(anndata_base_path / f'{patient_id}.h5ad')

def get_tumor_type(patient_id):
    return get_anndata(patient_id).obs['DX.name'].iloc[0]

def get_image(patient_id):
    patient_img_path = img_path / f"{patient_id}.tiff"
    return imread(patient_img_path)

import tifffile

from tqdm import tqdm

IRIDIUM_INDICES = [38, 39]

def get_raw_pixel_matrix(from_aggregated=True, file_limit=None):
    def get_patient_id(img_path):
        # Convert Path object to string if needed
        img_path = str(img_path)
        return img_path.split("/")[-1].split(".")[0]
    
    # Define source directory based on the flag
    base_dir = Path('/work/FAC/FBM/DBC/mrapsoma/prometex/data/NSCLC/02_processed')
    if from_aggregated:
        src_dir = base_dir / 'images/images_mcd_masked_averaged'
    else:
        src_dir = base_dir / 'images/images_mcd_masked'

    pixel_list = []
    
    # Check if directory exists
    if not src_dir.exists():
        raise FileNotFoundError(f"Directory not found: {src_dir}")
    
    # Iterate through image files in the directory
    file_count = 0 

    for image_path in tqdm(src_dir.glob('*.tiff')):  # assuming they're TIFF files
        if file_limit is not None:
            if file_count > file_limit:
                break

        try:
            image = tifffile.imread(image_path)
            patient_id = get_patient_id(image_path)  # pass image_path, not image
            mask = get_mask(patient_id).astype(np.uint32)  # assuming get_mask is defined elsewhere
            nonzero_indices = np.nonzero(mask)
            
            # Check if image and mask dimensions match
            if image.shape[1:] != mask.shape:
                raise ValueError(f"Image and mask dimensions don't match for {patient_id}")
            pixel_array = image[:, nonzero_indices[0], nonzero_indices[1]]
            # pixel_array = np.delete(pixel_array, IRIDIUM_INDICES, axis=0)
            pixel_list.append(pixel_array)
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            continue

    return np.concatenate(pixel_list, axis=1)

def preprocess(raw_pixel_array):   
    raw_pixel_array = np.arcsinh(raw_pixel_array)
    summary_dict = {
        'min': [],
        'max': [],
        'clip_lower': [],
        'clip_upper': [],
    }

    for k in range(raw_pixel_array.shape[0]):
        lower_bound = np.percentile(raw_pixel_array[k], 0)
        upper_bound = np.percentile(raw_pixel_array[k], 99)

        raw_pixel_array[k] = np.clip(raw_pixel_array[k], lower_bound, upper_bound)

        min_val = raw_pixel_array[k].min()
        max_val = raw_pixel_array[k].max()

        raw_pixel_array[k] = (raw_pixel_array[k] - min_val) / (max_val - min_val)

        summary_dict['min'].append(min_val)
        summary_dict['max'].append(max_val)
        summary_dict['clip_lower'].append(lower_bound)
        summary_dict['clip_upper'].append(upper_bound)

    return raw_pixel_array, summary_dict 

from sklearn.decomposition import PCA
import numpy as np

def create_pca_transform(pca_input, n_components=None):
    """
    Creates a PCA transform that maps (43, n) arrays to (k, n) arrays.
    
    Args:
        pixel_list: List of (43, m_i) arrays to train PCA on
        n_components: Number of components to keep (if None, keeps all)
    
    Returns:
        pca: Trained PCA object with transform method
    """
    # Transpose to sklearn's expected format (n_samples, n_features)
    X = pca_input.T  # Shape: (total_pixels, 43)
    
    # Create and fit PCA
    pca = PCA(n_components=n_components)
    pca.fit(X)
    
    return pca

if __name__ == "__main__":
    print("getting pca input...")
    pca_input = get_raw_pixel_matrix(from_aggregated=False)
    processed_pca_input, summary_dict = preprocess(pca_input)
    print("...done.")

    print("fitting pca...")
    # pca = create_pca_transform(processed_pca_input, n_components=3)
    print("...done.")

    print("dumping pca...")
    from joblib import dump, load
    # dump(pca, 'pca_model_no_dna_normalized.joblib')
    dump(summary_dict, 'preprocessing_stats_43dim.joblib')
    print("...done")

    print("dumped pca")
