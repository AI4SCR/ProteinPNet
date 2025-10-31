import sys
sys.path.append("/work/FAC/FBM/DBC/mrapsoma/prometex/projects/ProtoPNet")
import torch
import os
import argparse
from pathlib import Path
import joblib

from settings import img_size, train_push_batch_size
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import pandas as pd
from tqdm import tqdm
import anndata
import numpy as np

# Path configurations
nsclc_path = Path('/work/FAC/FBM/DBC/mrapsoma/prometex/data/NSCLC/02_processed')
metadata_path = nsclc_path / 'metadata/clinical.parquet'
anndata_base_path = nsclc_path / 'export/20250512_adata_graphs'


def get_anndata(patient_id): 
    return anndata.read_h5ad(anndata_base_path / f'{patient_id}.h5ad')

# Load metadata
df_metadata = pd.read_parquet(metadata_path)

def global_cell_id_to_feature_dict(global_cell_id):
    """Convert global cell ID to feature dictionary"""
    sample_id = "_".join(global_cell_id.split("_")[:-1])
    anndata_obj = get_anndata(sample_id)
    cell_idx = int(global_cell_id.split("_")[-1]) - 1

    feature_dict = {
        "sample_id": sample_id,
        "global_cell_id": global_cell_id,
        "expression_values": anndata_obj.X[cell_idx],
        "cell_type_res_0": anndata_obj.obs.loc[global_cell_id]['cell_category'],
        "cell_type_res_1": anndata_obj.obs.loc[global_cell_id]['cell_type'],
        "cell_type_res_2": anndata_obj.obs.loc[global_cell_id]['cell_subtype'],
        "centroid_x": anndata_obj.obs.loc[global_cell_id]['Center_X'],
        "centroid_y": anndata_obj.obs.loc[global_cell_id]['Center_Y'],
    }
    return feature_dict

def generate_global_cell_matrix_chunk(dataset, chunk_num, total_chunks, output_dir):
    """Generate a chunk of the global cell matrix with evenly distributed remainder"""
    all_cells = []
    total_samples = len(dataset)
    
    # Calculate base chunk size and remainder
    base_chunk_size = total_samples // total_chunks
    remainder = total_samples % total_chunks
    
    # Calculate start and end indices with distributed remainder
    if chunk_num <= remainder:
        # Earlier chunks get an extra sample
        start_idx = (chunk_num - 1) * (base_chunk_size + 1)
        end_idx = start_idx + (base_chunk_size + 1)
    else:
        # Later chunks get base size
        start_idx = remainder * (base_chunk_size + 1) + (chunk_num - remainder - 1) * base_chunk_size
        end_idx = start_idx + base_chunk_size
    
    print(f"Processing chunk {chunk_num} of {total_chunks} (samples {start_idx}-{end_idx-1})")
    
    for idx in range(start_idx, end_idx):
        sample_path, _ = dataset.samples[idx]
        sample_id = os.path.basename(sample_path).split(".")[0]
        anndata_obj = get_anndata(sample_id)
        
        for global_cell_id in tqdm(list(anndata_obj.obs.index), desc=f"Sample {idx}.{sample_id}"):
            feature_dict = global_cell_id_to_feature_dict(global_cell_id)
            all_cells.append(feature_dict)
    
    chunk_df = pd.DataFrame(all_cells)
    output_path = Path(output_dir) / f"data_matrix_part_{chunk_num}_of_{total_chunks}.pkl"
    joblib.dump(chunk_df, output_path)

def main():
    parser = argparse.ArgumentParser(description='Generate global cell matrix in chunks')
    parser.add_argument('--chunk_num', type=int, required=True, help='Current chunk number (1-based)')
    parser.add_argument('--total_chunks', type=int, required=True, help='Total number of chunks')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    train_dir = "/work/FAC/FBM/DBC/mrapsoma/prometex/projects/ProtoPNet/datasets/nsclc/train_normed_cropped"
    train_push_dataset = datasets.ImageFolder(
        train_dir,
        transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
        ]))
    
    # Process the specified chunk
    generate_global_cell_matrix_chunk(
        train_push_dataset,
        chunk_num=args.chunk_num,
        total_chunks=args.total_chunks,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()