import sys
sys.path.append("/work/FAC/FBM/DBC/mrapsoma/prometex/projects/ProtoPNet/GNN_spatial")
sys.path.append("/work/FAC/FBM/DBC/mrapsoma/prometex/projects/ProtoPNet/GNN_spatial/src")
from src.gnn_spatial.dataset.NSCLCload import NSCLC
import torch 
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
import pickle

import os 
from imageio import imread, imsave
from pathlib import Path
import re
import anndata
import pickle
import psutil
import shutil
import numpy as np
import pandas as pd
import skimage
from pathlib import Path

import networkx as nx

import psutil
import gc
from scipy.sparse import csr_matrix
from tqdm import tqdm
from skimage.measure import regionprops




class NSCLC_Dataset(LightningDataModule):
    def __init__(self,
                 subgraphs,
                 graph_type, 
                 batch_size,
                 patient_ids_array,
                 use_sampler): 
        #('hello')

        self.subgraphs = subgraphs
        self.graph_type = graph_type
        self.batch_size = batch_size
        self.patient_ids_array = patient_ids_array
        self.use_sampler = use_sampler

        self.NSCLS_path = Path('/work/FAC/FBM/DBC/mrapsoma/prometex/data/NSCLC/')

        self.masks_path_new = self.NSCLS_path / '02_processed/masks/20250430_cell_masks/'
        self.adatas_path_new = self.NSCLS_path / '02_processed/export/20250512_adata_graphs/'
        self.all_cells_adata_path_new = self.NSCLS_path / '02_processed/sce_objects/sce.h5ad'
        metadata_path = self.NSCLS_path / '02_processed/metadata/clinical.parquet'
        
        self.df_metadata = pd.read_parquet(metadata_path)


        self.data = self.read_data_new()



    def __getitem__(self, idx) :

        id = self.data['ids'][idx]
        patient_id = self.data['patient_id'][idx]
        #adj = self.data['adj_matrix'][idx]
        adj = 1
        nodes = self.data['nodes'][idx]
        edges = self.data['edges'][idx]
        protein_expression_vector = torch.tensor(self.data['protein_expressions'][idx], dtype=torch.float)
        subgraph_mask_ids = self.data['mask_ids'][idx]
        subgraph_cancer_grade_ids = self.data['cancer_grade_ids'][idx]
        subgraph_smoker_ids = self.data['smoker_ids'][idx]
        subgraph_relapse_ids = self.data['relapse_ids'][idx]
        subgraph_cell_category_ids = self.data['cell_categories'][idx]
        subgraph_cell_type_ids = self.data['cell_types'][idx]
        subgraph_cell_subtype_ids = self.data['cell_subtypes'][idx]
        subgraph_tumor_type_ids = self.data['tumor_types'][idx]
        subgraph_centroids = self.data['centroids'][idx]

        return id, patient_id, adj, nodes, edges, protein_expression_vector, subgraph_mask_ids, subgraph_cancer_grade_ids, subgraph_smoker_ids, subgraph_relapse_ids, subgraph_cell_category_ids, subgraph_cell_type_ids, subgraph_cell_subtype_ids, subgraph_tumor_type_ids, subgraph_centroids
    
    def __len__(self):
        return len(self.data['ids'])
    

    def collate_fn(self, batch):

        batch_ids = [batch[i][0] for i in range(len(batch))]
        batch_patient_ids = [batch[i][1] for i in range(len(batch))]
        batch_adjs = [batch[i][2] for i in range(len(batch))]

        batch_nodes = [batch[i][3] for i in range(len(batch))]
        batch_nodes = [g[0] for g in batch_nodes]

        batch_edges = [batch[i][4] for i in range(len(batch))]
        #batch_edges = [g[0] for g in batch_edges]

        batch_protein_expression_vectors = [batch[i][5] for i in range(len(batch))]

        batch_subgraph_mask_ids = [batch[i][6] for i  in range(len(batch))]

        batch_cancer_type_ids = [batch[i][7] for i  in range(len(batch))]
        batch_smoker_ids = [batch[i][8] for i  in range(len(batch))]
        batch_relapse_ids = [batch[i][9] for i  in range(len(batch))]

        batch_cell_categories = [batch[i][10] for i  in range(len(batch))]
        batch_cell_types = [batch[i][11] for i  in range(len(batch))]
        batch_cell_subtypes = [batch[i][12] for i  in range(len(batch))]
        batch_tumor_types = [batch[i][13] for i  in range(len(batch))]
        batch_centroids = [batch[i][14] for i  in range(len(batch))]


        max_edges = 0
        reindex_batch_edges = []

        i = 0

        for edges_i in batch_edges :

            reindex_edges_i = [(x + max_edges, y + max_edges) for x, y in edges_i]

            reindex_batch_edges.append(reindex_edges_i)

            max_edges = max_edges + np.max(edges_i)

            i += 1

        flatten_reindex_batch_edges = [item for sublist in reindex_batch_edges for item in sublist]


        final_batch = {}
        final_batch['ids'] = batch_ids
        final_batch['patient_ids'] = batch_patient_ids
        #final_batch['adjs'] = batch_adjs
        #final_batch['nodes'] = batch_nodes
        final_batch['edges'] = flatten_reindex_batch_edges
        final_batch['protein_expression_vectors'] = batch_protein_expression_vectors
        final_batch['subgraph_mask_ids'] = batch_subgraph_mask_ids
        final_batch['cancer_type_ids'] = batch_cancer_type_ids
        final_batch['smoker_ids'] = batch_smoker_ids
        final_batch['relapse_ids'] = batch_relapse_ids
        final_batch['cell_categories'] = batch_cell_categories
        final_batch['cell_types'] = batch_cell_types
        final_batch['cell_subtypes'] = batch_cell_subtypes
        final_batch['tumor_types'] = batch_tumor_types
        final_batch['centroids'] = batch_centroids
        final_batch['not_reindexed_edges'] = batch_edges


        return final_batch
    


    def read_data_new(self) : 

        subgraph_id = 0
        subgraph_ids = []
        subgraph_patient_id = []
        subgraph_adj = []
        subgraph_nodes = []
        subgraph_edges= []
        subgraph_protein_expressions = []
        subgraph_mask_ids = []
        subgraph_cancer_grade_ids = []
        subgraph_smoker_ids = []
        subgraph_relapse_ids = []
        subgraph_cell_categories = []
        subgraph_cell_types = []
        subgraph_cell_subtypes = []
        subgraph_tumor_types = []
        subgraph_centroids = []

        #for ids, patient_id in enumerate(self.patient_ids_array) :
        for ids, patient_id in enumerate(tqdm(self.patient_ids_array, desc="Processing patients")):
            #print('patinet ID ', patient_id)
            
            # check h5ad exists
            file_path = self.adatas_path_new / f'{patient_id}.h5ad'
            if not file_path.exists():
                print(f"File not found for patient ID {patient_id}, skipping...")
                continue

            adata = anndata.read_h5ad(self.adatas_path_new / f'{patient_id}.h5ad')
            #mask = imread(self.masks_path_new / f"{patient_id}.tiff")

            # IMPORTANT !!!! make sure order cells align with order IDs in the mask and this in the graph
            adata = adata[adata.obs.sort_values(by="CellNumber").index]

            adj_sparse = adata.uns[str(self.graph_type)]['adj']
            # make sure there will be no self connections in the grap
            #adj_sparse.setdiag(0)

            centroids = adata.obsm['centroids']
            obs = adata.obs

            cancer_grade_id = obs['Grade'].iloc[0]
            smoker_id = obs['Smok'].iloc[0]
            relapse_id = obs['OS'].iloc[0]
            tumor_type = obs['DX.name'].iloc[0]

            if tumor_type != 'Adenocarcinoma' and tumor_type != 'Squamous cell carcinoma':
                continue
            #tumor_type = self.df_metadata.loc[str(patient_id)]['typ']

            if not self.subgraphs : 
                G = nx.from_scipy_sparse_array(adj_sparse)
                my_list_nodes = list(G.nodes)
                my_list_edges = list(G.edges)

                mask_id = np.full(len(my_list_nodes), subgraph_id)
                edges_mask_id = np.full(len(my_list_edges), subgraph_id)

                cell_category = adata.obs['cell_category'].to_numpy()
                cell_type = adata.obs['cell_type'].to_numpy()
                cell_subtype = adata.obs['cell_subtype'].to_numpy()

                subgraph_ids.append(subgraph_id)
                subgraph_patient_id.append(patient_id)
                #subgraph_adj.append(adj_mask)
                subgraph_nodes.append(my_list_nodes)
                subgraph_edges.append(my_list_edges)
                subgraph_protein_expressions.append(adata.X)
                subgraph_mask_ids.append(mask_id)
                subgraph_cancer_grade_ids.append(cancer_grade_id)
                subgraph_smoker_ids.append(smoker_id)
                subgraph_relapse_ids.append(relapse_id)
                subgraph_cell_categories.append(cell_category)
                subgraph_cell_types.append(cell_type)
                subgraph_cell_subtypes.append(cell_subtype)
                subgraph_tumor_types.append(tumor_type)
                subgraph_centroids.append(centroids)

                subgraph_id += 1


            if self.subgraphs : 

                # max x and y coordinates 
                x_coordinate_max = np.max(centroids[:, 0])
                y_coordinate_max = np.max(centroids[:, 1])

                # generate intervals for x and y axis
                x_interval_step = x_coordinate_max / 3
                x_intervals = [x_interval_step * i for i in range(1, 4)]
                y_interval_step = y_coordinate_max / 3
                y_intervals = [y_interval_step * i for i in range(1, 4)]

                # compute masks for the different 9 patches
                mask_region_1 = self.create_mask(centroids, 0, x_intervals[0], 0, y_intervals[0])
                mask_region_2 = self.create_mask(centroids, 0, x_intervals[0], y_intervals[0], y_intervals[1])
                mask_region_3 = self.create_mask(centroids, 0, x_intervals[0], y_intervals[1], y_intervals[2])

                mask_region_4 = self.create_mask(centroids, x_intervals[0], x_intervals[1], 0, y_intervals[0])
                mask_region_5 = self.create_mask(centroids, x_intervals[0], x_intervals[1], y_intervals[0], y_intervals[1])
                mask_region_6 = self.create_mask(centroids, x_intervals[0], x_intervals[1], y_intervals[1], y_intervals[2])

                mask_region_7 = self.create_mask(centroids, x_intervals[1], x_intervals[2], 0, y_intervals[0])
                mask_region_8 = self.create_mask(centroids, x_intervals[1], x_intervals[2], y_intervals[0], y_intervals[1])
                mask_region_9 = self.create_mask(centroids, x_intervals[1], x_intervals[2], y_intervals[1], y_intervals[2])

                masks = [mask_region_1, mask_region_2, mask_region_3,
                        mask_region_4, mask_region_5, mask_region_6,
                        mask_region_7, mask_region_8, mask_region_9]
                

                for mask in masks : 
                    
                    x = adata.X
                    x_mask = x[mask]
                    adj_mask = adj_sparse[mask, :][:, mask]


                    cell_category = adata.obs['cell_category'].to_numpy()[mask]
                    cell_type = adata.obs['cell_type'].to_numpy()[mask]
                    cell_subtype = adata.obs['cell_subtype'].to_numpy()[mask]

                    assert x_mask.shape[0] == adj_mask.shape[0], f"Shape mismatch: x_mask has shape {x_mask.shape}, adj_mask has shape {adj_mask.shape}"

                    G = nx.Graph()
                    G = nx.from_scipy_sparse_array(adj_mask)
                    
                    if len(G.edges) == 0:
                        #print(f"Skipping subgraph {subgraph_id} due to no edges")
                        continue
                    
                    my_list_nodes = list(G.nodes)
                    my_list_edges = list(G.edges)

                    mask_id = np.full(len(my_list_nodes), subgraph_id)

                    subgraph_ids.append(subgraph_id)
                    subgraph_patient_id.append(patient_id)
                    #subgraph_adj.append(adj_mask)
                    subgraph_nodes.append(my_list_nodes)
                    subgraph_edges.append(my_list_edges)
                    subgraph_protein_expressions.append(x_mask)
                    subgraph_mask_ids.append(mask_id)
                    subgraph_cancer_grade_ids.append(cancer_grade_id)
                    subgraph_smoker_ids.append(smoker_id)
                    subgraph_relapse_ids.append(relapse_id)
                    subgraph_cell_categories.append(cell_category)
                    subgraph_cell_types.append(cell_type)
                    subgraph_cell_subtypes.append(cell_subtype)
                    subgraph_tumor_types.append(tumor_type)
                    subgraph_centroids.append(centroids[mask])

                    subgraph_id += 1

                del adata
                del mask
                gc.collect()

        data = {
            'ids': subgraph_ids,
            'patient_id': subgraph_patient_id,
            'adj_matrix': subgraph_adj,
            'nodes': subgraph_nodes,
            'edges': subgraph_edges,
            'protein_expressions': subgraph_protein_expressions,
            'mask_ids': subgraph_mask_ids,
            'cancer_grade_ids': subgraph_cancer_grade_ids,
            'smoker_ids': subgraph_smoker_ids,
            'relapse_ids': subgraph_relapse_ids,
            'tumor_types': subgraph_tumor_types,
            'cell_categories': subgraph_cell_categories,
            'cell_types': subgraph_cell_types,
            'cell_subtypes': subgraph_cell_subtypes,
            'centroids' : subgraph_centroids,
        }

        mem_after = psutil.Process().memory_info().rss / 1024 ** 2  
        print(f"Memory usage {mem_after:.2f} MB, ")
    
        return data


    
    def create_mask(self, cell_centroid_coordinates, x_min, x_max, y_min, y_max):
        return (cell_centroid_coordinates[:, 0] >= x_min) & (cell_centroid_coordinates[:, 0] < x_max) & \
            (cell_centroid_coordinates[:, 1] >= y_min) & (cell_centroid_coordinates[:, 1] < y_max)
            


    def dataloader(self, sampler) : 

        if self.use_sampler : 

            batch_sampler = torch.utils.data.sampler.BatchSampler(
                sampler,
                batch_size = self.batch_size,
                drop_last = True
            )
            return DataLoader(self, num_workers = 0, collate_fn=self.collate_fn, batch_sampler=batch_sampler)

        else : 

            return DataLoader(self, num_workers=0, batch_size=self.batch_size, collate_fn=self.collate_fn, drop_last=False)

class NsclcImageDataset(LightningDataModule) :
    def __init__(self,
                 patient_ids_array,
                 use_sampler=False,
                should_average_mask=True
        ): 
        
        self.patient_ids_array = patient_ids_array
        self.use_sampler = use_sampler
        self.should_average_mask = should_average_mask

        self.nsclc_path = Path('/work/FAC/FBM/DBC/mrapsoma/prometex/data/NSCLC/02_processed')

        self.mask_path = self.nsclc_path / 'masks/20250430_cell_masks/'
        self.anndata_path = self.nsclc_path / 'export/20250512_adata_graphs/'
        self.all_cells_anndata_path = self.nsclc_path / 'sce_objects/sce.h5ad'
        self.img_path = self.nsclc_path / 'images/images_mcd_masked_normed_43dim'

        metadata_path = self.nsclc_path / 'metadata/clinical.parquet'
        self.df_metadata = pd.read_parquet(metadata_path)

        self.data = self.read_data_new()

    def __getitem__(self, idx) :
        id = self.data['ids'][idx]
        patient_id = self.data['patient_id'][idx]
        tumor_type = self.data['tumor_type'][idx]
        imc_image = self.get_image(patient_id)
        
        return {
            "id": id,
            "patient_id": patient_id,
            "tumor_type": tumor_type,
            "imc_image": imc_image
        }

    def __len__(self):
        return len(self.data['ids'])
    
    def collate_fn(self, batch):
        """
        Collate function that handles batches containing strings, integers, and PyTorch tensors.
        
        Args:
            batch: List of dictionaries where each dictionary contains keys with string,
                integer, or tensor values.
                
        Returns:
            A dictionary with the same keys as the input dictionaries. Values are:
            - Lists of strings for string inputs
            - Lists of integers for integer inputs
            - Stacked tensors for tensor inputs
        """
        if not batch:
            return {}
        
        # Get the keys from the first sample in the batch
        keys = batch[0].keys()
        
        collated_batch = {}
        
        for key in keys:
            # Get all values for this key across the batch
            values = [sample[key] for sample in batch]
            
            # Handle different types
            if all(isinstance(v, str) for v in values):
                # For strings, just keep as a list
                collated_batch[key] = values
            elif all(isinstance(v, int) for v in values):
                # For integers, keep as a list or convert to tensor if you prefer
                collated_batch[key] = values
            elif all(isinstance(v, torch.Tensor) for v in values):
                # For tensors, stack along a new dimension
                collated_batch[key] = torch.stack(values)
            else:
                raise TypeError(f"Mixed or unsupported types for key '{key}' in batch")
        
        return collated_batch
    
    def get_anndata(self, patient_id): 
        return anndata.read_h5ad(self.anndata_base_path / f'{patient_id}.h5ad')
    
    def get_pre_mask_image(self, patient_id):
        patient_img_path = self.img_path / f"{patient_id}.tiff"
        im_pre_mask = imread(patient_img_path)
        return im_pre_mask

    def  get_image(self, patient_id):
        img_pre_mask = self.get_pre_mask_image(patient_id)
        mask = self.get_mask(patient_id)

        def get_aggregated_channel(im, mask, channel_idx):
            im_slice = im[channel_idx]
            regions = regionprops(mask, im_slice)
            
            # Create a lookup array (faster than dictionary for NumPy operations)
            max_label = np.max(mask)
            lookup = np.zeros(max_label + 1, dtype=np.float32)  # Index 0 unused (background)
            for region in regions:
                lookup[region.label] = region.mean_intensity
            
            # Vectorized lookup using the mask's labels
            aggregated_channel = np.where(
                mask > 0,
                lookup[mask],  # Direct array indexing
                0              # Leave 0s unchanged
            )
            return aggregated_channel

        def get_aggregated_img(im, mask):
            aggregated_img = np.zeros_like(im)
            for channel in range(im.shape[0]):
                aggregated_img[channel] = get_aggregated_channel(im, mask, channel)
            return aggregated_img
        

        def get_masked_img(im, mask):
            # applies mask to 
            mask = mask > 0
            mask = mask[None, ...]
            mask = mask.repeat(im.shape[0], axis=0)

            return mask * im 

        if self.should_average_mask:
            return get_aggregated_img(img_pre_mask, mask)
        else:
            return get_masked_img(img_pre_mask, mask)
    
    def get_mask(self, patient_id):
        patient_mask_path = self.mask_path / f"{patient_id}.tiff"
        return imread(patient_mask_path)
    
    def get_tumor_type(self, patient_id):
        return self.get_anndata(patient_id).obs['DX.name'].iloc[0]

    def read_data_new(self) : 
        subgraph_id = 0
        subgraph_ids = []
        subgraph_patient_id = []
        subgraph_protein_expression = []
        subgraph_cancer_grade_id = []
        subgraph_smoker_id = []
        subgraph_relapse_id = []
        subgraph_tumor_type = []

        #for ids, patient_id in enumerate(self.patient_ids_array) :
        for ids, patient_id in enumerate(tqdm(self.patient_ids_array, desc="Processing patients")):
            #print('patinet ID ', patient_id)
            
            # check h5ad exists
            file_path = self.anndata_path / f'{patient_id}.h5ad'
            if not file_path.exists():
                print(f"File not found for patient ID {patient_id}, skipping...")
                continue

            adata = anndata.read_h5ad(self.anndata_path / f'{patient_id}.h5ad')

            # IMPORTANT !!!! make sure order cells align with order IDs in the mask and this in the graph
            adata = adata[adata.obs.sort_values(by="CellNumber").index]
            obs = adata.obs

            cancer_grade_id = obs['Grade'].iloc[0]
            smoker_id = obs['Smok'].iloc[0]
            relapse_id = obs['OS'].iloc[0]
            tumor_type = obs['DX.name'].iloc[0]

            if tumor_type != 'Adenocarcinoma' and tumor_type != 'Squamous cell carcinoma':
                continue

            subgraph_ids.append(subgraph_id)
            subgraph_patient_id.append(patient_id)
            subgraph_protein_expression.append(adata.X)
            subgraph_cancer_grade_id.append(cancer_grade_id)
            subgraph_smoker_id.append(smoker_id)
            subgraph_relapse_id.append(relapse_id)
            subgraph_tumor_type.append(tumor_type)

            subgraph_id += 1

            del adata
            gc.collect()

        data = {
            'ids': subgraph_ids,
            'patient_id': subgraph_patient_id,
            'protein_expression': subgraph_protein_expression,
            'cancer_grade_id': subgraph_cancer_grade_id,
            'smoker_id': subgraph_smoker_id,
            'relapse_id': subgraph_relapse_id,
            'tumor_type': subgraph_tumor_type,
        }

        mem_after = psutil.Process().memory_info().rss / 1024 ** 2  
        print(f"Memory usage {mem_after:.2f} MB, ")
    
        return data

    def dataloader(self, sampler=None, batch_size=4): 
        if self.use_sampler: 
            assert sampler is not None

            batch_sampler = torch.utils.data.sampler.BatchSampler(
                sampler,
                batch_size = batch_size,
                drop_last = True
            )
            return DataLoader(self, num_workers = 0, collate_fn=self.collate_fn, batch_sampler=batch_sampler)

        return DataLoader(self, num_workers=0, batch_size=batch_size, collate_fn=self.collate_fn, drop_last=False)







