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
import cv2

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
        
def get_mask(patient_id):
    patient_mask_path = mask_path / f"{patient_id}.tiff"
    return imread(patient_mask_path)

def get_anndata(patient_id): 
    return anndata.read_h5ad(anndata_base_path / f'{patient_id}.h5ad')

import joblib
from pathlib import Path
import numpy as np
import os

def get_all_patient_ids(dataset):
    return [s[0].split("/")[-1].split(".")[0] for s in dataset.samples]

def get_global_cell_type_mapping(dataset, resolution=2, force_recompute=False, cache_path="cell_type_mapping.joblib"):
    """Get a global mapping of cell types to consistent numbers across all patients"""
    if resolution == 0:
        res_key = 'cell_category'
    elif resolution == 1:
        res_key = 'cell_type'
    elif resolution == 2:
        res_key = 'cell_subtype'
    else:
        raise ValueError("Resolution must be 0, 1, or 2")
    
    cache_path = Path(cache_path)
    if not force_recompute and cache_path.exists():
        print(joblib.load(cache_path))
        return joblib.load(cache_path)
    
    # Get all unique cell types across all patients
    all_cell_types = set()
    # Assuming you have a way to get all patient IDs
    for patient_id in get_all_patient_ids(dataset):  # You'll need to implement this
        adata = get_anndata(patient_id)
        all_cell_types.update(adata.obs[res_key].unique())
    
    # Create sorted mapping
    all_cell_types = sorted(all_cell_types)
    cell_type_to_num = {cell_type: i+1 for i, cell_type in enumerate(all_cell_types)}
    num_to_cell_type = {i+1: cell_type for i, cell_type in enumerate(all_cell_types)}
    
    mapping = {
        'cell_type_to_num': cell_type_to_num,
        'num_to_cell_type': num_to_cell_type,
        'all_cell_types': all_cell_types,
        'resolution': resolution
    }
    
    joblib.dump(mapping, cache_path)
    return mapping


def get_cell_type_mask(
    patient_id, 
    resolution=2,
    mask=None,
    adata=None,
    heatmap=False,
    bbox=None,
    global_mapping=None
):
    if resolution == 0:
        res_key = 'cell_category'
    elif resolution == 1:
        res_key = 'cell_type'
    elif resolution == 2:
        res_key = 'cell_subtype'
    else: 
        raise ValueError("Resolution must be 0, 1, or 2")

    if adata is None:
        adata = get_anndata(patient_id)
    
    mask = get_mask(patient_id) if mask is None else mask
    
    mask_cell_nums = list(set(k for k in mask.flatten() if k != 0))
    
    # Safe cell type lookup
    def get_cell_type(cell_num):
        matches = adata.obs[adata.obs['CellNumber'] == cell_num][res_key]
        if len(matches) == 0:
            return "Unknown"
        return matches.iloc[0]  # Take first match if multiple
    
    mask_cell_types = {mask_cell_num: get_cell_type(mask_cell_num) for mask_cell_num in mask_cell_nums}

    # Use global mapping if provided, otherwise create patient-specific mapping
    if global_mapping is None:
        all_cell_types = sorted(adata.obs[res_key].unique())
        cell_type_to_num = {cell_type: i+1 for i, cell_type in enumerate(all_cell_types)}
        num_to_cell_type = {i+1: cell_type for i, cell_type in enumerate(all_cell_types)}
    else:
        cell_type_to_num = global_mapping['cell_type_to_num']
        num_to_cell_type = global_mapping['num_to_cell_type']
        # Add any new cell types not in global mapping
        for cell_type in set(mask_cell_types.values()) - set(cell_type_to_num.keys()):
            new_num = len(cell_type_to_num) + 1
            cell_type_to_num[cell_type] = new_num
            num_to_cell_type[new_num] = cell_type

    def convert_value(mask_value):
        if mask_value == 0:
            return 0
        cell_type = mask_cell_types.get(mask_value, "Unknown")
        return cell_type_to_num.get(cell_type, 0)  # 0 for unknown/background

    mask_converted = np.vectorize(convert_value)(mask)

    return mask_converted, cell_type_to_num, num_to_cell_type

def plot_cell_type(mask, cell_type_to_num, num_to_cell_type, patient_id=None, save_pth=None, bbox=None):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import numpy as np

    # Get all possible cell types (sorted by their number)
    all_nums = sorted(num_to_cell_type.keys())
    cmap = plt.cm.get_cmap('tab20', len(all_nums))
    cmap.set_under('white')  # Values < vmin (i.e. 0) are white

    # Create legend elements in numerical order
    legend_elements = [
        Patch(facecolor=cmap(i), edgecolor='black', label=num_to_cell_type[num])
        for i, num in enumerate(all_nums)
    ]

    mask_sub = mask
    if bbox:
        mask_sub = mask_sub[bbox[0]:bbox[1], bbox[2]:bbox[3]]

    plt.figure(figsize=(10, 10))
    plt.imshow(mask_sub, cmap=cmap, vmin=1, vmax=len(all_nums), interpolation='nearest')
    plt.axis('off')

    # Add the legend
    plt.legend(handles=legend_elements, loc='upper right', title="Cell Types")
    if save_pth is None and patient_id is not None:
        save_pth = f"mask_converted_{patient_id}.png"
    if save_pth:
        plt.savefig(save_pth, bbox_inches='tight', dpi=300)
    plt.show()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
from matplotlib import cm

def get_caf_gradient_colors(n=12):
    """Generate a red to yellow gradient for CAFs using 'autumn' colormap"""
    cmap = cm.get_cmap('autumn', n)
    return [cmap(i)[:3] for i in range(n)]

def create_cell_type_colormap():
    """Create a custom colormap organized by cell type categories"""
    # Define color groups - each list contains colors for related cell types
    color_groups = [
        # T Cells (blues)
        ['#1f77b4', '#4b78ca', '#7aa6e6', '#2c5fbd', '#aec7e8', '#9467bd', '#a05eb5', '#ba9bcf', '#8c54ac', '#c5b0d5'],
        
        # B Cells (pink)
        ["#fe019a"],
        
        # CAFs (reds/oranges)
        get_caf_gradient_colors(11),
        
        # Myeloid/Neutrophils (greens)
        ['#5cb85c', '#3d8b3d'],
        
        # Vascular/Stromal (browns)
        ['#8c564b', '#a05a4e'],
        
        # Other/Unclassified (grays)
        ['#7f7f7f', '#5a5a5a', '#a3a3a3', '#c7c7c7', '#e0e0e0']
    ]
    
    # Start with white for background
    colors = [(1, 1, 1)]
    for group in color_groups:
        colors.extend([mcolors.to_rgb(c) for c in group])
    return ListedColormap(colors)

def get_cell_type_category(cell_type):
    """Categorize cell types for color grouping"""
    if 'CAF' in cell_type:
        return 'CAF'
    elif any(t in cell_type for t in ['CD4', 'CD8', 'Treg']):
        return 'T Cell'
    elif 'Bcell' in cell_type:
        return 'B Cell'
    elif any(t in cell_type for t in ['Myeloid', 'Neutrophil']):
        return 'Myeloid'
    elif any(t in cell_type for t in ['HEV', 'Lymphatic']):
        return 'Vascular'
    else:
        return 'Other'

def plot_cell_type(mask, cell_type_to_num, num_to_cell_type, patient_id=None, save_pth=None, bbox=None):
    # Create colormap
    cmap = create_cell_type_colormap()
    
    # Group cell types by category
    categories = {
        'T Cell': [],
        'B Cell': [],
        'CAF': [],
        'Myeloid': [],
        'Vascular': [],
        'Other': []
    }
    for cell_type in num_to_cell_type.values():
        categories[get_cell_type_category(cell_type)].append(cell_type)
    
    # Sort and flatten cell types for consistent index assignment
    all_cell_types = []
    for cat in ['T Cell', 'B Cell', 'CAF', 'Myeloid', 'Vascular', 'Other']:
        all_cell_types.extend(sorted(set(categories[cat])))

    # Assign contiguous indices
    cell_type_colors = {cell_type: idx + 1 for idx, cell_type in enumerate(all_cell_types)}

    # Create value-to-color mapping
    value_to_color = {0: 0}  # Background
    for num, cell_type in num_to_cell_type.items():
        value_to_color[num] = cell_type_colors[cell_type]

    # Apply mapping to mask
    colored_mask = np.vectorize(lambda x: value_to_color.get(x, 0))(mask)
    if bbox:
        colored_mask = colored_mask[bbox[0]:bbox[1], bbox[2]:bbox[3]]
    
    # Build legend with aligned color indices
    legend_elements = []
    for cell_type in all_cell_types:
        color_idx = cell_type_colors[cell_type]
        legend_elements.append(
            Patch(facecolor=cmap(color_idx), edgecolor='black', label=cell_type)
        )

    # Plot
    plt.figure(figsize=(12, 10))
    img = plt.imshow(colored_mask, cmap=cmap, vmin=0, vmax=len(all_cell_types)+1, interpolation='nearest')
    img.cmap.set_under('white')
    plt.axis('off')

    plt.legend(
        handles=legend_elements,
        loc='center left',
        bbox_to_anchor=(1, 0.5),
        title="Cell Types",
        fontsize='small',
        ncol=2
    )
    plt.tight_layout()

    if save_pth is None and patient_id is not None:
        save_pth = f"cell_type_map_{patient_id}.png"
    if save_pth:
        plt.savefig(save_pth, bbox_inches='tight', dpi=300)
    plt.show()

def find_k_closest_per_prototype(
    dataloader,
    prototype_network,
    k=5,
    preprocess_input_function=None,
    save_cell_masks=False,
    save_dir=None,  # New optional kwarg: if set, save images here
):
    """
    Finds k images with highest activation scores per prototype.
    
    Args:
        dataloader: PyTorch DataLoader.
        prototype_network: Model with `num_prototypes` attribute.
        k: Number of closest images per prototype.
        preprocess_input_function: Optional preprocessing on batch input.
        save_dir: Optional str. Directory to save images by prototype.
        image_loader_function: Required if save_dir is set. Callable: idx -> PIL.Image or np.ndarray.
        
    Returns:
        dict with 'indices' and 'scores' arrays.
    """
    prototype_network.eval()
    n_prototypes = prototype_network.num_prototypes
    
    top_scores = np.full((n_prototypes, k), -np.inf)
    top_indices = np.zeros((n_prototypes, k), dtype=np.int32)
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        for proto_id in range(n_prototypes):
            os.makedirs(os.path.join(save_dir, f"prototype_{proto_id}"), exist_ok=True)
    
    global_mapping = get_global_cell_type_mapping(dataloader.dataset, resolution=2)

    for batch_idx, (batch_input, _) in tqdm(enumerate(dataloader)):
        start_idx = batch_idx * dataloader.batch_size
        
        if preprocess_input_function:
            batch_input = preprocess_input_function(batch_input)
        
        with torch.no_grad():
            batch_input = batch_input.cuda()
            _, prototype_scores = prototype_network(batch_input)
        
        curr_indices = start_idx + np.arange(len(batch_input))
        
        for j in range(n_prototypes):
            per_prototype_scores = prototype_scores[:, j].cpu().numpy()
            combined_scores = np.concatenate([top_scores[j], per_prototype_scores])
            combined_indices = np.concatenate([top_indices[j], curr_indices])
            
            top_k_idx = np.argpartition(-combined_scores, k)[:k]
            top_scores[j] = combined_scores[top_k_idx]
            top_indices[j] = combined_indices[top_k_idx]
    
    # Sort each prototype's results by score descending
    for j in range(n_prototypes):
        order = np.argsort(-top_scores[j])
        top_scores[j] = top_scores[j][order]
        top_indices[j] = top_indices[j][order]

        if save_dir:
            proto_folder = os.path.join(save_dir, f"prototype_{j}")
            for rank, idx in enumerate(top_indices[j]):

                sample_id = dataloader.dataset.samples[idx][0].split("/")[-1].split(".")[0]

                img = dataloader.dataset[idx][0]
                img = cv2.cvtColor(img.numpy().transpose(1, 2, 0), cv2.COLOR_RGB2BGR) * 255.
                img_save_path = os.path.join(proto_folder, f"rank_{rank}_idx_{idx}_sample_{sample_id}.png")
                cv2.imwrite(img_save_path,img)
            
                if save_cell_masks:
                    cell_type_plot, cell_type_to_num, num_to_cell_type = get_cell_type_mask(
                        sample_id, 
                        resolution=2,
                        global_mapping=global_mapping
                    )
                    plot_cell_type(
                        cell_type_plot, 
                        cell_type_to_num, 
                        num_to_cell_type, 
                        save_pth=os.path.join(proto_folder, f"rank_{rank}_idx_{idx}_sample_{sample_id}_cell_type_plot.png")
                    )

                print(f"saved {idx}")

    return {
        'indices': top_indices,
        'scores': top_scores
    }

def get_prototype_specific_cell_type_mask(
    top_image_indices, 
    prototype_masks, 
    dataloader, 
    resolution=2
):
    cell_type_masks = [{} for _ in range(len(prototype_masks))]
    global_mapping = get_global_cell_type_mapping(dataloader.dataset, resolution=resolution)

    for prototype in tqdm(range(len(prototype_masks))):
        prototype_specific_indices = top_image_indices.get('indices', {}).get(prototype, [])
        if not prototype_specific_indices:
            continue  # Skip if no indices for this prototype

        for idx in prototype_specific_indices:
            try:
                img = dataloader.dataset[idx][0]
                sample_id = img.split("/")[-1].split(".")[0]

                cell_type_plot, cell_type_to_num, num_to_cell_type = get_cell_type_mask(
                    sample_id, 
                    resolution=resolution,
                    global_mapping=global_mapping
                )

                mask = prototype_masks[prototype][idx]
                cell_type_masks[prototype][idx] = cell_type_plot * (mask > 0).astype(cell_type_plot.dtype)
            except Exception as e:
                print(f"Error processing index {idx} for prototype {prototype}: {e}")
                continue

    return cell_type_masks, cell_type_to_num, num_to_cell_type

def get_prototype_activation_masks(
    top_image_indices, 
    prototype_network,
    dataloader,
    binarize=True,
    crop=False,
    ):  
    """
    Gets (cell OR pixel) x (binary OR cell type) non-spatial mask for the k patches with highest prototype activation.
    """
    prototype_network.eval()

    masks = [{} for _ in range(prototype_network.num_prototypes)]

    for prototype in tqdm(range(prototype_network.num_prototypes)):
        prototype_specific_indices = top_image_indices['indices'][prototype]
        for idx in prototype_specific_indices:
            img = dataloader.dataset[idx][0]
            _, proto_dist_torch = prototype_network.push_forward(img.cuda().unsqueeze(0))

            # proto_dist_img_j = proto_dist_[img_index_in_batch, j, :, :]
            proto_dist_img_j = proto_dist_torch[0, prototype, ...]

            assert len(proto_dist_img_j.shape) == 2, "shape mismatch (look around)"

            prototype_shape = prototype_network.prototype_shape
            n_prototypes = prototype_shape[0]
            proto_h = prototype_shape[2]
            proto_w = prototype_shape[3]
            max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

            # todo
            if prototype_network.prototype_activation_function == 'log':
                proto_act_img_j = torch.log((proto_dist_img_j + 1) / (proto_dist_img_j + prototype_network.epsilon))
            elif prototype_network.prototype_activation_function == 'linear':
                assert False, "fix this!!"
                proto_act_img_j = max_dist - proto_dist_img_j
            else:
                proto_act_img_j = prototype_activation_function_in_numpy(proto_dist_img_j)

            pth = dataloader.dataset.samples[idx][0]
            sample_id = pth.split("/")[-1].split(".")[0]
            upsampled_act_img_j = cv2.resize(proto_act_img_j.cpu().numpy(), dsize=(get_mask(sample_id).shape[1],get_mask(sample_id).shape[0]),
                                             interpolation=cv2.INTER_CUBIC)

            original_mask = get_mask(sample_id)

            if crop: 
                proto_bound_j = find_high_activation_crop(upsampled_act_img_j)
                mask = get_mask(prototype_bounds)
            else:
                pp95 = np.percentile(upsampled_act_img_j, 95)
                mask = original_mask[upsampled_act_img_j > pp95]
            
            if binarize:
                mask = (mask > 0).astype(np.uint32)

            masks[prototype][idx] = mask
    
    return masks

def get_matrix_from_masks(
    masks,
    top_image_indices,
    prototype_network,
    dataloader,
    aggregate_by_cell_type=False,
    random=False,
    output_dir="prototype_results"
):
    """
    note: masks can be binary or cell type. 
    """

    output_dir = output_dir + f"_aggregate={aggregate_by_cell_type}"
    os.makedirs(output_dir, exist_ok=True)
    
    preprocessing_stats = joblib.load("/work/FAC/FBM/DBC/mrapsoma/prometex/projects/ProtoPNet/preprocessing_stats.joblib")

    # for prototype in range(prototype_network.num_prototypes):
    print(len(top_image_indices['indices']))

    for prototype in range(len(top_image_indices['indices'])):
        prototype_results = {}
        
        for idx in top_image_indices['indices'][prototype]:
            print(prototype, idx)
            prototype_mask = masks[prototype][idx].astype(np.uint32)
            
            pth, _ = dataloader.dataset.samples[idx]
            sample_id = pth.split("/")[-1].split(".")[0]
            
            # Load and preprocess image
            original_img = get_image(sample_id)
            print(f"Memory usage: RSS {psutil.Process(os.getpid()).memory_info().rss/1024**2:.2f} MB")

            img = np.delete(original_img, IRIDIUM_INDICES, axis=0)
            img = np.arcsinh(img)

            for k in range(img.shape[0]):
                lower = preprocessing_stats['clip_lower'][k]
                upper = preprocessing_stats['clip_upper'][k]
                min_val = preprocessing_stats['min'][k]
                max_val = preprocessing_stats['max'][k]
                img[k, ...] = np.clip(img[k, ...], lower, upper)
                # img[k, ...] = (img[k, ...] - min_val) / (max_val - min_val)

            cell_mask = get_mask(sample_id).astype(np.uint32)
            assert cell_mask.max() > 1

            combined_mask = cell_mask * (prototype_mask > 0).astype(np.uint32)

            rows = np.any(combined_mask, axis=1)
            cols = np.any(combined_mask, axis=0)
            if not np.any(rows) or not np.any(cols):
                continue

            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]

            cropped_img = img[:, rmin:rmax+1, cmin:cmax+1]
            cropped_mask = combined_mask[rmin:rmax+1, cmin:cmax+1]

            processed_img = cropped_img * (cropped_mask > 0)[None, ...]

            if aggregate_by_cell_type:
                n_channels = cropped_img.shape[0]
                regions = regionprops(cropped_mask)  # No intensity image yet
                n_cells = len(regions)

                channel_data = np.zeros((n_channels, n_cells))

                for ch in range(n_channels):
                    # Calculate regionprops for this channel
                    ch_image = cropped_img[ch]
                    ch_regions = regionprops(cropped_mask, intensity_image=ch_image)
                    for i, region in enumerate(ch_regions):
                        channel_data[ch, i] = region.mean_intensity
            else:
                n_pixels = np.sum(cropped_mask > 0)
                n_channels = cropped_img.shape[0]
                channel_data = np.zeros((n_pixels, n_channels))
                for ch in range(n_channels):
                    channel_data[:, ch] = processed_img[ch][cropped_mask > 0]

            prototype_results[sample_id] = {
                # 'processed_img': processed_img,
                'channel_data': channel_data,
                # 'original_img': original_img,
                'mask': combined_mask,
                'bbox': (rmin, rmax, cmin, cmax),
                'aggregation_type': 'cell' if aggregate_by_cell_type else 'pixel'
            }

        # Save current prototype result to disk
        save_path = os.path.join(output_dir, f"prototype_{prototype}{'_random' if random else ''}.joblib")
        joblib.dump(prototype_results, save_path)
        print(f"Saved prototype {prototype} results to {save_path}")

        # Cleanup to free memory
        del prototype_results
        gc.collect()

    print("All prototypes processed and saved.")


if __name__ == "__main__":
    # 1. load dataloader
    train_dir = "/work/FAC/FBM/DBC/mrapsoma/prometex/projects/ProtoPNet/datasets/nsclc/train_normed_cropped"
    base = "/work/FAC/FBM/DBC/mrapsoma/prometex/projects/ProtoPNet/saved_models"
    run = "resnet152/electric-deluge-9"
    checkpoint = "60_9push0.8037"
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
    ppnet = torch.load(state_dict_path, map_location='cuda', weights_only=False)  # Add map_location if needed

    # 3. find top k images per prototype
    save_dir = os.path.join(base, run, "prototype_image_centers", checkpoint)
    os.makedirs(save_dir, exist_ok=True)
    top_image_indices = find_k_closest_per_prototype(
        dataloader=train_push_loader,
        prototype_network=ppnet,
        k=15,
        save_dir=save_dir
    )

#     # import joblib
#     # top_image_indices = joblib.load("top_image_indices.joblib")
#     print("top image indices extracted.")

#     # import joblib
#     # joblib.dump(top_image_indices, "top_image_indices.joblib")

#     # 4. get masks for top k images per prototype
    prototype_masks = get_prototype_activation_masks(
        dataloader=train_push_loader,
        top_image_indices=top_image_indices,
        prototype_network=ppnet,
    )

    cell_prototype_masks, cell_type_to_num, num_to_cell_type = get_prototype_specific_cell_type_mask(
        top_image_indices, 
        prototype_masks, 
        dataloader, 
        resolution=2
    )

    joblib.dump([cell_prototype_masks, cell_type_to_num, num_to_cell_type], "cell_prototype_masks.joblib")
    
#     # # masks = joblib.load("masks.joblib")
#     # # print("masks generated")

#     # joblib.dump(masks, "masks.joblib")

#     # # 5. get matrix from masks
#     # results = get_matrix_from_masks(
#     #     masks=masks,
#     #     top_image_indices=top_image_indices,
#     #     prototype_network=ppnet,
#     #     dataloader=train_push_loader,
#     #     aggregate_by_cell_type=True,
#     # )

#     # print("results done")

