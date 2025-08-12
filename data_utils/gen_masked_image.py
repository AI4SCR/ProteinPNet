import argparse
import numpy as np
import os
import tifffile  # Better for multi-dimensional scientific images

from pathlib import Path
from skimage.measure import regionprops

def get_pre_mask_image(img_path, patient_id):
    patient_img_path = img_path / f"{patient_id}.tiff"
    im_pre_mask = tifffile.imread(patient_img_path)
    return im_pre_mask

def get_mask(mask_path, patient_id):
    patient_mask_path = mask_path / f"{patient_id}.tiff"
    return tifffile.imread(patient_mask_path)

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

def get_image(img_path, mask_path, patient_id, should_aggregate):
    img_pre_mask = get_pre_mask_image(img_path, patient_id)
    mask = get_mask(mask_path, patient_id)

    if should_aggregate:
        return get_aggregated_img(img_pre_mask, mask)
    else:
        return get_masked_img(img_pre_mask, mask)

"""
python script that takes three arguments as command line arguments:
    - an input patient id
    - a setting (to aggregate or not on the mask)
    - an output directory

and generates a new tiff file that applies a mask to the image (either aggregating or not depending on the setting) and saves the output.
"""
def main(patient_id, should_aggregate, output_dir):
    """Load patient data, process, and save results."""
    os.makedirs(output_dir, exist_ok=True)

    nsclc_path = Path('/work/FAC/FBM/DBC/mrapsoma/prometex/data/NSCLC/02_processed')

    mask_path = nsclc_path / 'masks/20250430_cell_masks/'
    img_path = nsclc_path / 'images/images_mcd'

    image = get_image(img_path, mask_path, patient_id, should_aggregate)

    if should_aggregate: 
        output_dir = "/work/FAC/FBM/DBC/mrapsoma/prometex/data/NSCLC/02_processed/images/images_mcd_masked_averaged"
    else: 
        output_dir = "/work/FAC/FBM/DBC/mrapsoma/prometex/data/NSCLC/02_processed/images/images_mcd_masked"

    output_path = os.path.join(output_dir, f"{patient_id}.tiff")

    tifffile.imwrite(output_path, image)
    print(f"Saved {'aggregated' if should_aggregate else 'masked'} image to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process medical images with optional region aggregation.")
    parser.add_argument("--patient-id", type=str, required=True, help="Patient identifier (e.g., 'P001').")
    parser.add_argument("--aggregate", action="store_true", help="Aggregate regions by mean intensity.")
    parser.add_argument("--output-dir", type=str, help="Directory to save output images.", default="/work/FAC/FBM/DBC/mrapsoma/prometex/data/NSCLC/02_processed/images/images_mcd_masked")
    
    args = parser.parse_args()
    main(args.patient_id, args.aggregate, args.output_dir)