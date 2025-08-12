#!/bin/bash
# Loop through all patient files in the directory
start_path='/work/FAC/FBM/DBC/mrapsoma/prometex/data/NSCLC/02_processed/images/images_mcd'

# count=0
# max_files=5
# Iterate through each file in the directory

for file in "$start_path"/*; do
    # if [ $count -ge $max_files ]; then
    #     break
    # fi

    # Extract just the filename without path or extension
    patient=$(basename "$file" | cut -f 1 -d '.')
    
    # Submit jobs for this patient
    sbatch --wrap="dcsrsoft use 20241118; python gen_masked_image.py --patient-id $patient --aggregate" --time=12:00:00 --mem=16G
    sbatch --wrap="dcsrsoft use 20241118; python gen_masked_image.py --patient-id $patient" --time=12:00:00 --mem=16G

    # Increment counter
    # ((count++))
done