#!/bin/bash

TARGET_GPUS="0,1,2,3,4,5,6,7"
export CUDA_VISIBLE_DEVICES=$TARGET_GPUS
GPUS=$(echo $TARGET_GPUS | awk -F, '{print NF}')

VIDEOS_PATH="/path/to/your/video/dir"

FOLDER_NAME=$(basename "$VIDEOS_PATH")
OUTPUT_ROOT="./Results/${FOLDER_NAME}"

dimensions=("subject_consistency" "temporal_flickering" "background_consistency" "aesthetic_quality" "imaging_quality" "object_class" "multiple_objects" "color" "spatial_relationship" "scene" "temporal_style" "overall_consistency" "human_action" "motion_smoothness" "dynamic_degree" "appearance_style")

for dimension in "${dimensions[@]}"; do
    echo "Processing Dimension: $dimension | Path: $VIDEOS_PATH"
    
    torchrun --nproc_per_node=${GPUS} --standalone evaluate.py \
        --videos_path "$VIDEOS_PATH" \
        --dimension "$dimension" \
        --output_path "$OUTPUT_ROOT/"
done

RESULT_DIR=$(realpath "$OUTPUT_ROOT")
PARENT_DIR=$(dirname "$RESULT_DIR")

echo "Zipping results in $RESULT_DIR..."
cd "$RESULT_DIR" || exit
python -m zipfile -c "${PARENT_DIR}/${FOLDER_NAME}.zip" .

cd "$PARENT_DIR" || exit
echo "Calculating final score for ${FOLDER_NAME}.zip..."
python ../scripts/cal_final_score.py --zip_file "${PARENT_DIR}/${FOLDER_NAME}.zip"