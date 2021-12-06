#!/bin/bash
modelname=ssd_mobilenet_v2_coco_2018_03_29
workspace_path="/root/workspace"
tf_models_path="/root/models"
training_path="/root/workspace/training"
PYTHON=/usr/bin/python3
pipeline_filename="pipeline.confignew"

python3 $tf_models_path/research/object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path $training_path/$modelname/$pipeline_filename \
    --trained_checkpoint_prefix ./workspace/training/ssd_mobilenet_v2_coco_2018_03_29/model.ckpt-2130 \
    --output_directory ./workspace/export