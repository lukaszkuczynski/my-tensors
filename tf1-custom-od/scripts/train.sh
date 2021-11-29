#!/bin/bash
modelname=ssd_mobilenet_v2_coco_2018_03_29
workspace_path="/root/workspace"
tf_models_path="/root/models"
training_path="/root/workspace/training"
PYTHON=/usr/bin/python3
pipeline_filename="pipeline.confignew"

rm -rf $training_path/$modelname
mkdir -p $training_path/$modelname
cp $workspace_path/pre-trained-models/$modelname/$pipeline_filename $training_path/$modelname
cp $workspace_path/pre-trained-models/$modelname/model* $training_path/$modelname
ls $training_path/$modelname

python3 \
    $tf_models_path/research/object_detection/model_main.py \
    --alsologtostderr \
    --model_dir=$training_path/$modelname \
    --pipeline_config_path=$training_path/$modelname/$pipeline_filename