tfzippath=http://download.tensorflow.org/models/object_detection/$MODEL_NAME.tar.gz
wget $tfzippath
tar -xf $MODEL_NAME.tar.gz
rm $MODEL_NAME.tar.gz
mv $MODEL_NAME workspace/pre-trained-models/
# object detection scripts complain about batch_norm_trainable
sed -i 's/batch_norm_trainable: true//' workspace/pre-trained-models/$MODEL_NAME/pipeline.config
      
