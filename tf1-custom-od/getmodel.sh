echo $MODEL_NAME
exit 1
tfzippath=http://download.tensorflow.org/models/object_detection/$MODEL_NAME.tar.gz
wget $tfzippath
tar -xf $MODEL_NAME.tar.gz
rm $MODEL_NAME.tar.gz
mv $MODEL_NAME workspace/pre-trained-models/