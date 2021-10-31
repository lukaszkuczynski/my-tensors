import glob
import random
import math
import os
import tensorflow.compat.v1 as compat_tf
from object_detection.utils import label_map_util
from object_detection.utils import dataset_util
from lxml import etree
import PIL.Image
import io
import hashlib
from google.protobuf import text_format
from object_detection.protos import pipeline_pb2

PROJECT_NAME=os.getenv("IMAGE_PROJECT_NAME")
model_name = os.getenv("MODEL_NAME")

WORKSPACE_DIR = "/root/workspace/"

IMAGES_AND_LABELS_FOLDER = os.path.join(WORKSPACE_DIR, 'images')
IMAGES_AND_LABELS_PROJECT_FOLDER = os.path.join(IMAGES_AND_LABELS_FOLDER, PROJECT_NAME)
LABEL_MAP_PATH = os.path.join(WORKSPACE_DIR, "labels.pbtxt")
PRETRAINED_MODELS_FOLDER = os.path.join(WORKSPACE_DIR, "pre-trained-models")

def train_valid_test_split(path, split_ratio=(0.8,0.1,0.1)):
  all_images = list(os.path.basename(filename) for filename in glob.glob(os.path.join(path, "*.JPEG")))
  if len(split_ratio) != 3:
    raise AttributeError("you should provide a tuple with 3 fractions for split- train,valid,test")
  if sum(split_ratio) != 1:
    raise AttributeError("Split should add up to 1.0")
  train_len = math.floor(split_ratio[0] * len(all_images))
  random.seed(10)
  train_images = random.sample(all_images, train_len)
  other_images = list(set(all_images) - set(train_images))
  valid_len = math.floor(split_ratio[1] * len(all_images))
  valid_images = random.sample(other_images, valid_len)
  test_images = list(set(other_images) - set(valid_images))
  return train_images, valid_images, test_images

train_images, valid_images, test_images = train_valid_test_split(IMAGES_AND_LABELS_PROJECT_FOLDER)

print("train has %d elements, valid %d, test %d" % (len(train_images), len(valid_images), len(test_images)))

annotations_dir = IMAGES_AND_LABELS_PROJECT_FOLDER
data_dir = IMAGES_AND_LABELS_FOLDER
tf_record_folder = os.path.join(WORKSPACE_DIR, "tfrecord")


def dict_to_tf_example(data,
                       dataset_directory,
                       label_map_dict,
                       ignore_difficult_instances=False):
  """Convert XML derived dict to tf.Example proto.
  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.
  Args:
    data: dict holding PASCAL XML fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict)
    dataset_directory: Path to root directory holding PASCAL dataset
    label_map_dict: A map from string label names to integers ids.
    ignore_difficult_instances: Whether to skip difficult instances in the
      dataset  (default: False).
    image_subdirectory: String specifying subdirectory within the
      PASCAL dataset directory holding the actual image data.
  Returns:
    example: The converted tf.Example.
  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  img_path = os.path.join(data['folder'], data['filename'])
  full_path = os.path.join(dataset_directory, img_path)
  with compat_tf.gfile.GFile(full_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')
  key = hashlib.sha256(encoded_jpg).hexdigest()

  width = int(data['size']['width'])
  height = int(data['size']['height'])

  xmin = []
  ymin = []
  xmax = []
  ymax = []
  classes = []
  classes_text = []
  truncated = []
  poses = []
  difficult_obj = []
  if 'object' in data:
    for obj in data['object']:
      difficult = bool(int(obj['difficult']))
      if ignore_difficult_instances and difficult:
        continue

      difficult_obj.append(int(difficult))

      xmin.append(float(obj['bndbox']['xmin']) / width)
      ymin.append(float(obj['bndbox']['ymin']) / height)
      xmax.append(float(obj['bndbox']['xmax']) / width)
      ymax.append(float(obj['bndbox']['ymax']) / height)
      classes_text.append(obj['name'].encode('utf8'))
      classes.append(label_map_dict[obj['name']])
      truncated.append(int(obj['truncated']))
      poses.append(obj['pose'].encode('utf8'))

  example = compat_tf.train.Example(features=compat_tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
      'image/object/truncated': dataset_util.int64_list_feature(truncated),
      'image/object/view': dataset_util.bytes_list_feature(poses),
  }))
  return example



def files_to_tfrecord(filenames, data_dir, output_path, ignore_difficult_instances=False):
  filenames_no_extensions = [os.path.splitext(fn)[0] for fn in filenames]
  label_map_dict = label_map_util.get_label_map_dict(LABEL_MAP_PATH)
  examples = []
  with compat_tf.python_io.TFRecordWriter(output_path) as writer:
    for idx, example in enumerate(filenames_no_extensions):
      path = os.path.join(annotations_dir, example + '.xml')
      with compat_tf.gfile.GFile(path, 'r') as fid:
        xml_str = fid.read()    
      xml = etree.fromstring(xml_str)
      data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
      tf_example = dict_to_tf_example(data, data_dir, label_map_dict,
                                      ignore_difficult_instances)
      examples.append(tf_example)
      writer.write(tf_example.SerializeToString())





original_pipeline_path = os.path.join(PRETRAINED_MODELS_FOLDER, model_name, "pipeline.config")
pipeline_path = os.path.join(PRETRAINED_MODELS_FOLDER, model_name, "pipeline.confignew")
checkpoint_path = "checkpoint/ckpt-0"

local_data_folder = "./source_dir"

label_map_filename = "avocado_labels.pbtxt"


def edit_pipeline_config(old_path, new_path, cfg):
  pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()                                                                                                                                                                                                          

  with compat_tf.gfile.GFile(old_path, "r") as f:                                                                                                                                                                                                                     
      proto_str = f.read()                                                                                                                                                                                                                                          
      text_format.Merge(proto_str, pipeline_config)                                                                                                                                                                                                                 

  pipeline_config.model.ssd.num_classes = cfg['num_classes']
  pipeline_config.train_config.batch_size = cfg['batch_size']
  pipeline_config.train_config.fine_tune_checkpoint = cfg['fine_tune_checkpoint']
  pipeline_config.train_config.fine_tune_checkpoint_type = 'detection'
  pipeline_config.train_input_reader.label_map_path = cfg['label_map_path']
  pipeline_config.train_input_reader.tf_record_input_reader.input_path[0] = cfg['train_tf_path']
  pipeline_config.eval_input_reader[0].label_map_path = cfg['label_map_path']
  pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[0] = cfg['eval_tf_path']

  config_text = text_format.MessageToString(pipeline_config)                                                                                                                                                                                                        
  with compat_tf.gfile.Open(new_path, "wb") as f:                                                                                                                                                                                                                       
      f.write(config_text)                                                                                                                                                                                                                                          

        
        

data_folder = local_data_folder
        
new_config = {
    "batch_size": 8,
    "num_classes": 1,
    "label_map_path": os.path.join(data_folder, label_map_filename),
    "train_tf_path": os.path.join(data_folder, 'train'),
    "eval_tf_path": os.path.join(data_folder, 'valid'),
    "fine_tune_checkpoint": checkpoint_path
}





files_to_tfrecord(train_images, data_dir, os.path.join(tf_record_folder, 'train'))
files_to_tfrecord(valid_images, data_dir, os.path.join(tf_record_folder, 'valid'))
files_to_tfrecord(test_images, data_dir, os.path.join(tf_record_folder, 'test'))


edit_pipeline_config(original_pipeline_path, pipeline_path, new_config)
