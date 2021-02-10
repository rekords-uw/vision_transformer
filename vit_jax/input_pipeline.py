# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import flax
import jax
import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_datasets as tfds

import sys
if sys.platform != 'darwin':
  # A workaround to avoid crash because tfds may open to many files.
  import resource
  low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
  resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

# Adjust depending on the available RAM.
MAX_IN_MEMORY = 200_000

DATASET_PRESETS = {
    'cifar10': {
        'train': 'train[:98%]',
        'test': 'test',
        'resize': 512,
        'crop': 384,
        'total_steps': 10_000,
    },
    'cifar100': {
        'train': 'train[:98%]',
        'test': 'test',
        'resize': 512,
        'crop': 384,
        'total_steps': 10_000,
    },
    'imagenet2012': {
        'train': 'train[:99%]',
        'test': 'validation',
        'resize': 512,
        'crop': 384,
        'total_steps': 20_000,
    },
    'svhn_cropped': {
        'train': 'train[:98%]',
        'test': 'test',
        'resize': 512,
        'crop': 384,
        'total_steps': 10_000,
    },
}


def get_dataset_info(dataset, split):
  data_builder = tfds.builder(dataset)
  num_examples = data_builder.info.splits[split].num_examples
  num_classes = data_builder.info.features['label'].num_classes
  return {
    'num_examples': num_examples,
    'num_classes': num_classes
  }

def get_filenames(is_training, data_dir):
  if is_training:
    return [os.path.join(data_dir, 'train-%05d-of-01024' % i) for i in range(NUM_TRAIN_FILES)]
  else:
    return [os.path.join(data_dir, 'val-%05d-of-00128' % i) for i in range(NUM_TEST_FILES)]

def parse_example_proto(example_serialized):
  """
  Parses an Example proto containing an image.

  :param example_serialized: scalar Tensor tf.string
  :return:
      image_buffer: Tensor tf.string with contents of a JPEG file
      label: Tensor tf.float32 containing the label
  """
  feature_map = {'image/encoded': tf.io.FixedLenFeature([], dtype=tf.string, default_value=''),
                 'image/class/label': tf.io.FixedLenFeature([], dtype=tf.int64, default_value=-1),}

  features = tf.io.parse_single_example(serialized=example_serialized, features=feature_map)

  # want labels [0, 999] instead of [1, 1000]
  one_hot = True
  if one_hot:
    # one hot encoding without label smoothing
    smoothing = 0.1
    label = tf.one_hot(features['image/class/label'] - 1, 1000, 
                       #on_value=1-smoothing, off_value=smoothing/(1000-1),
                       dtype=tf.float32)
  else:
    label = tf.cast(features['image/class/label'] - 1, dtype=tf.float32)

  return features['image/encoded'], label

def preprocess_image(image_buffer, is_training, inception_crop, resize_size, crop_size):
  NUM_CHANNELS = 3
  image = tf.io.decode_jpeg(image_buffer, channels=NUM_CHANNELS)
  im = tf.cast(image, tf.float32)

  if is_training:
    if inception_crop:
      channels = NUM_CHANNELS
      begin, size, _ = tf.image.sample_distorted_bounding_box(
          tf.shape(im),
          tf.zeros([0, 0, 4], tf.float32),
          area_range=(0.05, 1.0),
          min_object_covered=0,  # Don't enforce a minimum area.
          use_image_if_no_bounding_boxes=True)
      im = tf.slice(im, begin, size)
      # Unfortunately, the above operation loses the depth-dimension. So we
      # need to restore it the manual way.
      im.set_shape([None, None, channels])
      im = tf.image.resize(im, [crop_size, crop_size])
    else:
      im = tf.image.resize(im, [resize_size, resize_size])
      im = tf.image.random_crop(im, [crop_size, crop_size, 3])
      im = tf.image.flip_left_right(im)
  else:
    # usage of crop_size here is intentional
    im = tf.image.resize(im, [crop_size, crop_size])
  im = (im - 127.5) / 127.5
  return im

def parse_record(raw_record, is_training, inception_crop, resize_size, crop_size):
  """
  Parse a record containing an image. Record is parsed into a label and image, and the image is passed through
  preprocessing steps.

  :param raw_record:
  :param is_training:
  :param dtype:
  :param center_crops_for_train
  :param single_scale
  :param color_aug
  :param standardize_train
  :param resnet_preprocessing
  :return:
  """
  image_buffer, label = parse_example_proto(raw_record)

  image = preprocess_image(image_buffer, is_training=is_training, inception_crop=inception_crop, 
                           resize_size=resize_size, crop_size=crop_size)

  return {'image': image, 'label': label}

def get_data_imagenet(base_data_dir,
                      mode,
                      repeats,
                      batch_size,
                      shuffle_buffer=MAX_IN_MEMORY,
                      inception_crop=True):
  preset = DATASET_PRESETS.get('imagenet2012')
  resize_size = preset['resize']
  crop_size = preset['crop']

  data_dir = 'train' if mode == 'train' else 'val'
  filenames = get_filenames(mode == 'train', os.path.join(base_data_dir, data_dir))
  dataset = tf.data.Dataset.from_tensor_slices(filenames)

  NUM_TRAIN_FILES = 1024
  if mode == 'train':
    # Shuffle the input files
    dataset = dataset.shuffle(buffer_size=NUM_TRAIN_FILES)
    dataset = dataset.interleave(lambda x: tf.data.TFRecordDataset(x, buffer_size=None, num_parallel_reads=None),
                                 num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # Prefetch a batch at a time to smooth out the time taken for processing.
    dataset = dataset.prefetch(buffer_size=batch_size)

  if is_training:
    # Shuffles records before repeating to respect epoch boundaries. Show every element of one epoch before next.
    dataset = dataset.shuffle(buffer_size=shuffle_buffer, reshuffle_each_iteration=True)

  # Repeats the dataset for the number of epochs to train.
  dataset = dataset.repeat(repeats)
  # Parses the raw records into images and labels.
  dataset = dataset.map(lambda value: parse_record(value, mode=='train', inception_crop=inception_crop, 
                                                   resize_size=resize_size, crop_size=crop_size),
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
  
  dataset = dataset.batch(batch_size, drop_remainder=True)

  num_devices = jax.local_device_count()

  def _shard(data):
    data['image'] = tf.reshape(data['image'],
                               [num_devices, -1, crop_size, crop_size, 3])
    data['label'] = tf.reshape(data['label'],
                               [num_devices, -1, 1000])
    return data

  if num_devices is not None:
    dataset = dataset.map(_shard, tf.data.experimental.AUTOTUNE)

  return dataset.prefetch(1)




def get_data(*,
             dataset,
             mode,
             repeats,
             batch_size,
             mixup_alpha=0,
             shuffle_buffer=MAX_IN_MEMORY,
             tfds_data_dir=None,
             tfds_manual_dir=None,
             inception_crop=True):
  """Returns dataset for training/eval.

  Args:
    dataset: Dataset name. Additionally to the requirement that this dataset
      must be in tensorflow_datasets, the dataset must be registered in
      `DATASET_PRESETS` (specifying crop size etc).
    mode: Must be "train" or "test".
    repeats: How many times the dataset should be repeated. For indefinite
      repeats specify None.
    batch_size: Global batch size. Note that the returned dataset will have
      dimensions [local_devices, batch_size / local_devices, ...].
    mixup_alpha: Coefficient for mixup combination. See 
      https://arxiv.org/abs/1710.09412
    shuffle_buffer: Number of elements to preload the shuffle buffer with.
    tfds_data_dir: Optional directory where tfds datasets are stored. If not
      specified, datasets are downloaded and in the default tfds data_dir on the
      local machine.
    inception_crop: If set to True, tf.image.sample_distorted_bounding_box()
      will be used. If set to False, tf.image.random_crop() will be used.
  """

  preset = DATASET_PRESETS.get(dataset)
  if preset is None:
    raise KeyError(f'Please add "{dataset}" to {__name__}.DATASET_PRESETS"')
  split = preset[mode]
  resize_size = preset['resize']
  crop_size = preset['crop']
  data_builder = tfds.builder(dataset, data_dir=tfds_data_dir)
  dataset_info = get_dataset_info(dataset, split)

  data_builder.download_and_prepare(
      download_config=tfds.download.DownloadConfig(manual_dir=tfds_manual_dir))
  data = data_builder.as_dataset(
      split=split, decoders={'image': tfds.decode.SkipDecoding()})
  decoder = data_builder.info.features['image'].decode_example

  def _pp(data):
    im = decoder(data['image'])
    if mode == 'train':
      if inception_crop:
        channels = im.shape[-1]
        begin, size, _ = tf.image.sample_distorted_bounding_box(
            tf.shape(im),
            tf.zeros([0, 0, 4], tf.float32),
            area_range=(0.05, 1.0),
            min_object_covered=0,  # Don't enforce a minimum area.
            use_image_if_no_bounding_boxes=True)
        im = tf.slice(im, begin, size)
        # Unfortunately, the above operation loses the depth-dimension. So we
        # need to restore it the manual way.
        im.set_shape([None, None, channels])
        im = tf.image.resize(im, [crop_size, crop_size])
      else:
        im = tf.image.resize(im, [resize_size, resize_size])
        im = tf.image.random_crop(im, [crop_size, crop_size, 3])
        im = tf.image.flip_left_right(im)
    else:
      # usage of crop_size here is intentional
      im = tf.image.resize(im, [crop_size, crop_size])
    im = (im - 127.5) / 127.5
    label = tf.one_hot(data['label'], dataset_info['num_classes'])  # pylint: disable=no-value-for-parameter
    return {'image': im, 'label': label}

  data = data.repeat(repeats)
  if mode == 'train':
    data = data.shuffle(min(dataset_info['num_examples'], shuffle_buffer))
  data = data.map(_pp, tf.data.experimental.AUTOTUNE)
  data = data.batch(batch_size, drop_remainder=True)

  def _mixup(data):
    beta_dist = tfp.distributions.Beta(mixup_alpha, mixup_alpha)
    beta = tf.cast(beta_dist.sample([]), tf.float32)
    data['image'] = (
        beta * data['image'] + (1 - beta) * tf.reverse(data['image'], axis=[0]))
    data['label'] = (
        beta * data['label'] + (1 - beta) * tf.reverse(data['label'], axis=[0]))
    return data

  if mixup_alpha is not None and mixup_alpha > 0.0 and mode == 'train':
    data = data.map(_mixup, tf.data.experimental.AUTOTUNE)

  # Shard data such that it can be distributed accross devices
  num_devices = jax.local_device_count()

  def _shard(data):
    data['image'] = tf.reshape(data['image'],
                               [num_devices, -1, crop_size, crop_size, 3])
    data['label'] = tf.reshape(data['label'],
                               [num_devices, -1, dataset_info['num_classes']])
    return data

  if num_devices is not None:
    data = data.map(_shard, tf.data.experimental.AUTOTUNE)

  return data.prefetch(1)


def prefetch(dataset, n_prefetch):
  """Prefetches data to device and converts to numpy array."""
  ds_iter = iter(dataset)
  ds_iter = map(lambda x: jax.tree_map(lambda t: np.asarray(memoryview(t)), x),
                ds_iter)
  if n_prefetch:
    ds_iter = flax.jax_utils.prefetch_to_device(ds_iter, n_prefetch)
  return ds_iter
