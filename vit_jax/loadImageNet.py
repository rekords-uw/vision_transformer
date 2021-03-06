import functools
import glob
import os
import time

from clu import metric_writers

import numpy as np

import jax
import jax.numpy as jnp

import flax
import flax.optim as optim
import flax.jax_utils as flax_utils

import tensorflow as tf

from vit_jax import checkpoint
from vit_jax import flags
from vit_jax import hyper
from vit_jax import logging
from vit_jax import input_pipeline
from vit_jax import models
from vit_jax import momentum_clip

def extract_keys(dict_in, depth=0):
  for key, value in dict_in.items():
    for i in range(depth):
      print(' ', end='')
    print('-', end='')
    print(key)
    if isinstance(value, dict): # If value itself is dictionary
      extract_keys(value, depth+1)
    else:
      for i in range(depth):
        print(' ', end='')
      print(value.shape)
            
def main(args):
  attn_record_layer = 11
  logdir = os.path.join(args.logdir, args.name)
  logger = logging.setup_logger(logdir)
  logger.info(args)

  logger.info(f'Available devices: {jax.devices()}')

  # Setup input pipeline

  print('Before All')
  ds_train = input_pipeline.get_data_imagenet(
      base_data_dir='/home/ubuntu/mountV/imagenet_tf_record_data',
      mode='test',
      repeats=1,
      batch_size=args.batch)

  print('G')
  
  for batch_temp in input_pipeline.prefetch(ds_train, args.prefetch):
    batch = batch_temp
    break

  #logger.info(ds_train)
  print('Before Loading')
  ds_test = input_pipeline.get_data_imagenet(
      base_data_dir='/home/ubuntu/mountV/imagenet_tf_record_data',
      mode='test',
      repeats=1,
      batch_size=args.batch_eval)
  #logger.info(ds_test)
  print('After Loading')

  #for batch in input_pipeline.prefetch(ds_test, args.prefetch):
    #print(batch['image'].shape)
    #print(type(batch['image'].dtype.name))

  # Build VisionTransformer architecture
  model = models.KNOWN_MODELS[args.model]
  VisionTransformer = model.partial(num_classes=1000, attn_record_layer=attn_record_layer)
  _, params = VisionTransformer.init_by_shape(
      jax.random.PRNGKey(0),
      # Discard the "num_local_devices" dimension for initialization.
      [(batch['image'].shape[1:], batch['image'].dtype.name)])

  pretrained_path = os.path.join(args.vit_pretrained_dir, f'{args.model}.npz')
  params = checkpoint.load_pretrained(
      pretrained_path=pretrained_path,
      init_params=params,
      model_config=models.CONFIGS[args.model],
      logger=logger)
  # params = checkpoint.load(pretrained_path)
  # params['pre_logits'] = {}
  #extract_keys(params)
  vit_fn_repl = jax.pmap(VisionTransformer.call)

  opt = momentum_clip.Optimizer(
      dtype=args.optim_dtype, grad_norm_clip=args.grad_norm_clip).create(params)
  opt_repl = flax_utils.replicate(opt)

  del opt
  del params
  print('***********************')

  acc = []

  first_batch = True
  get_acc = False
  for batch in input_pipeline.prefetch(ds_test, args.prefetch):
    output, attention_matrix = vit_fn_repl(opt_repl.target, batch['image'])
    if get_acc:
      batch_acc = np.mean(
        (np.argmax(output, axis=2) == np.argmax(batch['label'], axis=2)).ravel()
        )
      acc.append(batch_acc)
    if first_batch:
      print(batch['image'].shape)
      print(output.shape)
      print(attention_matrix.shape)
      jnp.save('res/batch_image_%s.npy' % args.dataset, batch['image'][:,:10,:])
      jnp.save('res/batch_attn_layer%d_%s.npy' % (attn_record_layer, args.dataset), attention_matrix[:,:10,:])
      jnp.save('res/batch_label_%s.npy' % args.dataset, batch['label'][:,:10,:])
      first_batch = False
      break

  if get_acc:
    acc_test = np.mean(acc)
    logger.info(f'Test accuracy: {acc_test:0.5f}')
    print(acc_test)

if __name__ == '__main__':
  # Make sure tf does not allocate gpu memory.
  tf.config.experimental.set_visible_devices([], 'GPU')

  parser = flags.argparser(models.KNOWN_MODELS.keys(),
                           input_pipeline.DATASET_PRESETS.keys())

  main(parser.parse_args())

