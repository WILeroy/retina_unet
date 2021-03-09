import os
import time

from tensorflow.compat.v1 import ConfigProto, InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import tensorflow as tf
from absl import app, flags

from core.dataset import CreateDataset
from core.unet import Unet

FLAGS = flags.FLAGS

flags.DEFINE_string('logdir', None, 'WithTensorBoard logdir.')
flags.DEFINE_string('label_file_path', None,
                    'Path of training dataset label file.')
flags.DEFINE_integer('seed', 0, 'Seed to training dataset.')
flags.DEFINE_float('initial_lr', 3e-4, 'Initial learning rate.')
flags.DEFINE_integer('batch_size', 2, 'Global batch size.')
flags.DEFINE_integer('max_iters', 1000, 'Maximum iterations.')
flags.DEFINE_integer('save_interval', 100, 'Interval to save model weights.')
flags.DEFINE_boolean('preprocess', False, 'Whether to use preprocessed image.')
flags.DEFINE_boolean('augmentation', False, 'Whether to use augmentation.')
flags.DEFINE_integer('image_size', 1024, 'Size of each image side to use.')
flags.DEFINE_integer('num_classes', 2, 'Number of classes.')
flags.DEFINE_boolean('transpose_conv', False,
                     'Whether to use Conv2DTranspose as upsample layer.')


def _learning_rate_schedule(global_step_value, max_iters, initial_lr):
  """Calculates learning_rate with linear decay.

  Args:
    global_step_value: int, global step.
    max_iters: int, maximum iterations.
    initial_lr: float, initial learning rate.

  Returns:
    lr: float, learning rate.
  """
  lr = initial_lr * (1.0 - global_step_value / max_iters)
  return lr


def main(argv):

  max_iters = FLAGS.max_iters
  initial_lr = FLAGS.initial_lr
  print(FLAGS.preprocess)
  # -------------------------------------------------
  # Create the training set.
  train_dataset = CreateDataset(
      file_path=FLAGS.label_file_path,
      image_size=FLAGS.image_size,
      batch_size=FLAGS.batch_size,
      preprocess=FLAGS.preprocess,
      augmentation=FLAGS.augmentation,
      seed=FLAGS.seed)
  train_iter = iter(train_dataset)

  model = Unet(
      data_format='channels_last',
      classes=FLAGS.num_classes,
      transpose_conv=FLAGS.transpose_conv)
  
  loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)

  # Setup summary writer.
  summary_writer = tf.summary.create_file_writer(
    os.path.join(FLAGS.logdir, 'train_logs'), flush_millis=10000)

  # Create a checkpoint directory to store the checkpoints.
  checkpoint_prefix = os.path.join(FLAGS.logdir, 'unet_tf2-ckpt')

  # Setup checkpoint directory.
  checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
  manager = tf.train.CheckpointManager(
    checkpoint,
    checkpoint_prefix,
    max_to_keep=10,
    keep_checkpoint_every_n_hours=3)
  # Restores the checkpoint, if existing.
  checkpoint.restore(manager.latest_checkpoint)

  def train_step(inputs):
    """Train one batch."""
    images, labels, masks = inputs

    with tf.GradientTape() as tape:
      logits = model.build_call(images, training=True)
      loss = loss_func(labels, logits)
    
    gradients = tape.gradient(loss, model.trainable_weights)
    clipped, _ = tf.clip_by_global_norm(gradients, clip_norm=tf.constant(10.0))
    optimizer.apply_gradients(zip(clipped, model.trainable_weights))

    return loss

  global_step_value = optimizer.iterations.numpy()
  last_summary_step_value = None
  last_summary_time = None
  while global_step_value < max_iters:
    input_batch = next(train_iter)

    # Set learning rate and run the training step over num_gpu gpus.
    optimizer.learning_rate = _learning_rate_schedule(
      optimizer.iterations.numpy(), max_iters, initial_lr)
    loss = train_step(input_batch)

    # Step number, to be used for summary/logging.
    global_step = optimizer.iterations
    global_step_value = global_step.numpy()

    # LR, losses and others summaries.
    with summary_writer.as_default():
      tf.summary.scalar(
        'learning_rate', optimizer.learning_rate, step=global_step)
      tf.summary.scalar(
        'loss', loss, step=global_step)
    
      # Summary for number of global steps taken per second.
      current_time = time.time()
      if (last_summary_step_value is not None and
          last_summary_time is not None):
        tf.summary.scalar(
            'global_steps_per_sec',
            (global_step_value - last_summary_step_value) /
            (current_time - last_summary_time),
            step=global_step)
      last_summary_step_value = global_step_value
      last_summary_time = current_time

    if (global_step_value % FLAGS.save_interval == 0) or (
        global_step_value >= max_iters):
      save_path = manager.save(checkpoint_number=global_step_value)
      
      weights_path = os.path.join(FLAGS.logdir, 'unet_weights')
      model.save_weights(weights_path, save_format='tf')


if __name__ == '__main__':
  app.run(main)
