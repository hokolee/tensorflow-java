from utils import *


def read_and_decode(filename_queue):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  return serialized_example


def parse_example(batch_serialized_example):
  features = tf.parse_example(batch_serialized_example, features={
    'label': tf.FixedLenFeature([1], tf.float32),
    'instance_id': tf.FixedLenFeature([1], tf.int64),
    'feature_id': tf.VarLenFeature(tf.int64),
    'value': tf.VarLenFeature(tf.float32),
    'wide_feature_id': tf.VarLenFeature(tf.int64),
    'wide_value': tf.VarLenFeature(tf.float32),
  })
  return features


def read_batch(filename_queue):
  serialized_example = read_and_decode(filename_queue)
  min_after_dequeue = 10000
  capacity = 50000
  batch_serialized_example = tf.train.shuffle_batch([serialized_example],
    batch_size=FLAGS.batch_size, num_threads=FLAGS.num_threads, capacity=capacity,
    min_after_dequeue=min_after_dequeue,
    allow_smaller_final_batch=True
  )
  return parse_example(batch_serialized_example)


def read_all_batch(filename_queue):
  features = read_batch(filename_queue)
  init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
  data = []
  line_count = 0
  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    try:
      while True:
        cur_features = sess.run(features)
        data.append(cur_features)
        line_count += len(cur_features)
        if len(data) % 1000 == 0:
          logger.info('load data, len: %d' % (len(data)))
    except tf.errors.OutOfRangeError as e:
      coord.request_stop(e)
    finally:
      coord.request_stop()
      coord.join(threads)
  logger.info('load data, len: %d' % len(data))
  return data
