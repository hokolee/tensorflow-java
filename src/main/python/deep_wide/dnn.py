import time

from inputs import *
from network import *
from validation_test import *
from model import *

def train():
  with tf.device('/cpu:0'):
    train_fq = tf.train.string_input_producer(tf.train.match_filenames_once(train_dir + '/part*'))
    train_batch = read_batch(train_fq)
    validation_fq = tf.train.string_input_producer(tf.train.match_filenames_once(validation_dir + '/part*'), num_epochs=1)
    validation_batch = read_all_batch(validation_fq)
    test_fq = tf.train.string_input_producer(tf.train.match_filenames_once(test_dir + '/part*'), num_epochs=1)
    test_batch = read_all_batch(test_fq)

  feature_id = tf.sparse_tensor_to_dense(train_batch['feature_id'], name='feature_id')
  value = tf.sparse_tensor_to_dense(train_batch['value'], name='value')
  wide_feature_id = tf.sparse_tensor_to_dense(train_batch['wide_feature_id'], name='wide_feature_id')
  wide_value = tf.sparse_tensor_to_dense(train_batch['wide_value'], name='wide_value')

#  dnn_inputs = tf.sparse_to_dense(train_batch['feature_id'], [None,num_features], train_batch['value'])
#  wide_inputs = tf.sparse_to_dense(train_batch['wide_feature_id'], [None,num_wide_features], train_batch['wide_value'])

  dnn_sparse_inputs = tf.sparse_merge(train_batch['feature_id'], train_batch['value'], num_features)
  dnn_inputs = tf.sparse_tensor_to_dense(dnn_sparse_inputs, name='dnn_inputs', validate_indices=False)
  wide_sparse_inputs = tf.sparse_merge(train_batch['wide_feature_id'], train_batch['wide_value'], num_wide_features)
  wide_inputs = tf.sparse_tensor_to_dense(wide_sparse_inputs, name='wide_inputs', validate_indices=False)

#  train_prediction = inference(feature_id, value, wide_feature_id, wide_value, layers)
  train_prediction = inference(dnn_inputs, wide_inputs, layers)

  train_mean_loss = log_loss(train_batch['label'], train_prediction)
  tf.summary.scalar('train_mean_loss', train_mean_loss)

  global_step = tf.Variable(1, name='global_step', trainable=False)

  lr, dp = get_config()

  learning_rate = tf.placeholder(tf.float32, shape=[])
  dropout = tf.placeholder(tf.float32, shape=[])

  # optimizer = tf.contrib.opt.LazyAdamOptimizer(learning_rate=learning_rate)
  # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

  train_op = optimizer.minimize(train_mean_loss, global_step=global_step)

  local_var = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES)
  global_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
  trainable_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

  logger.info("local_var:")
  for var in local_var:
    logger.info(var)
  logger.info("global_var:")
  for var in global_var:
    logger.info(var)
  logger.info("trainable_var:")
  for var in trainable_var:
    logger.info(var)

  init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
  summary_op = tf.summary.merge_all()

  writer = tf.summary.FileWriter(FLAGS.log_dir, tf.get_default_graph())
  saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, max_to_keep=1000000)

  graph_options = tf.GraphOptions(enable_bfloat16_sendrecv=False)
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95, allow_growth=True)
  config = tf.ConfigProto(graph_options=graph_options, gpu_options=gpu_options, log_device_placement=True, allow_soft_placement=True)

  step = 1

  with tf.Session(config=config) as sess:
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    # saver.restore(sess, FLAGS.log_dir + '/dnn-100000')
    # logger.info('model restore done')

    total_train = 0
    cur_train = 0
    validation_mean_loss = 0

    while not coord.should_stop():
      if step % FLAGS.validation_per_batch == 0 and step >= FLAGS.silent_before_batch:
        start = time.time()
        validation_mean_loss = calc_validation(sess, validation_batch, step)
        tf.summary.scalar('validation_mean_loss', validation_mean_loss)
        end = time.time()
        total_validation = end - start
        logger.info('step: %09d, train_time: %f, validation_time: %f, validation_mean_loss: %f',
                    step, total_train, total_validation, validation_mean_loss)
        total_train = 0

      if step % FLAGS.test_per_batch == 0 and step >= FLAGS.silent_before_batch:
        start = time.time()
        calc_test(sess, test_batch, validation_mean_loss, step)
        end = time.time()
        total_test = end - start
        logger.info('step: %09d, test_time: %f', step, total_test)
        logger.info('learning rate: %f, dropout: %f', lr, dp)

      if step % FLAGS.save_per_batch == 0:
        saver.save(sess, FLAGS.log_dir + "/dnn/model", global_step=step)

        export_path = FLAGS.log_dir + "/deep_wide_model/%d" % step
        os.system("rm -fr %s" % export_path)
        save(export_path, sess, dnn_inputs, wide_inputs, train_prediction)

      if step % FLAGS.log_per_batch == 0:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        _, mean_loss, tmp_step, summary = sess.run([train_op, train_mean_loss, global_step, summary_op],
                                                   run_metadata=run_metadata, options=run_options,
                                                   feed_dict={learning_rate : lr})
        writer.add_run_metadata(run_metadata, 'step%d' % step)
        writer.add_summary(summary, step)
        logger.info('step: %09d, cur_train_time: %f, train_mean_loss: %f', step, cur_train, mean_loss)
        cur_train = 0
        step = tmp_step
        lr, dp = get_config()
      else:
        start = time.time()
        _, mean_loss, step = sess.run([train_op, train_mean_loss, global_step], feed_dict={
          learning_rate: lr
        })
        end = time.time()
        total_train += end - start
        cur_train += end - start
      if step > FLAGS.max_steps:
        break
    coord.request_stop()
    coord.join(threads)
    writer.close()


if __name__ == '__main__':
  with tf.device('/gpu:0'):
    train()
