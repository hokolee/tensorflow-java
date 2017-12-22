import os
from utils import *

def save(export_path, sess, dnn_inputs, wide_inputs, train_prediction):
  print ("Exporting trained model to" + export_path)

  with tf.device('/cpu:0'):
    logger.info('features: %d, %d' % (num_features, num_wide_features))

    builder = tf.saved_model.builder.SavedModelBuilder(export_path)

    inputs = {
        'dnn_inputs':tf.saved_model.utils.build_tensor_info(dnn_inputs),
        'wide_inputs':tf.saved_model.utils.build_tensor_info(wide_inputs),
    }
    outputs = {'output':tf.saved_model.utils.build_tensor_info(train_prediction)}

    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
    signature = tf.saved_model.signature_def_utils.build_signature_def(inputs, outputs,method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
    builder.add_meta_graph_and_variables(sess,[tf.saved_model.tag_constants.SERVING],{'signature':signature},clear_devices=True,legacy_init_op=legacy_init_op)
    builder.save()