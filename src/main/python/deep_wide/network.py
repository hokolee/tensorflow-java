from utils import *
import math
import time

def weights_and_biases(layer1, layer2):
  # weights = tf.get_variable("weights", [layer1, layer2], initializer=tf.truncated_normal_initializer(stddev=0.1))
  weights = tf.get_variable("weights", [layer1, layer2], initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(layer1)), seed=int(time.time())))
  # biases = tf.get_variable("biases", [layer2], initializer=tf.truncated_normal_initializer(stddev=0.1, seed=int(time.time())))
  biases = tf.get_variable("biases", [layer2], initializer=tf.zeros_initializer())
  return weights, biases


def full_connect_layer(inputs, layer1, layer2):
  weights, biases = weights_and_biases(layer1, layer2)
  return tf.matmul(inputs, weights) + biases

def sparse_matmul_layer(ids, values, layer1, layer2):
  weights, biases = weights_and_biases(layer1, layer2)
  inputs = tf.sparse_merge(ids, values, layer1)
  return tf.sparse_tensor_dense_matmul(inputs, weights) + biases

def dense_matmul_layer(inputs, layer1, layer2):
  weights, biases = weights_and_biases(layer1, layer2)
  return tf.matmul(inputs, weights) + biases

def sparse_embedding_layer(ids, values, layer1, layer2):
  weights, biases = weights_and_biases(layer1, layer2)
  # Use "mod" if skewed
  return tf.nn.embedding_lookup_sparse(weights, ids, values, partition_strategy="mod", combiner="mean") + biases

def inference_deep_wide(ids, values, wide_ids, wide_values, dims):
  # Input layer
  with tf.variable_scope('input'):
    tmp0 = sparse_matmul_layer(ids, values, num_features, dims[1])  # may tune
    out0 = tf.nn.relu(tmp0)
  # Hidden 1
  with tf.variable_scope('layer1'):
    tmp1 = full_connect_layer(out0, dims[1], dims[2])
    out1 = tf.nn.relu(tmp1)
  # Hidden 2
  with tf.variable_scope('layer2'):
    tmp2 = full_connect_layer(out1, dims[2], dims[3])
    out2 = tf.nn.relu(tmp2)

  # Output wide layer
  with tf.variable_scope('wide_output'):
    wide_logits = sparse_matmul_layer(wide_ids, wide_values, num_wide_features, 1)  # may tune

  # Output
  with tf.variable_scope('output'):
    batch_logits = full_connect_layer(out2, dims[3], 1)
  logits = wide_logits + batch_logits
  return tf.nn.sigmoid(logits, name="predictions")

def inference(dnn_inputs, wide_inputs, dims):
  # Input layer
  with tf.variable_scope('input'):
    tmp0 = dense_matmul_layer(dnn_inputs, num_features, dims[1])  # may tune
    out0 = tf.nn.relu(tmp0)
  # Hidden 1
  with tf.variable_scope('layer1'):
    tmp1 = full_connect_layer(out0, dims[1], dims[2])
    out1 = tf.nn.relu(tmp1)
  # Hidden 2
  with tf.variable_scope('layer2'):
    tmp2 = full_connect_layer(out1, dims[2], dims[3])
    out2 = tf.nn.relu(tmp2)

  # Output wide layer
  with tf.variable_scope('wide_output'):
    wide_logits = dense_matmul_layer(wide_inputs, num_wide_features, 1)  # may tune

  # Output
  with tf.variable_scope('output'):
    batch_logits = full_connect_layer(out2, dims[3], 1)
  logits = wide_logits + batch_logits
  return tf.nn.sigmoid(logits, name="predictions")

#def inference(ids, values, wide_ids, wide_values, dims):
#  return inference_deep_wide(ids, values, wide_ids, wide_values, dims)

def log_loss(labels, predictions):
  labels = tf.Print(labels, [labels], "labels: ", first_n=100, summarize=20)
  predictions = tf.Print(predictions, [predictions], "predictions: ", first_n=100, summarize=20)
  loss = tf.losses.log_loss(labels=labels, predictions=predictions)
  loss = tf.Print(loss, [loss], "loss: ", first_n=100, summarize=200)
  return loss

