import tensorflow as tf
import logging
import argparse
import sys

parser = argparse.ArgumentParser()

parser.add_argument('--ps_hosts', type=str, help='PS hosts')
parser.add_argument('--worker_hosts', type=str, help='Worker hosts')
parser.add_argument('--job_name', type=str, default="worker", help='Job name')
parser.add_argument('--task_index', type=int, default=0, help='Task index')

# flags.DEFINE_string('hidden_layers', "768,512,256,192,128,64,1", 'Network hidden layers')
parser.add_argument('--hidden_layers', type=str, default="32,32,32", help='Network hidden_layers')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size to train')
parser.add_argument('--max_steps', type=int, default=40000, help='Max number of steps to train')
parser.add_argument('--num_threads', type=int, default=8, help='Number of threads')
parser.add_argument('--log_per_batch', type=int, default=100, help='Batch number of log')
parser.add_argument('--save_per_batch', type=int, default=1000, help='Batch number of save model')
parser.add_argument('--validation_per_batch', type=int, default=100, help='Batch number of do validation and save model')
parser.add_argument('--test_per_batch', type=int, default=100, help='Batch number of do test')
parser.add_argument('--silent_before_batch', type=int, default=0, help='Batch number of do test')
parser.add_argument('--log_dir', type=str, default='/tmp/dnn_log', help='log directory')
parser.add_argument('--result_dir', type=str, default='/tmp/dnn_result', help='result directory')

FLAGS, unparsed = parser.parse_known_args()
tmp = [sys.argv[0]] + unparsed

if not tf.gfile.Exists(FLAGS.log_dir):
  tf.gfile.MakeDirs(FLAGS.log_dir)

if not tf.gfile.Exists(FLAGS.result_dir):
  tf.gfile.MakeDirs(FLAGS.result_dir)

FORMAT = '%(asctime)-15s\t%(levelname)s\t%(message)s'
log_file_name = FLAGS.log_dir + '/dnn' + '_' + FLAGS.job_name + '_' + str(FLAGS.task_index) + '.log'
logging.basicConfig(format=FORMAT, filename=log_file_name, filemode='w', level=logging.DEBUG)
logger = logging.getLogger('dnn')

logger.info(tmp)
logger.info(FLAGS)

data_dir = '/tmp/recommend/feature/deepWide'

stats_dir = data_dir + '/stats'
train_dir = data_dir + '/dnn_samples/train'
validation_dir = data_dir + '/dnn_samples/validation'
test_dir = data_dir + '/dnn_samples/validation'

config_file = './dnn.config'

def get_config():
  with open(config_file, 'r') as config:
    for line in config:
      item = line.strip().split(' ')
      lr = float(item[0])
      dp = float(item[1])
      FLAGS.validation_per_batch = int(item[2])
      FLAGS.test_per_batch = int(item[3])
      FLAGS.silent_before_batch = int(item[4])
      return lr, dp

counter_file = stats_dir + '/train/counters/part-00000'
def get_counter():
  with open(counter_file, 'r') as counter:
    for line in counter:
      item = line.strip().split('\t')
      fea_total = int(item[0])
      fea_num = int(item[1])
      wide_fea_num = int(item[2])
      return fea_total, fea_num, wide_fea_num
total_features, num_features, num_wide_features = get_counter()
logger.info('features: %d, %d, %d' % (total_features, num_features, num_wide_features))

layers = [int(d) for d in FLAGS.hidden_layers.split(",")]
layers.insert(0, total_features)

mem = 0
for d in range(len(layers) - 1):
  mem += (layers[d] * layers[d + 1]) * 4.0 / 1024 / 1024 / 1024  # float32, maybe float16 is enough
logger.info('layers: %s, mem used: %f(Gb)' % (layers, mem))

logger.info(layers)
