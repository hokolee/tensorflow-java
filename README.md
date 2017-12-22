# load tensorflow's model by java

refer: https://github.com/tensorflow/serving/blob/master/tensorflow_serving/example/mnist_saved_model.py
* python mnist_saved_model.py --training_iteration=1 --model_version=1 /tmp/minist
* java -cp  target/tensorflow-java-1.0-SNAPSHOT.jar com.dnn.Model
 