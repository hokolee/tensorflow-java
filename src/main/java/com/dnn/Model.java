package com.dnn;

import com.google.protobuf.InvalidProtocolBufferException;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.framework.MetaGraphDef;
import org.tensorflow.framework.SignatureDef;
import org.tensorflow.framework.TensorInfo;

import java.util.*;

public class Model {
    private static String modelPath = "/tmp/minist/1";
    private SavedModelBundle bundle = null;
    private SignatureDef sig = null;

    public void load(String modePath) throws InvalidProtocolBufferException {
        bundle = SavedModelBundle.load(modePath, "serve");
        sig = MetaGraphDef.parseFrom(bundle.metaGraphDef()).getSignatureDefOrThrow("predict_images");
    }

    public float[] predict(Map<String, Object> inputValueMap) {
        try {
            Map<String, TensorInfo> inputMap = sig.getInputsMap();
            Session.Runner runner = this.bundle.session().runner();
            List<Tensor> tensorlist = new ArrayList<>();
            for (String key : inputValueMap.keySet()) {
                String tensorName = inputMap.get(key).getName();
                Tensor t = Tensor.create(inputValueMap.get(key));
                runner.feed(tensorName, t);
                tensorlist.add(t);
            }
            String outputName = sig.getOutputsOrThrow("scores").getName();
            Tensor y = runner.fetch(outputName).run().get(0);

            float[][] m = new float[1][10];
            float[][] vector = y.copyTo(m);

            y.close();
            for (Tensor t : tensorlist) {
                t.close();
            }
            tensorlist.clear();

            return vector[0];
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    public static void main(String[] args) throws InvalidProtocolBufferException {
        Model model = new Model();
        model.load(modelPath);

        float[] inputfloat = new float[784];
        for (int i = 0; i < 784; i++) {
            inputfloat[i] = (float) 0;
        }
        float[][] matrix = new float[1][784];
        matrix[0] = inputfloat;

        Map<String, Object> paramMap = new HashMap<>();
        paramMap.put("images", matrix);
        float[] res = model.predict(paramMap);

        System.out.println(Arrays.toString(res));

        float maxVal = 0;
        int inc = 0;
        int predict = -1;
        for (float val : res) {
            if (val > maxVal) {
                predict = inc;
                maxVal = val;
            }
            inc++;
        }
        System.out.println("class:" + predict);
    }
}
