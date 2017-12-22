package com.deep_wide;

import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.framework.MetaGraphDef;
import org.tensorflow.framework.SignatureDef;
import org.tensorflow.framework.TensorInfo;

import java.util.*;

public class Model {
    private static String modelPath = "/tmp/dnn_log/deep_wide_model/1000";
    private SavedModelBundle bundle = null;
    private SignatureDef sig = null;

    public void load(String modePath) throws Exception {
        bundle = SavedModelBundle.load(modePath, "serve");
        sig = MetaGraphDef.parseFrom(bundle.metaGraphDef()).getSignatureDefOrThrow("signature");
    }

    public float[] predict(Map<String, Object> inputValueMap) {
        try {
            Map<String, TensorInfo> inputMap = sig.getInputsMap();
            Session.Runner runner = this.bundle.session().runner();

            List<Tensor> tensorlist = new ArrayList<>();
            for (String key : inputValueMap.keySet()) {
                String tensorName = inputMap.get(key).getName();
                System.out.println(tensorName + "," + inputMap.get(key).getDtype() + "," + inputMap.get(key).getTensorShape());
                Tensor t = Tensor.create(inputValueMap.get(key));
                runner.feed(tensorName, t);
                tensorlist.add(t);
            }
            String outputName = sig.getOutputsOrThrow("output").getName();
            System.out.println("start" + outputName);

            Tensor y = runner.fetch(outputName).run().get(0);

            System.out.println("end" + y.toString());

            float[][] m = new float[1][1];
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

    public static void main(String[] args) throws Exception {
        Model model = new Model();
        model.load(modelPath);

        float[][] value_matrix = getFloats(559);
        float[][] wide_matrix = getFloats(554);

        Map<String, Object> paramMap = new LinkedHashMap<>();
        paramMap.put("dnn_inputs", value_matrix);
        paramMap.put("wide_inputs", wide_matrix);

        float[] res = model.predict(paramMap);
        System.out.println("res:" + Arrays.toString(res));
    }

    private static float[][] getFloats(int num_features) {
        float[][] value_matrix = new float[1][num_features];
        float[] inputFloat = new float[num_features];
        for (int i = 0; i < num_features; i++) {
            inputFloat[i] = (float) 0;
        }
        value_matrix[0] = inputFloat;
        return value_matrix;
    }
}
