package demo;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import engine.Value;
import neuralnetwork.MLP;

public class Demo {

    public static void main(String[] args) throws IOException {
        // create the model
        MLP mlp = new MLP(2, new int[] { 16, 16 });
        // read the inputs from demo-input.csv
        List<List<Value>> inputs = getInputs();
        // read the outputs from demo-output.csv
        List<Value> outputs = getOutputs();
        // run iterations
        for (int i = 0; i < 100; i++) {
            // get the loss by running thru the forward path
            Value val = loss(mlp, inputs, outputs);
            mlp.zeroGradients();
            val.backward();
            double learningRate = 1.0 - 0.9 * i / 100.0;
            List<Value> parameters = mlp.getParameters();
            for (Value parameter : parameters) {
                double data = parameter.getData();
                data -= learningRate * parameter.getGradient();
                parameter.setData(data);
            }
            System.out.println("step " + i + " loss " + val.getData());
        }
    }

    private static Value loss(MLP mlp, List<List<Value>> inputs, List<Value> outputs) {
        List<Value> predictedOutputs = new ArrayList<>();
        for (List<Value> inputRow : inputs) {
            Value out = mlp.forward(inputRow);
            predictedOutputs.add(out);
        }
        // svm "max-margin" loss
        List<Value> losses = new ArrayList<>();
        int predictedOutputSize = predictedOutputs.size();
        int actualOutputSize = outputs.size();
        for (int i = 0; i < Math.min(predictedOutputSize, actualOutputSize); i++) {
            Value val = new Value(1).add(new Value(-1).multiply(outputs.get(i)).multiply(predictedOutputs.get(i)))
                    .relu();
            losses.add(val);
        }
        // average loss
        Value averageLoss = getAverageLoss(losses);
        // L2 regularization
        Value alpha = new Value(0.0001);
        List<Value> parameters = mlp.getParameters();
        Value sumParameters = parameters.get(0).multiply(parameters.get(0));
        for (int i = 1; i < parameters.size(); i++) {
            sumParameters = sumParameters.add(parameters.get(i).multiply(parameters.get(i)));
        }
        Value regularizedLoss = alpha.multiply(sumParameters);
        return averageLoss.add(regularizedLoss);
    }

    private static Value getAverageLoss(List<Value> losses) {
        Value sum = losses.get(0);
        for (int i = 1; i < losses.size(); i++) {
            sum = sum.add(losses.get(i));
        }
        return sum.multiply(new Value(1.0 / losses.size()));
    }

    private static List<List<Value>> getInputs() throws IOException {
        List<List<Value>> inputs = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader("src/demo/demo-input.csv"))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                List<Value> row = new ArrayList<>();
                row.add(new Value(Double.parseDouble(values[0])));
                row.add(new Value(Double.parseDouble(values[1])));
                inputs.add(row);
            }
        }
        return inputs;
    }

    private static List<Value> getOutputs() throws IOException {
        List<Value> outputs = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader("src/demo/demo-output.csv"))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                outputs.add(new Value(Double.parseDouble(values[0])));
            }
        }
        return outputs;
    }

}
