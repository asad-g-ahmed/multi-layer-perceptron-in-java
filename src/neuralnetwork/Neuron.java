package neuralnetwork;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import engine.Value;

public class Neuron implements Module {

    private List<Value> weights = new ArrayList<>();

    private Value bias;

    private boolean nonLinear;

    public Neuron(int numInputs, boolean nonLinear) {
        double upperLimit = 1.0;
        double lowerLimit = -1.0;
        Random rand = new Random();
        for (int i = 0; i < numInputs; i++) {
            // generate a random double
            double randomDouble = rand.nextDouble() * (upperLimit - lowerLimit) + lowerLimit;
            this.weights.add(new Value(randomDouble));
        }
        this.bias = new Value(0.0);
        this.nonLinear = nonLinear;
    }

    public Value forward(List<Value> inputs) {
        int size = Math.min(inputs.size(), weights.size());
        Value retVal = this.bias;
        for (int i = 0; i < size; i++) {
            retVal = retVal.add(inputs.get(i).multiply(weights.get(i)));
        }
        if (this.nonLinear) {
            return retVal.relu();
        }
        return retVal;
    }

    @Override
    public List<Value> getParameters() {
        List<Value> parameters = new ArrayList<>(this.weights);
        parameters.add(this.bias);
        return parameters;
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append("Neuron [weights=");
        builder.append(weights.size());
        builder.append(", nonLinear=");
        builder.append(nonLinear);
        builder.append("]");
        return builder.toString();
    }

}
