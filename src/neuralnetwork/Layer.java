package neuralnetwork;

import java.util.ArrayList;
import java.util.List;

import engine.Value;

public class Layer implements Module {

    private List<Neuron> neurons = new ArrayList<>();

    public Layer(int numInputs, int numNeurons, boolean nonLinear) {
        for (int i = 0; i < numNeurons; i++) {
            Neuron neuron = new Neuron(numInputs, nonLinear);
            this.neurons.add(neuron);
        }
    }

    public List<Value> forward(List<Value> inputs) {
        List<Value> retList = new ArrayList<>();
        for (Neuron neuron : this.neurons) {
            Value value = neuron.forward(inputs);
            retList.add(value);
        }
        return retList;
    }

    @Override
    public List<Value> getParameters() {
        List<Value> parameters = new ArrayList<>();
        for (Neuron neuron : this.neurons) {
            parameters.addAll(neuron.getParameters());
        }
        return parameters;
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append("Layer [neurons=");
        builder.append(neurons.size());
        builder.append("]");
        return builder.toString();
    }

}
