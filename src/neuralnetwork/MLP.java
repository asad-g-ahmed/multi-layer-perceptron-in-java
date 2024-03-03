package neuralnetwork;

import java.util.ArrayList;
import java.util.List;

import engine.Value;

public class MLP implements Module {

    private List<Layer> layers = new ArrayList<>();

    public MLP(int numInputs, int[] numNeuronsInHiddenLayers) {
        // first layer
        Layer layer1 = new Layer(numInputs, numNeuronsInHiddenLayers[0], true);
        this.layers.add(layer1);
        for (int i = 0; i < numNeuronsInHiddenLayers.length - 1; i++) {
            Layer layer = new Layer(numNeuronsInHiddenLayers[i], numNeuronsInHiddenLayers[i + 1], true);
            this.layers.add(layer);
        }
        Layer finalLayer = new Layer(numNeuronsInHiddenLayers[numNeuronsInHiddenLayers.length - 1], 1, false);
        this.layers.add(finalLayer);
    }

    public Value forward(List<Value> inputs) {
        List<Value> retList = new ArrayList<>(inputs);
        for (Layer layer : this.layers) {
            retList = layer.forward(retList);
        }
        return retList.get(0);
    }

    @Override
    public List<Value> getParameters() {
        List<Value> parameters = new ArrayList<>();
        for (Layer layer : this.layers) {
            parameters.addAll(layer.getParameters());
        }
        return parameters;
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append("MLP [layers=");
        builder.append(layers.size());
        builder.append("]");
        return builder.toString();
    }

}
