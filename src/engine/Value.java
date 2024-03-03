package engine;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Value {

    private double data;

    private double gradient;

    private Value[] prev;

    private BackPropagatable backPropagatable;

    public Value(double data, Value... children) {
        this.data = data;
        this.gradient = 0.0;
        this.backPropagatable = () -> {
            // nothing to do in default case
        };
        this.prev = children;
    }

    /**
     * @return the data
     */
    public double getData() {
        return data;
    }

    /**
     * @param data
     *            the data to set
     */
    public void setData(double data) {
        this.data = data;
    }

    /**
     * @return the gradient
     */
    public double getGradient() {
        return gradient;
    }

    public Value add(Value other) {
        Value out = new Value(this.data + other.data, this, other);
        out.backPropagatable = () -> {
            this.gradient += out.gradient;
            other.gradient += out.gradient;
        };
        return out;
    }

    public Value multiply(Value other) {
        Value out = new Value(this.data * other.data, this, other);
        out.backPropagatable = () -> {
            this.gradient += other.data * out.gradient;
            other.gradient += this.data * out.gradient;
        };
        return out;
    }

    public Value power(double exponent) {
        Value out = new Value(Math.pow(this.data, exponent), this);
        out.backPropagatable = () -> this.gradient += exponent * Math.pow(this.data, exponent - 1) * out.gradient;
        return out;
    }

    public Value relu() {
        Value out = new Value(this.data < 0.0 ? 0.0 : this.data, this);
        out.backPropagatable = () -> this.gradient += (out.data > 0.0 ? 1.0 : 0.0) * out.gradient;
        return out;
    }

    public void backward() {
        List<Value> topology = new ArrayList<>();
        List<Value> visited = new ArrayList<>();
        buildTopology(this, topology, visited);
        Collections.reverse(topology);
        this.gradient = 1.0;
        for (Value value : topology) {
            value.backPropagatable.backward();
        }
    }

    public void zeroGradient() {
        this.gradient = 0.0;
    }

    private void buildTopology(Value value, List<Value> topology, List<Value> visited) {
        if (!visited.contains(value)) {
            visited.add(value);
            for (Value val : value.prev) {
                buildTopology(val, topology, visited);
            }
            topology.add(value);
        }
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append("Value [data=");
        builder.append(data);
        builder.append(", gradient=");
        builder.append(gradient);
        builder.append("]");
        return builder.toString();
    }

}
