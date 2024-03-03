package neuralnetwork;

import java.util.Collections;
import java.util.List;

import engine.Value;

public interface Module {

    public default void zeroGradients() {
        for (Value value : getParameters()) {
            value.zeroGradient();
        }
    }

    public default List<Value> getParameters() {
        return Collections.emptyList();
    }

}
