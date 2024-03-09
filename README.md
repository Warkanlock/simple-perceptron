# Simple Perceptron

A simple implementation using C of a Perceptron (kind of) of just one weight (a.k.a neuron).

This implementation allow us to create an approximation of a function using a set of points from a model.

## Example

```c
float model[][2] = {
    {1, 2}
    {2, 4}
    // ...
}

float* weights = prepare_weights(1);
float weight = train_weight(weights[0]);
float prediction = 1 * weight; // 2
```

## Aknowledgements

This implementation is based on the Machine Learning series streamed by Tsoding.
