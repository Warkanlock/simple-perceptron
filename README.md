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

## Output
```
training weight->0
weight->0: 1.999446
input: 0.000000, target: 0.000000, prediction: 0.000000
input: 1.000000, target: 2.000000, prediction: 1.999446
input: 2.000000, target: 4.000000, prediction: 3.998891
input: 3.000000, target: 6.000000, prediction: 5.998337
input: 4.000000, target: 8.000000, prediction: 7.997782
input: 5.000000, target: 10.000000, prediction: 9.997228
input: 6.000000, target: 12.000000, prediction: 11.996674
input: 7.000000, target: 14.000000, prediction: 13.996119
input: 8.000000, target: 16.000000, prediction: 15.995564
input: 9.000000, target: 18.000000, prediction: 17.995010
input: 10.000000, target: 20.000000, prediction: 19.994455
```

## Aknowledgements

This implementation is based on the Machine Learning series streamed by Tsoding on Twitch.
