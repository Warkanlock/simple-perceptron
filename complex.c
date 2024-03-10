#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define EPOCH 10000
#define LEARNING_RATE 0.1
#define N_INPUTS 3
#define N_OUTPUTS 1
#define N_SAMPLES 4
#define BIAS 1

// activation function
double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }

// derivative of the sigmoid function for backpropagation
double sigmoid_derivative(double x) { return x * (1.0 - x); }

typedef struct {
  double weights[N_INPUTS]; // weights for each input + bias
  double output;            // last output of the network
  double delta;             // for storing the error gradient
} NeuralNetwork;

void initialize_network(NeuralNetwork *nn, int samples, int high, int low) {
  for (int i = 0; i < samples; i++) {
    // initialize weights with random values
    nn->weights[i] = (double)rand() / RAND_MAX * (high - low) + low;
  }
}

double forward(NeuralNetwork *nn, double inputs[N_INPUTS]) {
  double activation = nn->weights[N_INPUTS - 1];

  for (int i = 0; i < N_INPUTS - 1; i++) {
    // sum(weight*input) + bias
    activation += nn->weights[i] * inputs[i];
  }

  // apply activation function
  nn->output = sigmoid(activation);

  return nn->output;
}

void calculate_error_gradient(NeuralNetwork *nn, double expected) {
  // calculate gradient of the error
  nn->delta = (expected - nn->output) * sigmoid_derivative(nn->output);
}

void update_weights(NeuralNetwork *nn, double inputs[N_INPUTS], double learning_rate) {
  for (int i = 0; i < N_INPUTS - 1; i++) {
    // update weights based on the error gradient
    nn->weights[i] += learning_rate * nn->delta * inputs[i];
  }

  // update bias weight
  nn->weights[N_INPUTS - 1] += learning_rate * nn->delta;
}

void train(NeuralNetwork *nn, double inputs[N_SAMPLES][N_INPUTS], double expected[N_SAMPLES], int epochs) {
  for (int epoch = 0; epoch < epochs; epoch++) {
    for (int i = 0; i < N_SAMPLES; i++) {
      // forward propagation
      forward(nn, inputs[i]);

      // calculate error and gradient
      calculate_error_gradient(nn, expected[i]);

      // update weights and bias
      update_weights(nn, inputs[i], LEARNING_RATE);
    }
  }
}

int main() {
  NeuralNetwork nn;
  initialize_network(&nn, N_SAMPLES, 1, -1);

  // include bias in inputs
  double inputs[N_SAMPLES][N_INPUTS] = {
      {0, 0, BIAS}, {0, 1, BIAS}, {1, 0, BIAS}, {1, 1, BIAS}};

  // expected outputs for an OR gate
  double outputs[N_SAMPLES] = {0, 1, 1, 1};

  // train the network
  train(&nn, inputs, outputs, EPOCH);

  for (int i = 0; i < N_SAMPLES; i++) {
    // predict using the trained network
    double prediction = forward(&nn, inputs[i]);

    printf("input: [%1.f, %1.f], predicted: %f, original: %f\n",
        inputs[i][0], inputs[i][1],
        prediction, outputs[i]);
  }

  return 0;
}
