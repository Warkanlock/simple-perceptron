#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>

#define epoch 100000
#define learning_rate 1e-2
#define epsilon 1e-3
#define train_size sizeof(train) / sizeof(train[0])
#define TEST_ENABLED true

// activation function
float activation(float x) {
    // sigmoid
    return 1.f / (1.f + expf(-x));
}

// train model data (x = x * 2)
float train[][2] = {
    {0.0, 0.0},
    {1.0, 2.0},
    {2.0, 4.0},
    {3.0, 6.0},
    {4.0, 8.0},
    {5.0, 10.0},
    {6.0, 12.0},
    {7.0, 14.0},
    {8.0, 16.0},
    {9.0, 18.0},
    {10.0, 20.0}
};

void define_seed(int seed){
    if(seed == 0) {
        srand((unsigned int)time(NULL));
    } else {
        srand((unsigned int)seed);
    }
}

float generate_random(float magnitude)  {
    return ((float) rand() / (float) RAND_MAX) * magnitude;
}

float cost(float weight, float bias) {
    float cost_result = 0.0f;

    for (size_t i = 0; i < train_size; i++) {
        float input = train[i][0];
        float target = train[i][1];

        // train model ( y = mx + b )
        float output = activation(input * weight + bias);

        // calculate error ( y - target )
        float error = output - target;

        cost_result += (error*error) / train_size;
    }

    return cost_result;
}

float train_weight(float intial_weight) {
    // cost function
    float h = epsilon;
    float rate = learning_rate;
    float bias = generate_random(1.0f);

    printf("bias: %f\n", bias);

    // weight used to train model
    float weight = intial_weight;

    // cost approximation to a derivate state
    for(size_t i = 0; i < epoch; ++i) {
        // calculate initial cost against weight / bias
        float initial_cost = cost(weight, bias);

        // calculate derivate of weight + bias
        float d_cost = (cost(weight + h, bias) - initial_cost)/h;
        float d_bias = (cost(weight, bias + h) - initial_cost)/h;

        // modify parameters to follow approximation
        weight -= rate * d_cost;
        bias -= rate * d_bias;
    }

    // return trained weight
    return weight;
}

float* prepare_weights(size_t total_weights) {
    float* weights = (float*) malloc(total_weights * sizeof(float));

    for(size_t i = 0; i < total_weights; ++i) {
        weights[i] = generate_random(1.0f);
    }

    return weights;
}

int main(void) {
    define_seed(time(0));

    size_t model_weights = 1;
    float* weights = prepare_weights(model_weights);

    // train model
    for(size_t i = 0; i < model_weights; ++i) {
        printf("initial weight[%d]: %f\n", (int)i, weights[i]);
        weights[i] = train_weight(weights[i]);
        printf("final weight[%d]: %f\n", (int)i, weights[i]);
    }

    // inference, used the trained weights against a test set
    if(TEST_ENABLED == true) {
        for(size_t i = 0; i < train_size; ++i) {
            float input = train[i][0];
            float target = train[i][1];

            // use the weight we just trained to predict the output
            float output = input * weights[0];

            printf("input: %f, target: %f, prediction: %f, accuracy: %.2f%% \n", input, target, output, 100 - (target - output));
        }
    } else {
        printf("Test is disabled\n");
    }

    return 0;
}
