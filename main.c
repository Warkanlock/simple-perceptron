#include <stdlib.h>
#include <stdio.h>
#include <time.h>

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

// a few hyperparameters
#define epoch 10000
#define learning_rate 1e-3
#define epsilon 1e-3
#define train_size sizeof(train) / sizeof(train[0])

void define_seed(int seed){
    // define seed,
    // if zero we just rely on time as seed
    if(seed > 0) {
        srand(seed);
    } else {
        srand(time(0));
    }
}

float generate_random(float magnitude)  {
    define_seed(0);
    return ((float) rand() / (float) RAND_MAX) * magnitude;
}

float cost(float weight) {
    float cost_result = 0.0f;

    for (size_t i = 0; i < train_size; i++) {
        float input = train[i][0];
        float target = train[i][1];

        // train model ( y = mx + b )
        float output = input * weight;

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

    // weight used to train model
    float weight = intial_weight;

    // cost approximation to a derivate state
    for(size_t i = 0; i < epoch; ++i) {
        float d_cost = (cost(weight + h) - cost(weight))/h;
        weight -= rate * d_cost;
    }

    // return trained weight
    return weight;
}

float* prepare_weights(size_t total_weights) {
    float* weights = (float*) malloc(total_weights * sizeof(float));

    for(size_t i = 0; i < total_weights; ++i) {
        weights[i] = generate_random(1);
    }

    return weights;
}

int main(void) {
    size_t model_weights = 1;
    float* weights = prepare_weights(model_weights);

    // train model
    for(size_t i = 0; i < model_weights; ++i) {
        printf("training weight->%d\n", (int)i);
        weights[i] = train_weight(weights[i]);
        printf("weight->%d: %f\n", (int)i, weights[i]);
    }

    // used the trained weights against a test set
    for(size_t i = 0; i < train_size; ++i) {
        float input = train[i][0];
        float target = train[i][1];

        // use the weight we just trained to predict the output
        float output = input * weights[0];

        printf("input: %f, target: %f, prediction: %f\n", input, target, output);
    }

    return 0;
}
