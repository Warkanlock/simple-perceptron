#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float sigmoidf(float x) { return 1.f / (1.f + expf(-x)); }

typedef float sample[3];

// Training datasets for different logic gates
sample or_train[] = {
    {0, 0, 0},
    {1, 0, 1},
    {0, 1, 1},
    {1, 1, 1},
};

sample and_train[] = {
    {0, 0, 0},
    {1, 0, 0},
    {0, 1, 0},
    {1, 1, 1},
};

sample nand_train[] = {
    {0, 0, 1},
    {1, 0, 1},
    {0, 1, 1},
    {1, 1, 0},
};

sample not_train[] = {
    {0, 0, 1},
    {1, 0, 0},
    {0, 1, 1},
    {1, 1, 0},
};

sample xor_train[] = {
    {0, 0, 0},
    {1, 0, 1},
    {0, 1, 1},
    {1, 1, 0},
};

// Select the training dataset
sample *train = and_train;
size_t train_count = sizeof(and_train) / sizeof(and_train[0]);

float rand_float(void) { return (float)rand() / RAND_MAX; }

// Cost function calculates the mean squared error across all training samples
float cost(float w1, float w2, float b) {
  float result = 0.0f;
  for (size_t i = 0; i < train_count; ++i) {
    float x1 = train[i][0];
    float x2 = train[i][1];
    // Forward propagation to calculate predicted output
    float y = sigmoidf(x1 * w1 + x2 * w2 + b);
    float d = y - train[i][2]; // Difference between predicted and actual output
    result += d * d;
  }
  result /= train_count;
  return result;
}

// Numerical gradient approximation
void dcost(float eps, float w1, float w2, float b, float *dw1, float *dw2,
           float *db) {
  float c = cost(w1, w2, b);
  *dw1 = (cost(w1 + eps, w2, b) - c) / eps;
  *dw2 = (cost(w1, w2 + eps, b) - c) / eps;
  *db = (cost(w1, w2, b + eps) - c) / eps;
}

// Direct gradient calculation
void gcost(float w1, float w2, float b, float *dw1, float *dw2, float *db) {
  *dw1 = *dw2 = *db = 0;
  for (size_t i = 0; i < train_count; ++i) {
    float xi = train[i][0];
    float yi = train[i][1];
    // Forward propagation to calculate network output
    float ai = sigmoidf(xi * w1 + yi * w2 + b);
    // Compute gradient contribution for the sample
    float di = 2 * (ai - train[i][2]) * ai * (1 - ai);
    *dw1 += di * xi;
    *dw2 += di * yi;
    *db += di;
  }
  *dw1 /= train_count;
  *dw2 /= train_count;
  *db /= train_count;
}

int main(void) {
  srand(time(0));          // Seed random number generator
  float w1 = rand_float(); // Initialize weight 1
  float w2 = rand_float(); // Initialize weight 2
  float b = rand_float();  // Initialize bias

  float rate = 0.1; // Learning rate

  // Training loop
  for (size_t i = 0; i < 10000; ++i) {
    float dw1, dw2, db;

    // Uncomment the desired gradient computation method
    float eps = 1e-4;
    dcost(eps, w1, w2, b, &dw1, &dw2, &db); // Numerical gradient approximation
    // gcost(w1, w2, b, &dw1, &dw2, &db); // Direct gradient calculation

    // Update parameters using gradients
    w1 -= rate * dw1;
    w2 -= rate * dw2;
    b -= rate * db;
  }

  // Evaluation
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 2; ++j) {
      float prediction = sigmoidf(i * w1 + j * w2 + b);
      printf("%zu | %zu = %f (rounded to %d)\n", i, j, prediction,
             prediction > 0.5 ? 1 : 0);
    }
  }

  return 0;
}
