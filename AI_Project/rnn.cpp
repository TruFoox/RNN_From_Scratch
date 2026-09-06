#define NOMINMAX
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <atomic>
#include <thread>
#include <unordered_map>
#include <math.h>
#include "IO.h"
#include "RNN.h"
#include <mutex>
#include <windows.h>
#include <algorithm>
#include <random>

void RNN::buildWeights() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> ranWeights(-1.0f / sqrt(input_size), 1.0f / sqrt(input_size));
    std::uniform_real_distribution<float> ranHidden(-1.0f / sqrt(hidden_size), 1.0f / sqrt(hidden_size));


    for (int i = 0; i < hidden_size; ++i) {
        for (int j = 0; j < input_size; ++j)
                input_weights[i][j] = ranWeights(gen);
    }

    for (int i = 0; i < hidden_size; ++i) {
        for (int j = 0; j < hidden_size; ++j)
                hidden_weights[i][j] = ranHidden(gen);
    }

    bias.resize(hidden_size);
    initial_hidden_state.resize(hidden_size);
    contributions.resize(hidden_size);
    hiddenMath.resize(hidden_size);
    pre_activation.resize(hidden_size);

    for (int i = 0; i < hidden_size; ++i) {
        bias[i] = ranWeights(gen);
        initial_hidden_state[i] = 0.0f;
    }

}

void RNN::trainWeights() {
    // Placeholder for training logic
    std::cout << "Training weights..." << std::endl;

    forward_pass();

    backward_pass();
}

void RNN::forward_pass() {
    for (int i = 0; i < hidden_size; ++i) {
        for (int j = 0; j < input_size; ++j) {
            contributions[i] += input_weights[i][j] * input[j];
        }
    }

    // Update hidden
    for (int i = 0; i < hidden_size; ++i) {
        for (int j = 0; j < hidden_size; ++j) {
            hiddenMath[i] += initial_hidden_state[j] * hidden_weights[i][j];
        }
    }

    for (int i = 0; i < hidden_size; ++i) {
        pre_activation[i] = contributions[i] + hiddenMath[i] + bias[i];

        initial_hidden_state[i] = tanh(pre_activation[i]);
    }

}