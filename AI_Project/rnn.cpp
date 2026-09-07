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

    std::uniform_real_distribution<float> ranWeights(-0.01f, 0.01f);
    std::uniform_real_distribution<float> ranHidden(-0.01f, 0.01f);

    bias.resize(hidden_size);
    pre_activation.resize(hidden_size);
    output.resize(hidden_size);
    prediction.resize(input_size);
    input_weights.resize(hidden_size);
    hidden_weights.resize(hidden_size);

    hidden_state.resize(input_size);
    hidden_gradient.resize(input_size);

    for (int i = 0; i < hidden_size; ++i) {
        input_weights[i].resize(input_size);
        hidden_weights[i].resize(hidden_size);
    }

    for (int t = 0; t < input_size; ++t) {
        hidden_state[t].resize(hidden_size, 0.0f);
        hidden_gradient[t].resize(hidden_size, 0.0f);
    }

    for (int i = 0; i < hidden_size; ++i) {
        for (int j = 0; j < input_size; ++j) {
            input_weights[i][j] = ranWeights(gen);
        }
    }

    for (int i = 0; i < hidden_size; ++i) {
        for (int j = 0; j < hidden_size; ++j) {
            hidden_weights[i][j] = ranHidden(gen);
        }
    }

    for (int i = 0; i < hidden_size; ++i) {
        bias[i] = ranHidden(gen);
        output[i] = ranHidden(gen);
    }

    outputBias = 0.0f;

    for (int t = 0; t < input_size; ++t) {
        for (int i = 0; i < hidden_size; ++i) {
            hidden_state[t][i] = 0.0f;
            hidden_gradient[t][i] = 0.0f;
        }
    }
}
void RNN::trainWeights() {
    std::cout << "Training weights..." << std::endl;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0, 100);

    for (int t = 0; t < input_size; ++t) {
        hidden_state[t].resize(hidden_size, 0.0f);
    }

    while (true) {

        for (int t = 0; t < input_size; ++t) {
            for (int i = 0; i < hidden_size; ++i) {
                hidden_state[t][i] = 0.0f;
            }
        }

        for (int t = 0; t < input_size; ++t) {
            forward_pass(t);
        }

        for (int t = input_size - 1; t >= 0; --t) {
            backward_pass(t);
        }

        if (input[1] < 200) {
            input[0] += 1;
            input[1] += 1;
        }
        else {
            input[0] = 0;
            input[1] = 1;
        }
        target = input[1] + 1;
    }
}

void RNN::forward_pass(int n) {
    std::vector<float> contributions(hidden_size, 0.0f);
    std::vector<float> hiddenMath(hidden_size, 0.0f);

    for (int i = 0; i < hidden_size; ++i) {
        for (int j = 0; j < input_size; ++j) {
            contributions[i] += input_weights[i][j] * input[j];
        }
    }

    if (n > 0) {
        for (int i = 0; i < hidden_size; ++i) {
            for (int j = 0; j < hidden_size; ++j) {
                hiddenMath[i] += hidden_state[n - 1][j] * hidden_weights[i][j];
            }
        }
    }

    for (int i = 0; i < hidden_size; ++i) {
        pre_activation[i] = contributions[i] + hiddenMath[i] + bias[i];
        hidden_state[n][i] = tanh(pre_activation[i]);
    }

    prediction[n] = outputBias;

    for (int i = 0; i < hidden_size; ++i) {
        prediction[n] += hidden_state[n][i] * output[i];
    }

    if (n == 0) {
        for (int i = 0; i < input_size; ++i) {
            std::cout << input[i] << " ";
        }
        std::cout << std::endl;

        float error = prediction[n] - target;

        std::cout << "Prediction: " << prediction[n] << " | Correct Answer: " << target << " | Error: " << error << std::endl;
        std::cout << std::endl;
    }
}

void RNN::backward_pass(int n) {
    float output_gradient = 2.0f * (prediction[n] - target);

    for (int i = 0; i < hidden_size; ++i) {
        float dL_dHidden = output_gradient * output[i];


        if (n < input_size - 1) {
            for (int j = 0; j < hidden_size; ++j) {
                dL_dHidden += hidden_weights[j][i] * hidden_gradient[n + 1][j];
            }
        }

        float tanhGradient =1.0f - hidden_state[n][i] * hidden_state[n][i];

        hidden_gradient[n][i] =dL_dHidden * tanhGradient;
        if (hidden_gradient[n][i] == 0 || hidden_gradient[n][i] == -0) {
            std::cout << "Gradients have collapsed\n";
        }
    }


    for (int i = 0; i < hidden_size; ++i) {
        output[i] -=learningRate * output_gradient * hidden_state[n][i];

        bias[i] -= learningRate * hidden_gradient[n][i];

        for (int j = 0; j < input_size; ++j) {
            input_weights[i][j] -= learningRate * hidden_gradient[n][i] * input[j];
        }

        if (n > 0) {
            for (int j = 0; j < hidden_size; ++j) {
                hidden_weights[i][j] -= learningRate * hidden_gradient[n][i] * hidden_state[n - 1][j];
            }
        }
    }

    outputBias -= learningRate * output_gradient;
}