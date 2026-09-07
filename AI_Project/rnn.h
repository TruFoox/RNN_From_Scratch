#pragma once
#include <string>
#include <vector>

class RNN {
private:
    int totalIterations;
	float learningRate = 0.000001f;
    int input_size = 2, hidden_size = 1000;
    char operation;
	std::vector<std::vector<float>> hidden_state, input_weights, hidden_weights, hidden_gradient; // Not array because it bitches and whines about the size variables not being a static constant
	std::vector<float> bias, output;
    float outputBias = 0.1;

    int input[2] = { 0, 0 }; // Input for the RNN
    float target = 0; // Correct answer
    std::vector<float> prediction;

    std::vector<float> pre_activation;


public: 
    void buildWeights(int maxNum);

	void trainWeights(char operation, int maxNum);

    void forward_pass(int n);
    void backward_pass(int n);
};