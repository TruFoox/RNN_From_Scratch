#pragma once
#include <string>
#include <vector>

class RNN {
private:

    int input_size = 2, hidden_size = 10;
	std::vector<std::vector<float>> input_weights, hidden_weights; // Not array because it bitches and whines about the size variables not being a static constant
	std::vector<float> bias, initial_hidden_state, output;
    int outputBias = 0.1;

    int input[2] = { 0, 1 };

    std::vector<float> contributions, hiddenMath, pre_activation;


public:
    /* Build random weights */
    void buildWeights();

	void trainWeights();

    void forward_pass();
    void backward_pass();
};