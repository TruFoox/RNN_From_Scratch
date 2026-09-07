#include <iostream>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <math.h>
#include "IO.h"
#include "rnn.h"
#include <random> 

int main()
{
	char operation;
	std::cout << "Input operation to train on: + - x /\n";
	std::cin >> operation;

	int maxNum;
	std::cout << "Input maximum number this operation will be performed with:\n";
	std::cin >> maxNum;

	RNN rnn;
	rnn.buildWeights(maxNum);

	rnn.trainWeights(operation, maxNum);
}

