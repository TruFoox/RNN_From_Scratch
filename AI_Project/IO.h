#pragma once
#include <vector>
#include <string>

std::vector<std::vector<float>> read2DVector(const std::string& filenamem, const int embedding_dim);
void write2DVector(const std::string& filename, const std::vector<std::vector<float>>& vec);

std::vector<std::vector<std::vector<float>>> read3DVector(const std::string& filename, const int embedding_dim);
void write3DVector(const std::string& filename, const std::vector<std::vector<std::vector<float>>>& vec3D);