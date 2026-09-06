#include "IO.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>


void write2DVector(const std::string& filename, const std::vector<std::vector<float>>& vec) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        std::cerr << "Error opening file for writing: " << filename << "\n";
        return;
    }

    size_t rows = vec.size();
    size_t cols = vec.empty() ? 0 : vec[0].size();
    out.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    out.write(reinterpret_cast<const char*>(&cols), sizeof(cols));

    for (const auto& row : vec)
        out.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(float));
}

std::vector<std::vector<float>> read2DVector(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Error opening file for reading: " << filename << "\n";
        return {};
    }

    size_t rows, cols;
    in.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    in.read(reinterpret_cast<char*>(&cols), sizeof(cols));

    std::vector<std::vector<float>> vec(rows, std::vector<float>(cols));
    for (auto& row : vec)
        in.read(reinterpret_cast<char*>(row.data()), cols * sizeof(float));

    return vec;
}


void write3DVector(const std::string& filename, const std::vector<std::vector<std::vector<float>>>& vec3D) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        std::cerr << "Error opening file for writing: " << filename << "\n";
        return;
    }

    size_t depth = vec3D.size();
    out.write(reinterpret_cast<const char*>(&depth), sizeof(depth));

    for (const auto& mat : vec3D) {
        size_t rows = mat.size();
        size_t cols = mat.empty() ? 0 : mat[0].size();
        out.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
        out.write(reinterpret_cast<const char*>(&cols), sizeof(cols));

        for (const auto& row : mat)
            out.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(float));
    }
}

std::vector<std::vector<std::vector<float>>> read3DVector(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Error opening file for reading: " << filename << "\n";
        return {};
    }

    size_t depth;
    in.read(reinterpret_cast<char*>(&depth), sizeof(depth));
    std::vector<std::vector<std::vector<float>>> vec3D(depth);

    for (size_t d = 0; d < depth; ++d) {
        size_t rows, cols;
        in.read(reinterpret_cast<char*>(&rows), sizeof(rows));
        in.read(reinterpret_cast<char*>(&cols), sizeof(cols));

        vec3D[d].resize(rows, std::vector<float>(cols));
        for (auto& row : vec3D[d])
            in.read(reinterpret_cast<char*>(row.data()), cols * sizeof(float));
    }

    return vec3D;
}

