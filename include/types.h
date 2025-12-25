#pragma once

#include <string>
#include <vector>
#include <atomic>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <mutex>

struct ModelConfig {
    int id;
    std::string name;
    double parameter_size_gb;
    int num_layers;
    double layer_size_gb;

    //Factory methods (common models)
    static ModelConfig llama3_8b(){
        return {0, "Llama3-8B", 16.0, 32, 0.5};
    }

    static ModelConfig qwen_72b() {
        return {1, "Qwen-72B", 144.0, 80, 1.8};
    }

    static ModelConfig mixtral_8x7b() {
        return {2, "Mixtral-8x7B", 93.0, 32, 2.9};
    }
};

struct Bandwidth {

    double gbps;

    explicit Bandwidth(double gbps = 0.0) : gbps(gbps) {}

    double transfer_time(double size_gb) const {
        if (gbps <= 0 ) return std::numeric_limits<double>::infinity();
        return (size_gb*8.0)/gbps;
    }
    friend std::ostream& operator<<(std::ostream& os, const Bandwidth& bw) {
        os << bw.gbps << "Gpbs";
        return os;
    }
};
