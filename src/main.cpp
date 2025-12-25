#include <iostream>
#include <memory>
#include "types.h"
#include "gpu.h"
#include "network.h"


int main() {
    std::cout << "BlitzScale Simulator Starting..." << std::endl;

    // Example usage
    ModelConfig llama = ModelConfig::llama3_8b();
    std::cout << "Model: " << llama.name << " (" << llama.parameter_size_gb << " GB)" << std::endl;

    auto topology = NetworkTopology::create_simple(4, 8);
    std::cout << "  - Hosts: " << topology.num_hosts() << "\n";
    std::cout << "  - Total GPUs: " << topology.num_gpus() << "\n";
    std::cout << "  - NVLink BW: " << topology.nvlink_bandwidth() << "\n";
    std::cout << "  - RDMA BW: " << topology.rdma_bandwidth() << "\n\n";

    // GpuInstance gpu(0, 80.0); // 80GB GPU
    // std::cout << "GPU initialized with " << gpu.memory_capacity() << " GB capacity" << std::endl;

    std::vector<std::unique_ptr<GpuInstance>> instances;
    for (size_t i = 0; i < topology.num_gpus(); ++i){
        instances.push_back(std::make_unique<GpuInstance>(static_cast<int>(i), 80.0));
    }
    std::cout << "💾 Created " << instances.size() << " GPU instances (80GB each)\n\n";

    // auto& gpu0 = instances[0];
    // auto& gpu1 = instances[1];
    auto bw_same_host = topology.bandwidth_between(0, 1);

    std::cout << "GPU0 -> GPU1 (same host, NVLink):\n";
    std::cout << "  Bandwidth: " << bw_same_host << "\n";
    std::cout << "  " << llama.name << " transfer: " << std::fixed
              << std::setprecision(4) << bw_same_host.transfer_time(llama.parameter_size_gb) << "s\n\n";

    return 0;

}
