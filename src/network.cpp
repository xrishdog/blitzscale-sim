#include "network.h"
#include "types.h"

NetworkTopology NetworkTopology::create_simple(int num_hosts, int gpus_per_host) {
    NetworkTopology topo;

    int gpu_counter = 0;
    for (int host_idx = 0; host_idx < num_hosts; ++host_idx) {
        std::vector<int> gpus;

        for (int i = 0; i < gpus_per_host; ++i){
            int gpu_id = gpu_counter++;
            gpus.push_back(gpu_id);
            topo.gpu_to_host_[gpu_id] = host_idx;
        }
        topo.host_to_gpus_[host_idx] = gpus;
        topo.host_to_leaf_[host_idx] = host_idx/2;
    }

    return topo;
}

Bandwidth NetworkTopology::bandwidth_between(int gpu_a, int gpu_b) const {
    if (gpu_a == gpu_b) {
        return Bandwidth(std::numeric_limits<double>::infinity());

    }

    int host_a = gpu_to_host_.at(gpu_a);
    int host_b = gpu_to_host_.at(gpu_b);

    return (host_a == host_b) ? nvlink_bw_ : rdma_bw_;
}

bool NetworkTopology::same_leaf(int gpu_a, int gpu_b) const {

    int host_a = gpu_to_host_.at(gpu_a);
    int host_b = gpu_to_host_.at(gpu_b);

    return host_to_leaf_.at(host_a) == host_to_leaf_.at(host_b);
}

int NetworkTopology::get_host(int gpu_id) const {
    auto it = gpu_to_host_.find(gpu_id);

    if (it == gpu_to_host_.end()) {
        throw std::out_of_range("GPU ID not found: " + std::to_string(gpu_id));
    }
    return it -> second;
}

const std::vector<int>& NetworkTopology::gpus_on_host(int host_id) const{
    auto it = host_to_gpus_.find(host_id);

    if (it == host_to_gpus_.end()) {
        throw std::out_of_range("Host ID not found: " + std::to_string(host_id));
    }

    return it-> second;
}
