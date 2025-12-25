#pragma once

#include "types.h"
#include <unordered_map>
#include <vector>
#include <stdexcept>

class NetworkTopology {
private:
    std::unordered_map<int, std::vector<int>> host_to_gpus_;
    std::unordered_map<int, int> gpu_to_host_;
    std::unordered_map<int, int> host_to_leaf_;

    Bandwidth nvlink_bw_; //Intra-host
    Bandwidth pcie_bw_; //fallback
    Bandwidth rdma_bw_; //Inter-host

public:
    NetworkTopology()
        : nvlink_bw_(1600.0)
        , pcie_bw_(256.0)
        , rdma_bw_(200.0)
    {}

    //Factory: create a simple topology with N hosts, M Gpus per host
    static NetworkTopology create_simple(int num_hosts, int gpus_per_host);

    //get bandwidth between two gpus
    Bandwidth bandwidth_between(int gpu_a, int gpu_b) const;


    bool same_leaf(int gpu_a, int gpu_b) const;

    int get_host(int gpu_id) const;

    const std::vector<int>& gpus_on_host(int host_id) const;

    size_t num_hosts() const { return host_to_gpus_.size(); }
    size_t num_gpus() const { return gpu_to_host_.size(); }

    const Bandwidth& nvlink_bandwidth() const {return nvlink_bw_; }
    const Bandwidth& pcie_bandwidth() const {return pcie_bw_; }
    const Bandwidth& rdma_bandwidth() const {return rdma_bw_; }


};
