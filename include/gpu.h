#pragma once

#include <set>
#include <mutex>
#include <shared_mutex>
#include <atomic>


//GPU states
enum class InstanceState {
    IDLE,
    SERVING,
    SCALING
};

inline const char* state_to_string(InstanceState state){
    switch (state) {
        case InstanceState::IDLE: return "IDLE";
        case InstanceState::SERVING: return "SERVING";
        case InstanceState::SCALING: return "SCALING";
        default: return "UNKNOWN";
    }
}

class GpuInstance {
private:
    int id_;
    double memory_capacity_gb_;
    double memory_used_gb_;

    // (model_id, layer_id ) pairs
    std::set<std::pair<int,int>> loaded_layers_;
    mutable std::shared_mutex layers_mutex_; //Multiple reads, single write

    InstanceState state_;
    mutable std::mutex state_mutex_;

    std::atomic<uint64_t> request_served_{0};

public:
    explicit GpuInstance(int id, double memory_capacity = 80.0);

    //Layer management
    bool has_layer(int model_id, int layer_id) const;
    bool load_layer(int model_id, int layer_id, double size_gb);
    bool unload_layer(int model_id, int layer_id, double size_gb);
    bool has_full_model(int model_id, int num_layers) const;
    int num_layers_loaded(int model_id, int total_layers) const;

    //State management
    InstanceState get_state() const;
    void set_state(InstanceState state);

    // Stats
    void record_request() { request_served_.fetch_add(1, std::memory_order_relaxed); }
    int id() const { return id_; }

    //Accessors
    double memory_used() const {return memory_used_gb_; }
    double memory_capacity() const { return memory_capacity_gb_; }
    double memory_available() const { return memory_capacity_gb_ - memory_used_gb_;}
};
