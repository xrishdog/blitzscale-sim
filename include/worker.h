#pragma once

#include "gpu.h"
#include "network.h"
#include "types.h"
#include <thread>
#include <atomic>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>

enum class WorkType {
    INFERENCE,      // Run inference on a request
    LOAD_LAYER,     // Load a layer from another GPU or parameter pool
    SYNC_BARRIER,   // Participate in a synchronization barrier
    SHUTDOWN        // Terminate the worker
};

struct WorkItem {
    WorkType type;
    int model_id = -1;
    int layer_id = -1;
    int source_gpu = -1;      // For transfers: where the layer comes from
    int request_id = -1;      // For inference: which request

    static WorkItem inference(int req_id, int model) {
        return {WorkType::INFERENCE, model, -1, -1, req_id};
    }

    static WorkItem load_layer(int model, int layer, int source) {
        return {WorkType::LOAD_LAYER, model, layer, source, -1};
    }

    static WorkItem shutdown() {
        return {WorkType::SHUTDOWN, -1, -1, -1, -1};
    }
};

class GpuWorker {
private:
    int gpu_id_;
    GpuInstance& gpu_;
    NetworkTopology& topology_;
    std::atomic<bool> running_{false};
    std::thread thread_;

    std::queue<WorkItem> work_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;

public:
    GpuWorker(int id, GpuInstance& gpu, NetworkTopology& topo);
    ~GpuWorker();

    void start();
    void stop();
    void join();

    int gpu_id() const { return gpu_id_; }
    bool is_running() const { return running_.load(); }

private:
    void run();

    void handle_inference(const WorkItem& item);
    void handle_load_layer(const WorkItem& item);
};
