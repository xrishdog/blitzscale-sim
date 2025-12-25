#include "worker.h"
#include "gpu.h"
#include "network.h"
#include <thread>

GpuWorker::GpuWorker(int id, GpuInstance& gpu, NetworkTopology& topo)
    : gpu_id_(id)
    , gpu_(gpu)
    , topology_(topo)
{}

GpuWorker::~GpuWorker() {
    stop();
    join();
}

void GpuWorker::start() {
    running_.store(true);
    thread_ = std::thread(&GpuWorker::run, this);
}

void GpuWorker::stop() {
    running_.store(false);
}

void GpuWorker::join() {
    if (thread_.joinable()) {
        thread_.join();
    }
}

void GpuWorker::run() {
    while(running_.load()){

    }
}
