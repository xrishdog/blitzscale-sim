#include "gpu.h"
#include <shared_mutex>

GpuInstance::GpuInstance(int id, double memory_capacity)
  : id_ (id),
    memory_capacity_gb_(memory_capacity),
    memory_used_gb_(0.0),
    state_(InstanceState::IDLE)
{}

bool GpuInstance::has_layer(int model_id, int layer_id) const {
    std::shared_lock<std::shared_mutex> lock(layers_mutex_);

    return loaded_layers_.count({model_id, layer_id}) > 0;
}

bool GpuInstance::load_layer(int model_id, int layer_id, double size_gb) {

    std::unique_lock<std::shared_mutex> lock(layers_mutex_);

    if (memory_used_gb_ + size_gb > memory_capacity_gb_) {
        return false;
    }

    //Check if already loaded
    auto key = std::make_pair(model_id, layer_id);
    if (loaded_layers_.count(key) > 0){
        return true;  //already loaded
    }

    loaded_layers_.insert(key);
    memory_used_gb_ += size_gb;
    return true;
}

bool GpuInstance::unload_layer(int model_id, int layer_id, double size_gb){

    std::unique_lock<std::shared_mutex> lock(layers_mutex_);

    //Check if layer is already unloaded
    auto key = std::make_pair(model_id, layer_id);
    auto it = loaded_layers_.find(key);

    if (it == loaded_layers_.end()){
        return false;
    }

    loaded_layers_.erase(it);
    memory_used_gb_ -= size_gb;

    if (memory_used_gb_ < 0) memory_used_gb_ = 0;
    return true;
}

bool GpuInstance::has_full_model(int model_id, int num_layers) const {
    std::shared_lock<std::shared_mutex> lock(layers_mutex_);

    //check if all the layers are in the set
    for(int i = 0; i < num_layers; i++){
        if (loaded_layers_.count({model_id, i}) == 0){
            return false;
        }
    }
    return true;
}

int GpuInstance::num_layers_loaded(int model_id, int total_layers) const{
    std::shared_lock<std::shared_mutex> lock(layers_mutex_);

    int layers_loaded_count = 0;

    for(int i = 0; i < total_layers; i ++){
        if(loaded_layers_.count({model_id, i}) > 0){
            layers_loaded_count++;
        }
    }

    return layers_loaded_count;
}

InstanceState GpuInstance::get_state() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return state_;
}

void GpuInstance::set_state(InstanceState state){
    std::lock_guard<std::mutex> lock(state_mutex_);
    state_ = state;
}
