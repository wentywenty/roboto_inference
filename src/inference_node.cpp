#include "inference_node.hpp"

ObsStackOrder InferenceNode::parse_obs_stack_order(const std::string& stack_order_name) {
    if (stack_order_name == "frame_major") {
        return ObsStackOrder::FrameMajor;
    }
    if (stack_order_name == "obs_major") {
        return ObsStackOrder::ObsMajor;
    }
    throw std::runtime_error("Unsupported obs stack order: " + stack_order_name);
}

void InferenceNode::update_stacked_obs(std::vector<float>& input_buffer, const std::vector<float>& obs,
                                       int obs_num, int frame_stack, ObsStackOrder stack_order,
                                       const std::vector<int>& field_sizes, bool is_first_frame) {
    if (stack_order == ObsStackOrder::FrameMajor) {
        if (is_first_frame) {
            for (int frame = 0; frame < frame_stack; frame++) {
                std::copy(obs.begin(), obs.end(), input_buffer.begin() + frame * obs_num);
            }
        } else {
            std::move(input_buffer.begin() + obs_num,
                      input_buffer.begin() + frame_stack * obs_num,
                      input_buffer.begin());
            std::copy(obs.begin(), obs.end(), input_buffer.begin() + (frame_stack - 1) * obs_num);
        }
        return;
    }

    int input_offset = 0;
    int obs_offset = 0;

    for (const int field_size : field_sizes) {
        if (is_first_frame) {
            for (int frame = 0; frame < frame_stack; frame++) {
                std::copy(obs.begin() + obs_offset, obs.begin() + obs_offset + field_size,
                          input_buffer.begin() + input_offset + frame * field_size);
            }
        } else {
            std::move(input_buffer.begin() + input_offset + field_size,
                      input_buffer.begin() + input_offset + frame_stack * field_size,
                      input_buffer.begin() + input_offset);
            std::copy(obs.begin() + obs_offset, obs.begin() + obs_offset + field_size,
                      input_buffer.begin() + input_offset + (frame_stack - 1) * field_size);
        }
        input_offset += field_size * frame_stack;
        obs_offset += field_size;
    }
}

void InferenceNode::setup_model(std::unique_ptr<ModelContext>& ctx, std::string model_path, int input_size){
    if (!ctx) {
        ctx = std::make_unique<ModelContext>();
    }

    Ort::SessionOptions session_options;
    session_options.DisablePerSessionThreads();
    session_options.EnableCpuMemArena();
    session_options.EnableMemPattern();
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    
    ctx->session = std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_options);
    
    ctx->num_inputs = ctx->session->GetInputCount();
    if (ctx->num_inputs != 1) {
        throw std::runtime_error("Only single-input ONNX models are supported: " + model_path);
    }
    ctx->input_names.resize(ctx->num_inputs);

    for (size_t i = 0; i < ctx->num_inputs; i++) {
        Ort::AllocatedStringPtr input_name = ctx->session->GetInputNameAllocated(i, allocator_);
        ctx->input_names[i] = input_name.get();
        auto type_info = ctx->session->GetInputTypeInfo(i);
        ctx->input_shape = type_info.GetTensorTypeAndShapeInfo().GetShape();
        if (ctx->input_shape[0] == -1) ctx->input_shape[0] = 1;
    }

    size_t model_input_size = 1;
    for (size_t i = 0; i < ctx->input_shape.size(); i++) {
        model_input_size *= static_cast<size_t>(ctx->input_shape[i]);
    }
    if (model_input_size != static_cast<size_t>(input_size)) {
        throw std::runtime_error(
            "ONNX input size mismatch for " + model_path + ": model expects " +
            std::to_string(model_input_size) + " values, but config provides " + std::to_string(input_size));
    }
    ctx->input_buffer.resize(input_size);

    ctx->num_outputs = ctx->session->GetOutputCount();
    ctx->output_names.resize(ctx->num_outputs);
    ctx->output_buffer.resize(joint_num_);

    for (size_t i = 0; i < ctx->num_outputs; i++) {
        Ort::AllocatedStringPtr output_name = ctx->session->GetOutputNameAllocated(i, allocator_);
        ctx->output_names[i] = output_name.get();
        auto type_info = ctx->session->GetOutputTypeInfo(i);
        ctx->output_shape = type_info.GetTensorTypeAndShapeInfo().GetShape();
    }

    ctx->input_names_raw = std::vector<const char *>(ctx->num_inputs, nullptr);
    ctx->output_names_raw = std::vector<const char *>(ctx->num_outputs, nullptr);
    for (size_t i = 0; i < ctx->num_inputs; i++) {
        ctx->input_names_raw[i] = ctx->input_names[i].c_str();
    }
    for (size_t i = 0; i < ctx->num_outputs; i++) {
        ctx->output_names_raw[i] = ctx->output_names[i].c_str();
    }

    ctx->memory_info = std::make_unique<Ort::MemoryInfo>(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU));
    
    ctx->input_tensor = std::make_unique<Ort::Value>(Ort::Value::CreateTensor<float>(
        *ctx->memory_info, ctx->input_buffer.data(), ctx->input_buffer.size(), ctx->input_shape.data(), ctx->input_shape.size()));
        
    ctx->output_tensor = std::make_unique<Ort::Value>(Ort::Value::CreateTensor<float>(
        *ctx->memory_info, ctx->output_buffer.data(), ctx->output_buffer.size(), ctx->output_shape.data(), ctx->output_shape.size()));
}

void InferenceNode::reset() {
    is_running_.store(false);
    is_interrupt_.store(false);
    is_beyondmimic_.store(false);
    obs_.resize(obs_num_);
    active_ctx_ = normal_ctx_.get();
    std::fill(obs_.begin(), obs_.end(), 0.0f);
    std::fill(cmd_vel_.begin(), cmd_vel_.end(), 0.0f);
    for (auto& segment : obs_segments_) {
        std::fill(segment.begin(), segment.end(), 0.0f);
    }
    for (auto& segment : motion_obs_segments_) {
        std::fill(segment.begin(), segment.end(), 0.0f);
    }
    for (auto& segment : extra_obs_segments_) {
        std::fill(segment.begin(), segment.end(), 0.0f);
    }
    std::fill(perception_obs_buffer_.begin(), perception_obs_buffer_.end(), 0.0f);
    if (normal_ctx_) {
        std::fill(normal_ctx_->input_buffer.begin(), normal_ctx_->input_buffer.end(), 0.0f);
        std::fill(normal_ctx_->output_buffer.begin(), normal_ctx_->output_buffer.end(), 0.0f);
    }
    for (auto& ctx : motion_ctxs_) {
        if (ctx) {
            std::fill(ctx->input_buffer.begin(), ctx->input_buffer.end(), 0.0f);
            std::fill(ctx->output_buffer.begin(), ctx->output_buffer.end(), 0.0f);
        }
    }
    for (int i = 0; i < joint_num_; i++) {
        act_[i] = joint_default_angle_[i];
        last_act_[i] = joint_default_angle_[i];
    }
    is_first_frame_ = true;
    motion_frame_ = 0;
    if(use_interrupt_){
        std::fill(interrupt_action_.begin(), interrupt_action_.end(), 0.0f);
    }
}

void InferenceNode::apply_action() {
    if(!is_running_.load() || !robot_->is_init_.load()){
        return;
    }
    {
        std::unique_lock<std::mutex> lock(act_mutex_);
        for (size_t i = 0; i < act_.size(); i++) {
            last_act_[i] = act_alpha_ * act_[i] + (1 - act_alpha_) * last_act_[i];
        }
    }
    robot_->apply_action(last_act_);
}

void InferenceNode::control() {
    pthread_setname_np(pthread_self(), "control");
    struct sched_param sp{}; sp.sched_priority = 70;
    if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp) != 0) {
        RCLCPP_FATAL(this->get_logger(), "Failed to set realtime priority for control thread");
        rclcpp::shutdown();
        return;
    }
    auto period = std::chrono::microseconds(static_cast<long long>(dt_ * 1000000));
    while(rclcpp::ok()){
        auto loop_start = std::chrono::steady_clock::now();
        try {
            apply_action();
        } catch (const std::exception& e) {
            RCLCPP_FATAL(this->get_logger(), "Exception in control thread: %s", e.what());
            rclcpp::shutdown();
            return;
        }
        auto loop_end = std::chrono::steady_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(loop_end - loop_start);
        auto sleep_time = period - elapsed_time;
        if (sleep_time > std::chrono::microseconds(0)) {
            std::this_thread::sleep_for(sleep_time);
        }
    }
}

void InferenceNode::inference() {
    pthread_setname_np(pthread_self(), "inference");
    struct sched_param sp{}; sp.sched_priority = 70;
    if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp) != 0) {
        RCLCPP_FATAL(this->get_logger(), "Failed to set realtime priority for inference thread");
        rclcpp::shutdown();
        return;
    }
    auto period = std::chrono::microseconds(static_cast<long long>(dt_ * 1000 * 1000 * decimation_));

    while(rclcpp::ok()){
        auto loop_start = std::chrono::steady_clock::now();
        if(!is_running_.load()){
            std::this_thread::sleep_for(period);
            continue;
        }

        try {
            std::unique_lock<std::mutex> mode_lock(mode_mutex_);
            const bool is_beyondmimic = is_beyondmimic_.load();
            std::vector<std::vector<float>>& obs_segments = is_beyondmimic ? motion_obs_segments_ : obs_segments_;
            const std::vector<ObsSourceSpec>& obs_layout = is_beyondmimic ? motion_obs_layout_ : obs_layout_;
            update_obs_segments(obs_segments, obs_layout);
            publish_imu();
            publish_joint_states();
            const std::vector<int>& layout_sizes = is_beyondmimic ? motion_obs_layout_sizes_ : obs_layout_sizes_;
            flatten_obs_segments(obs_segments, layout_sizes, obs_.begin());

            std::transform(obs_.begin(), obs_.end(), obs_.begin(), [this](float val) {
                return std::clamp(val, -clip_observations_, clip_observations_);
            });

            int obs_num = is_beyondmimic ? motion_obs_num_: obs_num_;
            int frame_stack = is_beyondmimic ? motion_frame_stack_ : frame_stack_;
            ObsStackOrder stack_order = is_beyondmimic ? motion_obs_stack_order_ : obs_stack_order_;
            update_stacked_obs(active_ctx_->input_buffer, obs_, obs_num, frame_stack, stack_order, layout_sizes, is_first_frame_);
            if(use_attn_enc_){
                update_obs_segments(extra_obs_segments_, extra_obs_layout_);
                flatten_obs_segments(extra_obs_segments_, extra_obs_layout_sizes_, active_ctx_->input_buffer.begin() + frame_stack * obs_num);
            }
            if (is_beyondmimic) {
                step_motion_frame();
            }
            is_first_frame_ = false;

            active_ctx_->session->Run(Ort::RunOptions{nullptr}, 
                active_ctx_->input_names_raw.data(), active_ctx_->input_tensor.get(), active_ctx_->num_inputs,
                active_ctx_->output_names_raw.data(), active_ctx_->output_tensor.get(), active_ctx_->num_outputs);

            {
                std::unique_lock<std::mutex> lock(act_mutex_);
                for (int i = 0; i < active_ctx_->output_buffer.size(); i++) {
                    active_ctx_->output_buffer[i] = std::clamp(active_ctx_->output_buffer[i], -clip_actions_, clip_actions_);
                    act_[usd2urdf_[i]] = active_ctx_->output_buffer[i];
                    act_[usd2urdf_[i]] = act_[usd2urdf_[i]] * action_scale_ + joint_default_angle_[usd2urdf_[i]];
                }
                if(use_interrupt_ && is_interrupt_.load()){
                    std::unique_lock<std::mutex> lock(interrupt_mutex_);
                    for (size_t i = 0; i < interrupt_action_.size(); i++) {
                        act_[act_.size() - interrupt_action_.size() + i] = interrupt_action_[i];
                    }
                }
                publish_action();
            }
        } catch (const std::exception& e) {
            RCLCPP_FATAL(this->get_logger(), "Exception in inference thread: %s", e.what());
            rclcpp::shutdown();
            return;
        }

        auto loop_end = std::chrono::steady_clock::now();
        // 使用微秒进行计算
        auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(loop_end - loop_start);
        auto sleep_time = period - elapsed_time;

        if (sleep_time > std::chrono::microseconds(0)) {
            std::this_thread::sleep_for(sleep_time);
        } else {
            // 警告信息也使用更精确的单位
            RCLCPP_WARN(this->get_logger(), "Inference loop overran! Took %lld us, but period is %lld us.", static_cast<long long>(elapsed_time.count()), static_cast<long long>(period.count()));
        }
    }
}

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    if (mlockall(MCL_CURRENT | MCL_FUTURE) == -1) {
        RCLCPP_WARN(rclcpp::get_logger("main"), "mlockall failed.");
    }
    pthread_setname_np(pthread_self(), "main");
    struct sched_param sp{}; sp.sched_priority = 50;
    if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp) != 0) {
        RCLCPP_FATAL(rclcpp::get_logger("main"), "Failed to set realtime priority for main thread");
        rclcpp::shutdown();
        return 1;
    }
    std::shared_ptr<InferenceNode> node;
    try {
        node = std::make_shared<InferenceNode>();
        rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
        executor.add_node(node);
        RCLCPP_INFO(node->get_logger(), "Press 'X' to initialize/deinitialize motors");
        RCLCPP_INFO(node->get_logger(), "Press 'A' to reset motors");
        RCLCPP_INFO(node->get_logger(), "Press 'B' to start/pause inference");
        RCLCPP_INFO(node->get_logger(), "Press 'Y' to switch between Gamepad Control / cmd_vel Control");
        if (node->use_interrupt_ || node->use_beyondmimic_){
            RCLCPP_INFO(node->get_logger(), "Press 'LB' to switch policy mode (available in beyondmimic / interrupt modes)");
        }
        if (node->use_beyondmimic_){
            RCLCPP_INFO(node->get_logger(), "Press 'RB' to switch motion sequence (available in beyondmimic mode)");
        }
        RCLCPP_INFO(node->get_logger(), "Right Stick: Control forward, backward, left and right movement");
        RCLCPP_INFO(node->get_logger(), "LT/RT: Control turning (left / right rotation)");
        executor.spin();
    } catch (const std::exception &e) {
        RCLCPP_FATAL(rclcpp::get_logger("main"), "Exception caught: %s", e.what());
    }
    rclcpp::shutdown();
    node.reset();
    return 0;
}
