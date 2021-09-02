#include "classification.h"

Queue<std::vector<mlperf::QuerySample>> Program::samples_queue;
CK::DataHandler<tvm::runtime::NDArray> Program::data_handler;


int Program::inference(int img_idx, WorkerData* worker_data) {
    int idx = worker_data->benchmark->get_image_idx(img_idx);
    worker_data->runtime.set_input_zero_copy(0, Program::data_handler.sample_reference(idx));
    worker_data->runtime.run();
    worker_data->runtime.get_output(0, Program::data_handler.output_reference(idx));
    return worker_data->benchmark->get_image_label(
            static_cast<const float *>(Program::data_handler.output_reference(idx)->data));
}

bool Program::query_response(WorkerData* worker_data, int vl) {
    std::vector<mlperf::QuerySample> samples = samples_queue.pop();
    if (samples[0].id == mlperf::QuerySampleIndex(-1)) {
        return false;
    }
    std::vector<mlperf::QuerySampleResponse> responses;
    responses.reserve(samples.size());
    float encoding_buffer[samples.size()];
    int i = 0;
    for (auto s : samples) {
        int predicted_class = inference(s.index, worker_data);

        if (vl > 1) {
            std::cout << "Query image index: " << s.index << " -> Predicted class: " << predicted_class << std::endl
                      << std::endl;
        } else if (vl) {
            std::cout << 'p' << std::flush;
        }

        encoding_buffer[i] = (float) predicted_class;
        responses.push_back({s.id, uintptr_t(&encoding_buffer[i]), sizeof(encoding_buffer[i])});
        ++i;
    }
    mlperf::QuerySamplesComplete(responses.data(), responses.size());
    return true;
}

void Program::worker_action(WorkerData* worker_data, int vl) {
    while (query_response(worker_data, vl)) {}
}

std::thread Program::create_worker(WorkerData* worker_data, int vl) {
    std::thread thr(worker_action, worker_data, vl);
    return thr;
}

Program::Program() {
    ctx = {kDLCPU, 0};
    settings = new CK::BenchmarkSettings();
    session = new CK::BenchmarkSession(get_settings());

    std::cout << "\nLoading graph..." << std::endl;
    auto mod_factory = tvm::runtime::Module::LoadFromFile(get_settings()->graph_file());

    tvm::ShapeTuple input_shape = get_settings()->do_nchw_reorder
            ? tvm::ShapeTuple({get_settings()->batch_size, get_settings()->num_channels, get_settings()->image_size, get_settings()->image_size})
            : tvm::ShapeTuple({get_settings()->batch_size, get_settings()->image_size, get_settings()->image_size, get_settings()->num_channels});
    tvm::ShapeTuple output_shape = {get_settings()->batch_size, get_settings()->num_classes};
    DLDataType dtype{kDLFloat, 32, 1};
    std::cout << CK::format("Input tensor dimensions: %d*%d*%d*%d", input_shape[0], input_shape[1], input_shape[2],
                            input_shape[3]) << std::endl;
    std::cout << CK::format("Output tensor dimensions: %d*%d", output_shape[0], output_shape[1]) << std::endl;

    data_handler.set_sample_factory([input_shape, dtype, ctx=ctx](void* data, int size) {
        auto sample = tvm::runtime::NDArray::Empty(input_shape, dtype, ctx);
        auto* sample_data = static_cast<float*>(sample->data);
        auto* float_data = static_cast<float*>(data);
        std::copy(float_data, float_data + size, sample_data);
        return sample;
    });
    data_handler.set_empty_output_factory([output_shape, dtype, device=ctx]() {
        return tvm::runtime::NDArray::Empty(output_shape, dtype, device);
    });

    workers_data.resize(get_settings()->num_workers);
    workers.resize(get_settings()->num_workers);
    for (int i = 0; i < get_settings()->num_workers; i++) {
        workers_data[i].init_device(ctx);
        workers_data[i].init_runtime(mod_factory);
        workers_data[i].init_benchmark(get_settings());
        workers[i] = create_worker(&workers_data[i], get_settings()->verbosity_level);
    }
}

Program::~Program() {
    for (int i = 0; i < get_settings()->num_workers; i++) {
        Program::samples_queue.push({{mlperf::QuerySampleIndex(-1)}});
    }
    for (auto &worker : workers) {
        worker.join();
    }
    delete session;
    delete settings;
}

void Program::LoadNextBatch(const std::vector<mlperf::QuerySampleIndex> &img_indices) {
    auto vl = get_settings()->verbosity_level;

    if (vl > 1) {
        std::cout << "LoadNextBatch([";
        for (auto idx : img_indices) {
            std::cout << idx << ' ';
        }
        std::cout << "])" << std::endl;
    } else if (vl) {
        std::cout << 'B' << std::flush;
    }
    session->load_filenames(img_indices);
    data_handler.load_images(session, get_settings(), get_settings()->verbosity_level);
    for (auto& worker_data: workers_data) {
        worker_data.benchmark->set_session(session);
    }
    if (vl) {
        std::cout << std::endl;
    }
}

void Program::UnloadBatch(const std::vector<mlperf::QuerySampleIndex> &img_indices) {
    auto b_size = img_indices.size();

    auto vl = get_settings()->verbosity_level;

    if (vl > 1) {
        std::cout << "Unloading a batch[" << b_size << "]" << std::endl;
    } else if (vl) {
        std::cout << 'U' << std::flush;
    }

    data_handler.unload_images();
}


SystemUnderTestTVM::SystemUnderTestTVM(Program *_prg, mlperf::TestScenario _test_scenario)
    : mlperf::SystemUnderTest()
    , prg(_prg)
    , test_scenario(_test_scenario) {
    query_counter = 0;
};

void SystemUnderTestTVM::IssueQuery(const std::vector<mlperf::QuerySample> &samples) {
    if (test_scenario == mlperf::TestScenario::Offline || test_scenario == mlperf::TestScenario::SingleStream) {
        const int batch_size = prg->get_settings()->batch_size;
        const int num_samples = samples.size();
        for (int sample_idx = 0; sample_idx < num_samples; sample_idx += batch_size) {
            int batch_sample_count = sample_idx + batch_size < num_samples ? batch_size : num_samples - sample_idx;
            std::vector<mlperf::QuerySample> batch_samples(batch_sample_count);
            for (int sample_in_batch_idx = 0; sample_in_batch_idx < batch_sample_count; ++sample_in_batch_idx) {
                batch_samples[sample_in_batch_idx] = (samples[sample_idx + sample_in_batch_idx]);
            }
            Program::samples_queue.push(batch_samples);
        }
    } else {    // must be Server
        Program::samples_queue.push(samples);
    }

    ++query_counter;
    auto vl = prg->get_settings()->verbosity_level;
    if (vl > 1) {
        std::cout << query_counter << ") IssueQuery([" << samples.size() << "]," << samples[0].id << ","
                  << samples[0].index << ")" << std::endl;
    } else if (vl) {
        std::cout << 'Q' << std::flush;
    }
}

void SystemUnderTestTVM::FlushQueries() {
    auto vl = prg->get_settings()->verbosity_level;
    if (vl) {
        std::cout << std::endl;
    }
}

void SystemUnderTestTVM::ReportLatencyResults(const std::vector<mlperf::QuerySampleLatency>& latencies_ns) {
    size_t size = latencies_ns.size();
    uint64_t avg = accumulate(latencies_ns.begin(), latencies_ns.end(), uint64_t(0)) / size;

    std::vector<mlperf::QuerySampleLatency> sorted_lat(latencies_ns.begin(), latencies_ns.end());
    sort(sorted_lat.begin(), sorted_lat.end());

    std::cout << std::endl << "------------------------------------------------------------";
    std::cout << std::endl << "|            LATENCIES (in nanoseconds and fps)            |";
    std::cout << std::endl << "------------------------------------------------------------";
    size_t p50 = size * 0.5;
    size_t p90 = size * 0.9;
    std::cout << std::endl << "Number of queries run: " << size;
    std::cout << std::endl << "Min latency:                      " << sorted_lat[0]       << "ns  (" << 1e9/sorted_lat[0]         << " fps)";
    std::cout << std::endl << "Median latency:                   " << sorted_lat[p50]     << "ns  (" << 1e9/sorted_lat[p50]       << " fps)";
    std::cout << std::endl << "Average latency:                  " << avg                 << "ns  (" << 1e9/avg                   << " fps)";
    std::cout << std::endl << "90 percentile latency:            " << sorted_lat[p90]     << "ns  (" << 1e9/sorted_lat[p90]       << " fps)";

    std::cout << std::endl << "Max latency:                      " << sorted_lat[size-1]  << "ns  (" << 1e9/sorted_lat[size-1]    << " fps)";
    std::cout << std::endl << "------------------------------------------------------------ " << std::endl;
}