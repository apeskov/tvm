#include "classification.h"
#include <cassert>

Program::Program() {
    settings = new CK::BenchmarkSettings();
    session = new CK::BenchmarkSession(settings);

    std::cout << "\nLoading graph..." << std::endl;
    mod_factory = tvm::runtime::Module::LoadFromFile(settings->graph_file());

    tvm::ShapeTuple input_shape = settings->do_nchw_reorder
            ? tvm::ShapeTuple({settings->batch_size, settings->num_channels, settings->image_size, settings->image_size})
            : tvm::ShapeTuple({settings->batch_size, settings->image_size, settings->image_size, settings->num_channels});
    tvm::ShapeTuple output_shape = {settings->batch_size, settings->num_classes};
    DLDataType dtype{kDLFloat, 32, 1};
    std::cout << CK::format("Input tensor dimensions: %d*%d*%d*%d", input_shape[0], input_shape[1], input_shape[2],
                            input_shape[3]) << std::endl;
    std::cout << CK::format("Output tensor dimensions: %d*%d", output_shape[0], output_shape[1]) << std::endl;

    input_tensor = tvm::runtime::NDArray::Empty(input_shape, dtype, get_runtime().ctx);
    output_tensor = tvm::runtime::NDArray::Empty(output_shape, dtype, get_runtime().ctx);
    benchmark = std::make_unique<CK::Benchmark<float, CK::InNormalize, CK::OutCopy>>(settings,
                                                                                     static_cast<float *>(input_tensor->data),
                                                                                     static_cast<float *>(output_tensor->data));
    benchmark->has_background_class = settings->num_classes == 1001;
}

Program::~Program() {
    delete session;
    delete settings;
}

void Program::LoadNextBatch(const std::vector<mlperf::QuerySampleIndex> &img_indices) {
    auto vl = settings->verbosity_level;

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
    benchmark->load_images(session);
    if (vl) {
        std::cout << std::endl;
    }
}

void Program::ColdRun() {
    auto vl = settings->verbosity_level;

    if (vl > 1) {
        std::cout << "Triggering a Cold Run..." << std::endl;
    } else if (vl) {
        std::cout << 'C' << std::flush;
    }
    auto input_shape = tvm::ShapeTuple {1, 3, 224, 224};
    DLDataType dtype{kDLFloat, 32, 1};

    auto tmp_input_tensor = tvm::runtime::NDArray::Empty(input_shape, dtype, get_runtime().ctx);

    module_inference(tmp_input_tensor);
}

static int __argMax(tvm::runtime::NDArray &arr) {
    assert(arr.Shape().size() == 2);
    assert(arr.Shape()[0] == 1);
    auto size = arr.Shape()[1];
    auto data_ptr = static_cast<float*>(arr->data);
    auto max_val = data_ptr[0];
    auto max_pos = 0;
    for (int i = 0; i < size; i++) {
      if (data_ptr[i] > max_val) {
        max_val = data_ptr[i];
        max_pos = i;
      }
    }
    return max_pos;
}

int Program::InferenceOnce(int img_idx) {
    auto input = benchmark->get_image(img_idx);
    auto res = module_inference(input);
    return __argMax(res);
}

void Program::UnloadBatch(const std::vector<mlperf::QuerySampleIndex> &img_indices) {
    auto b_size = img_indices.size();

    auto vl = settings->verbosity_level;

    if (vl > 1) {
        std::cout << "Unloading a batch[" << b_size << "]" << std::endl;
    } else if (vl) {
        std::cout << 'U' << std::flush;
    }

    benchmark->unload_images(b_size);
}

thread_local RuntimeModule* Program::runtime_ = nullptr;

RuntimeModule& Program::get_runtime() {
  if (Program::runtime_ == nullptr) {
    Program::runtime_ = new RuntimeModule(mod_factory);
  }
  return *Program::runtime_;
}

tvm::runtime::NDArray Program::module_inference(tvm::runtime::NDArray& input) {
    auto runtime = get_runtime();
    runtime.set_input_zero_copy(0, input);
    runtime.run();
    auto res = runtime.get_output(0);
    return res;
}

SystemUnderTestTVM::SystemUnderTestTVM(Program *_prg) : mlperf::SystemUnderTest() {
    prg = _prg;
    query_counter = 0;
};

void SystemUnderTestTVM::IssueQuery(const std::vector<mlperf::QuerySample> &samples) {
    using Time = std::chrono::high_resolution_clock;
    auto start = Time::now();

    ++query_counter;
    auto vl = prg->settings->verbosity_level;
    if (vl > 1) {
        std::cout << query_counter << ") IssueQuery([" << samples.size() << "]," << samples[0].id << ","
                  << samples[0].index << ")" << std::endl;
    } else if (vl) {
        std::cout << 'Q' << std::flush;
    }

    std::vector<mlperf::QuerySampleResponse> responses;
    responses.reserve(samples.size());
    float encoding_buffer[samples.size()];
    int i = 0;
    for (auto s : samples) {
        int predicted_class = prg->InferenceOnce(s.index);
        // int predicted_class = 0;

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
    auto dur = Time::now() - start;
    auto dur_ms = std::chrono::duration_cast<std::chrono::microseconds>(dur).count();
    // std::cout << "[XXX] " << dur_ms << " ms" << std::endl;
}

void SystemUnderTestTVM::FlushQueries() {
    auto vl = prg->settings->verbosity_level;
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

    if(!prg->settings->trigger_cold_run) {
        std::cout << std::endl << "First query (cold model) latency: " << latencies_ns[0]     << "ns  (" << 1e9/latencies_ns[0]       << " fps)";
    }
    std::cout << std::endl << "Max latency:                      " << sorted_lat[size-1]  << "ns  (" << 1e9/sorted_lat[size-1]    << " fps)";
    std::cout << std::endl << "------------------------------------------------------------ " << std::endl;
}
