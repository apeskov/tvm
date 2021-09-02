#ifndef TVM_LOADGEN_CLASSIFICATION_H
#define TVM_LOADGEN_CLASSIFICATION_H

#include <algorithm>
#include <memory>
#include <numeric>

#include "tvm/runtime/module.h"
#include "tvm/runtime/packed_func.h"

#include "loadgen.h"
#include "query_sample_library.h"
#include "system_under_test.h"

#include "concurrent_queue.h"
#include "test_settings.h"
#include "benchmark.h"


struct RuntimeModule {
    tvm::runtime::PackedFunc set_input_zero_copy{};
    tvm::runtime::PackedFunc get_output{};
    tvm::runtime::PackedFunc run{};

    RuntimeModule() = default;

    RuntimeModule(tvm::runtime::Module& mod_factory, const DLDevice& ctx) {
        tvm::runtime::Module gmod = mod_factory.GetFunction("default")(ctx);
        set_input_zero_copy = gmod.GetFunction("set_input_zero_copy");
        get_output = gmod.GetFunction("get_output");
        run = gmod.GetFunction("run");
    }
};


struct WorkerData {
    DLDevice device;
    RuntimeModule runtime;
    std::unique_ptr<CK::IBenchmark> benchmark;

    void init_device(const DLDevice& device_) {
        device = device_;
    }

    void init_runtime(tvm::runtime::Module& mod_factory) {
        runtime = RuntimeModule(mod_factory, device);
    }

    void init_benchmark(const CK::BenchmarkSettings* settings) {
        benchmark = std::make_unique<CK::Benchmark>(settings);
        benchmark->has_background_class = settings->num_classes == 1001;
    }
};


class Program {
public:
    Program();
    ~Program();

    void LoadNextBatch(const std::vector<mlperf::QuerySampleIndex>& img_indices);
    void UnloadBatch(const std::vector<mlperf::QuerySampleIndex>& img_indices);
    int available_images_max() const { return get_settings()->list_of_available_imagefiles().size(); }
    int images_in_memory_max() const { return get_settings()->images_in_memory_max; }
    const CK::BenchmarkSettings* get_settings() const { return settings; }

    static Queue<std::vector<mlperf::QuerySample>> samples_queue;
    static CK::DataHandler<tvm::runtime::NDArray> data_handler;
private:
    static int inference(int img_idx, WorkerData* worker_data);
    static bool query_response(WorkerData* worker_data, int vl);
    static void worker_action(WorkerData* worker_data, int vl);
    static std::thread create_worker(WorkerData* worker_data, int vl);

    DLDevice ctx{};
    std::vector<std::thread> workers;
    std::vector<WorkerData> workers_data;

    const CK::BenchmarkSettings *settings;
    CK::BenchmarkSession *session;
};


class SystemUnderTestTVM : public mlperf::SystemUnderTest {
public:
    explicit SystemUnderTestTVM(Program *_prg, mlperf::TestScenario _test_scenario);
    ~SystemUnderTestTVM() override = default;

    const std::string& Name() const override { return name_; }
    void IssueQuery(const std::vector<mlperf::QuerySample>& samples) override;
    void FlushQueries() override;
    void ReportLatencyResults(const std::vector<mlperf::QuerySampleLatency>& latencies_ns) override;

private:
    std::string name_{"TVM_SUT"};
    Program *prg;
    const mlperf::TestScenario test_scenario;
    long query_counter;
};

class QuerySampleLibraryTVM : public mlperf::QuerySampleLibrary {
public:
    explicit QuerySampleLibraryTVM(Program *_prg) : mlperf::QuerySampleLibrary() {
        prg = _prg;
    };
    ~QuerySampleLibraryTVM() override = default;

    const std::string& Name() const override { return name_; }
    size_t TotalSampleCount() override { return prg->available_images_max(); }
    size_t PerformanceSampleCount() override { return prg->images_in_memory_max(); }
    void LoadSamplesToRam( const std::vector<mlperf::QuerySampleIndex>& samples) override {
        prg->LoadNextBatch(samples);
    }
    void UnloadSamplesFromRam( const std::vector<mlperf::QuerySampleIndex>& samples) override {
        prg->UnloadBatch(samples);
    }

private:
    std::string name_{"TVM_QSL"};
    Program *prg;
};

#endif //TVM_LOADGEN_CLASSIFICATION_H