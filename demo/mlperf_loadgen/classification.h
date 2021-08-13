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

#include "test_settings.h"
#include "benchmark.h"


struct RuntimeModule {
    DLDevice ctx{};
    tvm::runtime::PackedFunc set_input;
    tvm::runtime::PackedFunc set_input_zero_copy;
    tvm::runtime::PackedFunc get_output;
    tvm::runtime::PackedFunc run;

    explicit RuntimeModule(const std::string& module_path) {
        ctx = {kDLCPU, 0};
        tvm::runtime::Module mod_factory = tvm::runtime::Module::LoadFromFile(module_path);
        tvm::runtime::Module gmod = mod_factory.GetFunction("default")(ctx);
        set_input = gmod.GetFunction("set_input");
        set_input_zero_copy = gmod.GetFunction("set_input_zero_copy");
        get_output = gmod.GetFunction("get_output");
        run = gmod.GetFunction("run");
    }
};


class Program {
public:
    Program();
    ~Program();

    void LoadNextBatch(const std::vector<mlperf::QuerySampleIndex>& img_indices);
    void ColdRun();
    int InferenceOnce(int img_idx);
    void UnloadBatch(const std::vector<mlperf::QuerySampleIndex>& img_indices);

    int available_images_max() const { return settings->list_of_available_imagefiles().size(); }
    int images_in_memory_max() const { return settings->images_in_memory_max; }

    CK::BenchmarkSettings *settings;
private:
    tvm::runtime::NDArray module_inference(tvm::runtime::NDArray& input);

    CK::BenchmarkSession *session;
    std::unique_ptr<CK::IBenchmark> benchmark;
    std::unique_ptr<RuntimeModule> runtime;
    tvm::runtime::NDArray input_tensor;
    tvm::runtime::NDArray output_tensor;
};


class SystemUnderTestTVM : public mlperf::SystemUnderTest {
public:
    explicit SystemUnderTestTVM(Program *_prg);
    ~SystemUnderTestTVM() override = default;

    const std::string& Name() const override { return name_; }
    void IssueQuery(const std::vector<mlperf::QuerySample>& samples) override;
    void FlushQueries() override;
    void ReportLatencyResults(const std::vector<mlperf::QuerySampleLatency>& latencies_ns) override;

private:
    std::string name_{"TVM_SUT"};
    Program *prg;
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
