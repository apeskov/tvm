#include "classification.h"


void TestTVM(Program *prg) {
    const std::string mlperf_conf_path = prg->get_settings()->project_root + "/mlperf.conf";
    const std::string user_conf_path = prg->get_settings()->project_root + "/user.conf";
    std::string model_name = "resnet50";
    std::cout << "Path to mlperf.conf : " << mlperf_conf_path << std::endl;
    std::cout << "Path to user.conf : " << user_conf_path << std::endl;
    std::cout << "Model Name: " << model_name << std::endl;

    mlperf::TestSettings ts;

    ts.scenario = (prg->get_settings()->loadgen_scenario_str == "SingleStream")
                  ? mlperf::TestScenario::SingleStream
                  : (prg->get_settings()->loadgen_scenario_str == "MultiStream")
                    ? mlperf::TestScenario::MultiStream
                    : (prg->get_settings()->loadgen_scenario_str == "MultiStreamFree")
                      ? mlperf::TestScenario::MultiStreamFree
                      : (prg->get_settings()->loadgen_scenario_str == "Server")
                        ? mlperf::TestScenario::Server
                        : (prg->get_settings()->loadgen_scenario_str == "Offline")
                          ? mlperf::TestScenario::Offline
                          : mlperf::TestScenario::SingleStream;

    if (!prg->get_settings()->loadgen_mode_str.empty())
        ts.mode = (prg->get_settings()->loadgen_mode_str == "SubmissionRun")
                  ? mlperf::TestMode::SubmissionRun
                  : (prg->get_settings()->loadgen_mode_str == "AccuracyOnly")
                    ? mlperf::TestMode::AccuracyOnly
                    : (prg->get_settings()->loadgen_mode_str == "PerformanceOnly")
                      ? mlperf::TestMode::PerformanceOnly
                      : (prg->get_settings()->loadgen_mode_str == "FindPeakPerformance")
                        ? mlperf::TestMode::FindPeakPerformance
                        : mlperf::TestMode::SubmissionRun;

    if (ts.FromConfig(mlperf_conf_path, model_name, prg->get_settings()->loadgen_scenario_str)) {
        std::cout << "Issue with mlperf.conf file at " << mlperf_conf_path << std::endl;
        exit(1);
    }

    if (ts.FromConfig(user_conf_path, model_name, prg->get_settings()->loadgen_scenario_str)) {
        std::cout << "Issue with user.conf file at " << user_conf_path << std::endl;
        exit(1);
    }

    mlperf::LogSettings log_settings;
    log_settings.log_output.outdir = prg->get_settings()->result_dir;
    log_settings.log_output.prefix_with_datetime = false;
    log_settings.enable_trace = true;

    SystemUnderTestTVM sut(prg, ts.scenario);
    QuerySampleLibraryTVM qsl(prg);
    mlperf::StartTest(&sut, &qsl, ts, log_settings);

    std::string cmd = std::string("python3") + " " +
                      prg->get_settings()->project_root + "/accuracy-imagenet.py" + " " +
                      "--mlperf-accuracy-file " + log_settings.log_output.outdir + "/mlperf_log_accuracy.json" + " " +
                      "--imagenet-val-file " + prg->get_settings()->project_root + "/val.txt" + " " +
                      "--dtype float32";
    system(cmd.c_str());
}


int main(int argc, char* argv[]) {
    try {
        auto *prg = new Program();
        TestTVM(prg);
        delete prg;
    }
    catch (const std::string& error_message) {
        std::cerr << "ERROR: " << error_message << std::endl;
        return -1;
    }
    return 0;
}