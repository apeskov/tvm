#include "classification.h"


void TestTVM(Program *prg) {
    const std::string mlperf_conf_path = prg->settings->project_root + "/mlperf.conf";
    const std::string user_conf_path = prg->settings->project_root + "/user.conf";
    std::string model_name = "resnet50";
    std::cout << "Path to mlperf.conf : " << mlperf_conf_path << std::endl;
    std::cout << "Path to user.conf : " << user_conf_path << std::endl;
    std::cout << "Model Name: " << model_name << std::endl;

    mlperf::TestSettings ts;

    ts.scenario = (prg->settings->loadgen_scenario_str == "SingleStream")
                  ? mlperf::TestScenario::SingleStream
                  : (prg->settings->loadgen_scenario_str == "MultiStream")
                    ? mlperf::TestScenario::MultiStream
                    : (prg->settings->loadgen_scenario_str == "MultiStreamFree")
                      ? mlperf::TestScenario::MultiStreamFree
                      : (prg->settings->loadgen_scenario_str == "Server")
                        ? mlperf::TestScenario::Server
                        : (prg->settings->loadgen_scenario_str == "Offline")
                          ? mlperf::TestScenario::Offline
                          : mlperf::TestScenario::SingleStream;

    if (!prg->settings->loadgen_mode_str.empty())
        ts.mode = (prg->settings->loadgen_mode_str == "SubmissionRun")
                  ? mlperf::TestMode::SubmissionRun
                  : (prg->settings->loadgen_mode_str == "AccuracyOnly")
                    ? mlperf::TestMode::AccuracyOnly
                    : (prg->settings->loadgen_mode_str == "PerformanceOnly")
                      ? mlperf::TestMode::PerformanceOnly
                      : (prg->settings->loadgen_mode_str == "FindPeakPerformance")
                        ? mlperf::TestMode::FindPeakPerformance
                        : mlperf::TestMode::SubmissionRun;

    if (ts.FromConfig(mlperf_conf_path, model_name, prg->settings->loadgen_scenario_str)) {
        std::cout << "Issue with mlperf.conf file at " << mlperf_conf_path << std::endl;
        exit(1);
    }

    if (ts.FromConfig(user_conf_path, model_name, prg->settings->loadgen_scenario_str)) {
        std::cout << "Issue with user.conf file at " << user_conf_path << std::endl;
        exit(1);
    }

    mlperf::LogSettings log_settings;
    log_settings.log_output.outdir = prg->settings->result_dir;
    log_settings.log_output.prefix_with_datetime = false;
    log_settings.enable_trace = true;

    if (prg->settings->trigger_cold_run) {
        prg->ColdRun();
    }

    SystemUnderTestTVM sut(prg);
    QuerySampleLibraryTVM qsl(prg);

    size_t num_threads = prg->settings->num_workers;
    if (ts.scenario == mlperf::TestScenario::Server && num_threads > 1) {
      ts.server_num_issue_query_threads = num_threads;

      for (int i = 0; i < num_threads; i++) {
        auto thr = std::thread([prg]() {
          for (int j = 0; j < 10; j++)
            prg->ColdRun();

          mlperf::RegisterIssueQueryThread();
        });
        thr.detach();
      }
    }

    mlperf::StartTest(&sut, &qsl, ts, log_settings);

    if (ts.mode == mlperf::TestMode::AccuracyOnly) {
      std::string python_exe = "../../.venv/bin/python3";
      std::string cmd = python_exe + " " + prg->settings->project_root + "/accuracy-imagenet.py" +
                        " " + "--mlperf-accuracy-file " + log_settings.log_output.outdir +
                        "/mlperf_log_accuracy.json" + " " + "--imagenet-val-file " +
                        prg->settings->project_root + "/val.txt" + " " + "--dtype float32";
      system(cmd.c_str());
    }
}


int main(__unused int argc, __unused char* argv[]) {
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

