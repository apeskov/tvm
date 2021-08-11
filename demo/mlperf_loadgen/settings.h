#ifndef TVM_LOADGEN_SETTINGS_H
#define TVM_LOADGEN_SETTINGS_H

namespace CK {
    /// Load mandatory string value from the environment.
    inline std::string getenv_s(const std::string& name) {
        const char *value = getenv(name.c_str());
        if (!value)
            throw "Required environment variable " + name + " is not set";
        return std::string(value);
    }

    inline std::string getenv_opt_s(const std::string& name, const std::string& default_value) {
        const char *value = getenv(name.c_str());
        if (!value)
            return default_value;
        else
            return std::string(value);
    }


    /// Load mandatory integer value from the environment.
    inline int getenv_i(const std::string& name) {
        const char *value = getenv(name.c_str());
        if (!value)
            throw "Required environment variable " + name + " is not set";
        return static_cast<int>(std::strtol(value, nullptr, 10));
    }

    /// Load mandatory float value from the environment.
    inline float getenv_f(const std::string& name) {
        const char *value = getenv(name.c_str());
        if (!value)
            throw "Required environment variable " + name + " is not set";
        return std::strtof(value, nullptr);
    }

    /// Load an optional boolean value from the environment.
    inline bool getenv_b(const char *name) {
        std::string value = getenv(name);

        return (value == "YES" || value == "yes" || value == "ON" || value == "on" || value == "1");
    }

    /// Dummy `sprintf` like formatting function using std::string.
    /// It uses buffer of fixed length so can't be used in any cases,
    /// generally use it for short messages with numeric arguments.
    template <typename ...Args>
    inline std::string format(const char* str, Args ...args) {
        char buf[1024];
        sprintf(buf, str, args...);
        return std::string(buf);
    }

    class BenchmarkSettings {
    private:
        std::vector<std::string> _available_image_list;
        const std::string _graph_file           = getenv_s("CK_ENV_TENSORFLOW_MODEL_TFLITE_FILEPATH");;
    public:
        const std::string images_dir            = getenv_s("CK_ENV_DATASET_IMAGENET_PREPROCESSED_DIR");
        const std::string available_images_file = getenv_s("CK_ENV_DATASET_IMAGENET_PREPROCESSED_SUBSET_FOF");
        const std::string result_dir            = getenv_s("CK_RESULTS_DIR");
        const std::string project_root          = getenv_s("PROJECT_ROOT");
        const int images_in_memory_max          = getenv_i("CK_LOADGEN_BUFFER_SIZE");
        const bool trigger_cold_run             = getenv_b("CK_LOADGEN_TRIGGER_COLD_RUN");
        const int verbosity_level               = getenv_i("CK_VERBOSE");

        const int image_size                    = getenv_i("CK_ENV_DATASET_IMAGENET_PREPROCESSED_INPUT_SQUARE_SIDE");
        const int batch_size                    = getenv_i("CK_ENV_BATCH_SIZE");
        const int num_channels                  = getenv_i("CK_ENV_NUM_CHANNELS");
        const int num_classes                   = getenv_i("CK_ENV_NUM_CLASSES");
        const bool do_nchw_reorder              = getenv_b("CK_ENV_NCHW");
        const char *given_channel_means_str     = getenv("ML_MODEL_GIVEN_CHANNEL_MEANS");
        const char *given_channel_std_str       = getenv("ML_MODEL_GIVEN_CHANNEL_STD");

        const std::string loadgen_scenario_str  = getenv_s("CK_LOADGEN_SCENARIO");
        const std::string loadgen_mode_str      = getenv_s("CK_LOADGEN_MODE");

        float given_channel_means[3]{};
        float given_channel_std[3]{};

        BenchmarkSettings() {
            // Parse normalization parameters
            if (given_channel_means_str) {
                std::stringstream ss_mean(given_channel_means_str);
                std::stringstream ss_std(given_channel_std_str);
                for (int i = 0; i < 3; i++) {
                    ss_mean >> given_channel_means[i];
                    ss_std >> given_channel_std[i];
                }
            }

            // Load list of images to be processed
            std::ifstream file(available_images_file);
            if (!file)
                throw "Unable to open the available image list file " + available_images_file;
            for (std::string s; !getline(file, s).fail();)
                _available_image_list.emplace_back(s);

            // Create results dir if none
            auto dir = opendir(result_dir.c_str());
            if (dir) closedir(dir); else system(("mkdir " + result_dir).c_str());

            print_settings();
        }

        std::string graph_file() const { return _graph_file; }
        const std::vector<std::string>& list_of_available_imagefiles() const { return _available_image_list; }

        void print_settings() const {
            std::cout << "******* Benchmark Settings *******" << std::endl;
            std::cout << "Graph file: " << _graph_file << std::endl;
            std::cout << "Image dir: " << images_dir << std::endl;
            std::cout << "Image list: " << available_images_file << std::endl;
            std::cout << "Number of available imagefiles: " << _available_image_list.size() << std::endl;
            std::cout << "Result dir: " << result_dir << std::endl;
            std::cout << "Project root: " << project_root << std::endl;
            std::cout << "How many images fit in memory buffer: " << images_in_memory_max << std::endl;
            std::cout << "Trigger cold run: " << trigger_cold_run << std::endl;
            std::cout << "Verbosity level: " << verbosity_level << std::endl;

            std::cout << "Image size: " << image_size << std::endl;
            std::cout << "Batch size: " << batch_size << std::endl;
            std::cout << "Image channels: " << num_channels << std::endl;
            std::cout << "Prediction classes: " << num_classes << std::endl;
            std::cout << "Do NCHW reorder: " << do_nchw_reorder << std::endl;
            if(given_channel_means_str)
                std::cout << "Per-channel means to subtract: " << given_channel_means[0]
                          << ", " << given_channel_means[1]
                          << ", " << given_channel_means[2] << std::endl;
            if(given_channel_std_str)
                std::cout << "Per-channel std to scaling: " << given_channel_std[0]
                          << ", " << given_channel_std[1]
                          << ", " << given_channel_std[2] << std::endl;

            std::cout << "LoadGen Scenario: " << loadgen_scenario_str << std::endl;
            std::cout << "LoadGen Mode: " << (!loadgen_mode_str.empty() ? loadgen_mode_str : "(empty string)" ) << std::endl;
            std::cout << "**********************************" << std::endl;
        }
    };
} // namespace CK

#endif //TVM_LOADGEN_SETTINGS_H
