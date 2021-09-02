#ifndef TVM_LOADGEN_BENCHMARK_H
#define TVM_LOADGEN_BENCHMARK_H

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <memory>
#include <cstring>
#include <thread>
#include <utility>
#include <vector>
#include <map>

#include "settings.h"

#define DEBUG(msg) std::cout << "DEBUG: " << msg << std::endl;

namespace CK {

    class BenchmarkSession {
    public:
        explicit BenchmarkSession(const BenchmarkSettings* settings): _settings(settings) {
        }

        virtual ~BenchmarkSession() = default;

        const std::vector<std::string>& load_filenames(const std::vector<size_t>& img_indices) {
            _filenames_buffer.clear();
            _filenames_buffer.reserve( img_indices.size() );
            idx2loc.clear();

            auto list_of_available_imagefiles = _settings->list_of_available_imagefiles();
            auto count_available_imagefiles   = list_of_available_imagefiles.size();

            int loc=0;
            for (auto idx : img_indices) {
                if(idx<count_available_imagefiles) {
                    _filenames_buffer.emplace_back(list_of_available_imagefiles[idx]);
                    idx2loc[idx] = loc++;
                } else {
                    std::cerr << "Trying to load filename[" << idx << "] when only " << count_available_imagefiles << " images are available" << std::endl;
                    exit(1);
                }
            }

            return _filenames_buffer;
        }

        const std::vector<std::string>& current_filenames() const { return _filenames_buffer; }

        std::map<int,int> idx2loc;

    private:
        const BenchmarkSettings* _settings;
        std::vector<std::string> _filenames_buffer;
    };


    template <typename TData>
    class StaticBuffer {
    public:
        StaticBuffer(size_t size, std::string dir): _size(size), _dir(std::move(dir)) {
            _buffer = new TData[size];
        }

        virtual ~StaticBuffer() {
            delete[] _buffer;
        }

        TData* data() const { return _buffer; }
        int size() const { return _size; }

    protected:
        const int _size;
        const std::string _dir;
        TData* _buffer;
    };


    class ImageDataLoader {
    public:
        static void load_uint8_image(const std::string &dir, const std::string &filename,
                                     uint8_t *buffer, int size, int vl) {
            auto path = dir + '/' + filename;
            std::ifstream file(path, std::ios::in | std::ios::binary);
            if (!file) throw "Failed to open image data " + path;
            file.read(reinterpret_cast<char *>(buffer), size);
            if (vl > 1) {
                std::cout << "Loaded file: " << path << std::endl;
            } else if (vl) {
                std::cout << 'l' << std::flush;
            }
        }
    };


    class ImageData : public StaticBuffer<uint8_t> {
    public:
        explicit ImageData(const BenchmarkSettings* s)
            : StaticBuffer(s->image_size * s->image_size * s->num_channels, s->images_dir) {}

        void load(const std::string& filename, int vl) {
            ImageDataLoader::load_uint8_image(_dir, filename, _buffer, _size, vl);
        }
    };


    class PreprocessedImageData : public StaticBuffer<float> {
    public:
        explicit PreprocessedImageData(const BenchmarkSettings* s)
                : StaticBuffer(s->image_size * s->image_size * s->num_channels, s->images_dir)
                , _given_channel_means(s->given_channel_means)
                , _given_channel_std(s->given_channel_std)
                , _num_channels(s->num_channels)
                , _nchw_order(s->do_nchw_reorder)
        {}

        void load(const std::string& filename, int vl) {
            auto* load_buffer = new uint8_t[_size];
            ImageDataLoader::load_uint8_image(_dir, filename, load_buffer, _size, vl);
            preprocess_image(load_buffer);
            delete[] load_buffer;
        }

    private:
        void preprocess_image(const uint8_t* load_buffer) {
            size_t pixel_idx = 0;
            size_t image_size = _size / _num_channels;
            for (size_t source_idx = 0; source_idx < _size; source_idx++) {
                if (source_idx != 0 && source_idx % _num_channels == 0) {
                    pixel_idx++;
                }
                size_t channel_idx = source_idx % _num_channels;
                size_t target_idx = _nchw_order ? channel_idx * image_size + pixel_idx : source_idx;

                float pixel_value = load_buffer[source_idx];
                pixel_value = (pixel_value - _given_channel_means[channel_idx]) / _given_channel_std[channel_idx];
                _buffer[target_idx] = pixel_value;
            }
        }

        const float *_given_channel_means;
        const float *_given_channel_std;
        int _num_channels;
        bool _nchw_order;
    };


    class ResultData : public StaticBuffer<float> {
    public:
        explicit ResultData(const BenchmarkSettings* s): StaticBuffer<float>(
                s->num_classes, s->result_dir) {}

        void save(const std::string& filename) const {
            auto path = _dir + '/' + filename + ".txt";
            std::ofstream file(path);
            if (!file) throw "Unable to create result file " + path;
            for (int i = 0; i < _size; i++)
                file << _buffer[i] << std::endl;
        }

        int argmax() const {
            return std::distance(_buffer,
                                 std::max_element(_buffer, _buffer + _size));
        }
    };


    template <typename ForwardIterator>
    class MetricEvaluator {
    public:
        static int argmax(ForwardIterator first, ForwardIterator last) {
            return std::distance(first,
                                 std::max_element(first, last));
        }
    };


    template <typename TData>
    class DataHandler {
    public:
        void set_sample_factory(const std::function<TData(void*, int)>& sample_factory) {
            _sample_factory = sample_factory;
        }

        void set_empty_output_factory(const std::function<TData()>& empty_output_factory) {
            _empty_output_factory = empty_output_factory;
        }

        int load_images(const BenchmarkSession* session, const BenchmarkSettings* settings, int vl) {
            const std::vector<std::string>& image_filenames = session->current_filenames();
            const int length = image_filenames.size();
            _input_samples.resize(length);
            _output_data.resize(length);

            int i = 0;
            for (const auto& image_file : image_filenames) {
                PreprocessedImageData sample(settings);
                sample.load(image_file, vl);
                _input_samples[i] = _sample_factory(sample.data(), sample.size());
                _output_data[i] = _empty_output_factory();
                i++;
            }
            return length;
        }

        void unload_images() {
            _input_samples.clear();
            _output_data.clear();
        }

        const TData& sample_reference(int idx) const { return _input_samples.at(idx); }
        const TData& output_reference(int idx) const { return _output_data.at(idx); }
        TData& output_reference(int idx) { return _output_data.at(idx); }

    private:
        std::vector<TData> _input_samples;
        std::vector<TData> _output_data;
        std::function<TData(void*, int)> _sample_factory;
        std::function<TData()> _empty_output_factory;
    };


    class IBenchmark {
    public:
        bool has_background_class = false;

        virtual ~IBenchmark() = default;

        virtual void set_session(const BenchmarkSession *session) = 0;
        virtual int get_image_idx(int image_id) const = 0;
        virtual int get_image_label(const float* output) const = 0;
    };


    class Benchmark : public IBenchmark {
    public:
        explicit Benchmark(const BenchmarkSettings* settings): _settings(settings) {}

        void set_session(const BenchmarkSession *session) override {
            _session = session;
        }

        int get_image_idx(int image_id) const override {
            return _session->idx2loc.at(image_id);
        }

        int get_image_label(const float* output) const override {
            int probe_offset = has_background_class ? 1 : 0;
            return MetricEvaluator<const float*>::argmax(output + probe_offset,
                                                         output + _settings->num_classes);
        }

    private:
        const BenchmarkSettings* _settings;
        const BenchmarkSession* _session;
    };


} // namespace CK

#endif // TVM_LOADGEN_BENCHMARK_H