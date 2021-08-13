#ifndef TVM_LOADGEN_BENCHMARK_H
#define TVM_LOADGEN_BENCHMARK_H

#include <cstdio>
#include <cstdlib>

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


    class ImageData : public StaticBuffer<uint8_t> {
    public:
        explicit ImageData(const BenchmarkSettings* s): StaticBuffer(
                s->image_size * s->image_size * s->num_channels,
                s->images_dir) {}

        void load(const std::string& filename, int vl) {
            auto path = _dir + '/' + filename;
            std::ifstream file(path, std::ios::in | std::ios::binary);
            if (!file) throw "Failed to open image data " + path;
            file.read(reinterpret_cast<char*>(_buffer), _size);
            if( vl > 1) {
                std::cout << "Loaded file: " << path << std::endl;
            } else if ( vl ) {
                std::cout << 'l' << std::flush;
            }
        }
    };


    class ResultData : public StaticBuffer<float> {
    public:
        explicit ResultData(const BenchmarkSettings* s): StaticBuffer<float>(
                s->num_classes, s->result_dir) {}

        void save(const std::string& filename) {
            auto path = _dir + '/' + filename + ".txt";
            std::ofstream file(path);
            if (!file) throw "Unable to create result file " + path;
            for (int i = 0; i < _size; i++)
                file << _buffer[i] << std::endl;
        }

        int argmax() {
            int   arg_index = 0;
            float max_value = _buffer[0];

            for (int i = 1; i < _size; i++) {
                if (_buffer[i] > max_value) {
                    arg_index = i;
                    max_value = _buffer[i];
                }
            }

            return arg_index;
        }
    };


    class IBenchmark {
    public:
        bool has_background_class = false;

        virtual ~IBenchmark() = default;
        virtual void load_images(BenchmarkSession *session) = 0;
        virtual void unload_images(size_t num_examples) = 0;
        virtual void save_results() = 0;
        virtual int get_next_result() = 0;
        virtual tvm::runtime::NDArray get_image(int img_idx) = 0;
    };


    template <typename TData, typename TInConverter, typename TOutConverter>
    class Benchmark : public IBenchmark {
    public:
        Benchmark(const BenchmarkSettings* settings, TData *in_ptr, TData *out_ptr): _settings(settings) {
            _in_ptr = in_ptr;
            _out_ptr = out_ptr;
            _in_converter.reset(new TInConverter(settings));
            _out_converter.reset(new TOutConverter(settings));
        }

        void load_images(BenchmarkSession *_session) override {
            session = _session;
            auto vl = _settings->verbosity_level;

            const std::vector<std::string>& image_filenames = session->current_filenames();

            int length = image_filenames.size();
            _current_buffer_size = length;
            _in_batch = new std::unique_ptr<ImageData>[length];
            _out_batch = new std::unique_ptr<ResultData>[length];
            int i = 0;
            for (const auto& image_file : image_filenames) {
                _in_batch[i] = std::make_unique<ImageData>(_settings);
                _out_batch[i] = std::make_unique<ResultData>(_settings);
                _in_batch[i]->load(image_file, vl);
                i++;
            }
        }

        void unload_images(size_t num_examples) override {
            for(size_t i=0;i<num_examples;i++) {
                delete _in_batch[i].get();
                delete _out_batch[i].get();
            }
        }

        tvm::runtime::NDArray get_image(int img_idx) override {
          auto input_shape = tvm::ShapeTuple {1, 3, 224, 224};
          DLDataType dtype{kDLFloat, 32, 1};
          auto ctx = DLDevice {kDLCPU, 0};
          auto tmp_input_tensor = tvm::runtime::NDArray::Empty(input_shape, dtype, ctx);
          auto ptr = static_cast<float *>(tmp_input_tensor->data);
          _in_converter->convert(_in_batch[ session->idx2loc[img_idx] ].get(), ptr);
          return tmp_input_tensor;
        }

        int get_next_result() override {
            int probe_offset = has_background_class ? 1 : 0;
            ResultData *next_result_ptr = _out_batch[_out_buffer_index++].get();
            _out_converter->convert(_out_ptr + probe_offset, next_result_ptr);
            _out_buffer_index %= _current_buffer_size;
            return next_result_ptr->argmax();
        }

        void save_results() override {
            const std::vector<std::string>& image_filenames = session->current_filenames();
            int i = 0;
            for (const auto& image_file : image_filenames) {
                _out_batch[i++]->save(image_file);
            }
        }

    private:
        const BenchmarkSettings* _settings;
        BenchmarkSession* session{};
        int _out_buffer_index = 0;
        int _current_buffer_size = 0;
        TData* _in_ptr;
        TData* _out_ptr;
        std::unique_ptr<ImageData> *_in_batch{};
        std::unique_ptr<ResultData> *_out_batch{};
        std::unique_ptr<TInConverter> _in_converter;
        std::unique_ptr<TOutConverter> _out_converter;
    };


    class IInputConverter {
    public:
        virtual ~IInputConverter() = default;
        virtual void convert(const ImageData* source, void* target) = 0;
    };


    class InNormalize : public IInputConverter {
    public:
        explicit InNormalize(const BenchmarkSettings* s):
                _given_channel_means(s->given_channel_means),
                _given_channel_std(s->given_channel_std),
                _num_channels(s->num_channels),
                _nchw_order(s->do_nchw_reorder) {
        }

        void convert(const ImageData* source, void* target) override {
            auto* float_target = static_cast<float*>(target);

            size_t pixel_idx = 0;
            size_t image_size = source->size() / _num_channels;
            for (size_t source_idx = 0; source_idx < source->size(); source_idx++) {
                if (source_idx != 0 && source_idx % _num_channels == 0) {
                    pixel_idx++;
                }
                size_t channel_idx = source_idx % _num_channels;
                size_t target_idx = _nchw_order ? channel_idx * image_size + pixel_idx : source_idx;

                float pixel_value = source->data()[source_idx];
                pixel_value = (pixel_value - _given_channel_means[channel_idx]) / _given_channel_std[channel_idx];
                float_target[target_idx] = pixel_value;
            }
        }

    private:
        const float *_given_channel_means;
        const float *_given_channel_std;
        const int _num_channels;
        bool _nchw_order = true;
    };


    class OutCopy {
    public:
        explicit OutCopy(const BenchmarkSettings* s) {}

        static void convert(const float* source, ResultData* target) {
            std::copy(source, source + target->size(), target->data());
        }
    };

} // namespace CK

#endif // TVM_LOADGEN_BENCHMARK_H
