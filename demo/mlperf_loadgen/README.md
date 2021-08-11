MLPerf runner
=============

Attempt to reach some performance results for Server/Offline modes.



PROJECT_ROOT=/Users/apeskov/git/tvm/demo/mlperf_loadgen
CK_ENV_TENSORFLOW_MODEL_TFLITE_FILEPATH=/Users/apeskov/git/tvm/demo/__prebuilt/dnnl_int8_resnet50.dylib
CK_ENV_DATASET_IMAGENET_PREPROCESSED_DIR=/Users/agladyshev/CK/local/venv/reproduce-mlperf/CK/local/env/8048a8c42f09f606;
CK_ENV_DATASET_IMAGENET_PREPROCESSED_SUBSET_FOF=/Users/agladyshev/CK/local/venv/reproduce-mlperf/CK/local/env/8048a8c42f09f606/image_list_5.txt;
CK_RESULTS_DIR=/Users/agladyshev/workspace/tvm-samples/mlperf_loadgen/cmake-build-debug/mlperf_results;
CK_LOADGEN_BUFFER_SIZE=1024;
CK_LOADGEN_TRIGGER_COLD_RUN=yes;
CK_VERBOSE=1;
CK_ENV_DATASET_IMAGENET_PREPROCESSED_INPUT_SQUARE_SIDE=224;
CK_ENV_BATCH_SIZE=1;
CK_ENV_NUM_CHANNELS=3;
CK_ENV_NUM_CLASSES=1000;
ML_MODEL_GIVEN_CHANNEL_MEANS=123.68 116.78 103.94;
ML_MODEL_GIVEN_CHANNEL_STD=58.395 57.12 57.375;
CK_ENV_NCHW=1;
CK_LOADGEN_SCENARIO=Server
CK_LOADGEN_MODE=AccuracyOnly
