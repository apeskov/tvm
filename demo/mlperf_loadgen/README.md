MLPerf runner
=============

Attempt to reach some performance results for Server/Offline modes.

PROJECT_ROOT=/Users/apeskov/git/tvm/demo/mlperf_loadgen \
CK_ENV_TENSORFLOW_MODEL_TFLITE_FILEPATH=/Users/apeskov/git/tvm/demo/__prebuilt/dnnl_int8_resnet50.dylib \
CK_ENV_DATASET_IMAGENET_PREPROCESSED_DIR=/Users/apeskov/git/tvm/demo/mlperf_loadgen/__preprocessed \
CK_ENV_DATASET_IMAGENET_PREPROCESSED_SUBSET_FOF=/Users/apeskov/git/tvm/demo/mlperf_loadgen/__preprocessed/image_list.txt \
CK_RESULTS_DIR=/Users/apeskov/git/tvm/demo/mlperf_loadgen/__tmp \
CK_LOADGEN_BUFFER_SIZE=1024 \
CK_LOADGEN_TRIGGER_COLD_RUN=yes \
CK_VERBOSE=1 \
CK_ENV_DATASET_IMAGENET_PREPROCESSED_INPUT_SQUARE_SIDE=224 \
CK_ENV_BATCH_SIZE=1 \
CK_ENV_NUM_CHANNELS=3 \
CK_ENV_NUM_CLASSES=1000 \
ML_MODEL_GIVEN_CHANNEL_MEANS="123.68 116.78 103.94" \
ML_MODEL_GIVEN_CHANNEL_STD=58.395 57.12 57.375 \
CK_ENV_NCHW=1 \
CK_LOADGEN_SCENARIO=Server \
CK_LOADGEN_MODE=PerformanceOnly \
/path_to/tvm_loadgen

preprocess_image_dataset.py
```shell
_CONVERT_TO_UNSIGNED=0 \
_QUANTIZE=0 \
_QUANT_OFFSET=0 \
_QUANT_SCALE=1 \
_SUBSET_FOF=image_list.txt \
_SUBSET_OFFSET=0 \
_SUBSET_VOLUME=50000 \
_INPUT_SQUARE_SIDE=224 \
_CROP_FACTOR=100.0 \
_DATA_TYPE=uint8 \
_NEW_EXTENSION=rgb8 \
_NORMALIZE_DATA=0 \
_SUBTRACT_MEAN=0 \
_GIVEN_CHANNEL_MEANS="" \
_INTERPOLATION_METHOD=INTER_AREA \
\
python3 ./preprocess_image_dataset.py /Users/apeskov/git/ILSVRC2012/Imagenet ./preprocessed
```
