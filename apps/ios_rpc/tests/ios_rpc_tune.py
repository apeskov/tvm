# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Test code for iOS RPC.

To use it, start a rpc proxy with "python -m tvm.exec.rpc_proxy".
And configure the proxy host field as commented.
"""
import time

import tvm
from tvm import relay, auto_scheduler

from tvm.contrib import xcode
import numpy as np

from tvm.rpc.tracker import Tracker
from tvm.rpc.server import Server

proxy_port = 9090
key = "iphone"

# Change target configuration, this is setting for iphone6s
arch = "arm64"
sdk = "iphoneos"
target = "llvm -mtriple=%s-apple-darwin" % arch
target_host = target


def ios_create_dylib(output, objects, *kwargs):
    xcode.create_dylib(output, objects, arch=arch, sdk=sdk)


ios_create_dylib.output_format = "dylib"

# TODO: WA.
auto_scheduler.measure.BuildFunc.name = "custom"
auto_scheduler.measure.BuildFunc.build_func = ios_create_dylib


def get_some_conv_model():
    dtype = "float32"
    shape = [1, 64, 112, 112]
    channels = 64
    groups = 1
    kernel = [3, 3]
    a = relay.var("a", shape=shape, dtype=dtype)
    weight_shape = (channels, shape[1] // groups, *kernel)
    w = tvm.nd.array(np.random.uniform(-128, 127, weight_shape).astype(dtype))
    weights = relay.const(w, dtype)
    out = relay.nn.conv2d(
        a,
        weights,
        kernel_size=kernel,
        groups=groups,
        channels=channels,
        out_dtype=dtype,
    )
    params = {"w": w}

    b = tvm.nd.array(np.random.uniform(-10, 10, weight_shape[0]).astype(dtype))
    biasc = relay.const(b, dtype)
    out = relay.nn.bias_add(out, biasc, axis=1)
    params["b"] = b
    out = relay.nn.relu(out)
    mod = tvm.IRModule.from_expr(out)

    return mod, params


def get_some_model():
    a = relay.var("a", shape=[1, 100])
    b = relay.const(1, dtype="float32")
    out = relay.add(a, b)
    out = relay.erf(out)
    mod = tvm.IRModule.from_expr(out)
    params = {"b": b}

    return mod, params


def test_rpc_auto_schedule():
    """
    Test ability to run auto_schedule tuning on iOS Device/Simulator
    """
    log_file = "blabla.txt"
    mod, params = get_some_conv_model()

    tasks, task_weights = auto_scheduler.extract_tasks(mod, params, target=target, target_host=target_host)

    measure_runner = auto_scheduler.RPCRunner(
        key="iphone", host="0.0.0.0", port=9190,
        repeat=2, min_repeat_ms=10,
        number=1, timeout=2, n_parallel=1
    )

    builder = auto_scheduler.LocalBuilder(build_func=ios_create_dylib, n_parallel=1, timeout=2)

    # Totally it will has 8 trials, 2 round with chunk by 2
    tune_option = auto_scheduler.TuningOptions(
        builder=builder,
        runner=measure_runner,
        num_measure_trials=8,
        num_measures_per_round=4,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=2
    )

    tuner = tvm.auto_scheduler.TaskScheduler(tasks, task_weights)
    tuner.tune(tune_option)
    # TODO(apeskov): check tuning result, no errors and timeouts


def test_rpc_auto_schedule_local():
    """
    Test ability to run auto_schedule tuning on MacOS locally
    """
    log_file = "blabla.txt"  # TODO: make it in tmp folder
    dev_key = "iphone"

    def tracker():
        pass
    tracker.host = "0.0.0.0"
    tracker.port = 9190
    # tracker = Tracker(silent=True)
    # server = Server(key=dev_key, tracker_addr=(tracker.host, tracker.port))
    # server = Server(key=dev_key, tracker_addr=("0.0.0.0", 9190))
    mod, params = get_some_conv_model()

    tasks, task_weights = auto_scheduler.extract_tasks(mod, params, target=target, target_host=target_host)

    # time.sleep(30)

    measure_runner = auto_scheduler.RPCRunner(
        key=dev_key, host=tracker.host, port=tracker.port,
        repeat=2, min_repeat_ms=10,
        number=1, timeout=30, n_parallel=1
    )

    builder = auto_scheduler.LocalBuilder(build_func="default", n_parallel=1, timeout=30)

    # Totally it will has 8 trials, 2 round with chunk by 2
    tune_option = auto_scheduler.TuningOptions(
        builder=builder,
        runner=measure_runner,
        num_measure_trials=8,
        num_measures_per_round=4,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=2
    )

    tuner = tvm.auto_scheduler.TaskScheduler(tasks, task_weights)
    tuner.tune(tune_option)
    # TODO(apeskov): check tuning result, no errors and timeouts

    # server.terminate()
    # tracker.terminate()


if __name__ == "__main__":
    # test_rpc_auto_tvm()
    # test_rpc_auto_tir()
    # test_rpc_auto_schedule()
    test_rpc_auto_schedule_local()
