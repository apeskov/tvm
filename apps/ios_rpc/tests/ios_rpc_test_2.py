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

import numpy as np
import socket

import tvm
from tvm import te, rpc
from tvm.contrib import utils, xcode
from tvm.rpc.tracker import Tracker
from tvm.rpc.proxy import Proxy
from ios_rpc_server import RPCServerIOS

arch = "arm64"
sdk = "iphoneos"
target = "llvm -mtriple=%s-apple-darwin" % arch
key = "some_ios_device"

temp = utils.tempdir()


def prepare_to_do_something():
    n = tvm.runtime.convert(1024)
    A = te.placeholder((n,), name="A")
    B = te.compute(A.shape, lambda *i: A(*i) + 1.0, name="B")
    s = te.create_schedule(B.op)
    xo, xi = s[B].split(B.op.axis[0], factor=64)
    s[B].parallel(xi)
    s[B].pragma(xo, "parallel_launch_point")
    s[B].pragma(xi, "parallel_barrier_when_finish")
    f = tvm.build(s, [A, B], target, name="myadd_cpu")
    path_dso = temp.relpath("cpu_lib.dylib")
    f.export_library(path_dso, xcode.create_dylib, arch=arch, sdk=sdk)


def do_something(remote):
    path_dso = temp.relpath("cpu_lib.dylib")
    dev = remote.cpu(0)
    remote.upload(path_dso)
    f = remote.load_module("cpu_lib.dylib")
    a_np = np.random.uniform(size=1024).astype("float32")
    a = tvm.nd.array(a_np, dev)
    b = tvm.nd.array(np.zeros(1024, dtype="float32"), dev)
    time_f = f.time_evaluator(f.entry_name, dev, number=10)
    cost = time_f(a, b).mean
    np.testing.assert_equal(b.numpy(), a.numpy() + 1)


def _get_server_ip(server):
    if server.host != "0.0.0.0":
        return server.host
    else:
        #  get ip of local host
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 1))
        return s.getsockname()[0]


def test_with_rpc_proxy():
    """
    Host -- Proxy -- RPC serv
    """
    # proxy_serv = Proxy(host="0.0.0.0")  # enforce IPv4 localhost
    # ios_rpc_serv = RPCServerIOS.create_ios_rpc_server(key=key, address=_get_server_ip(proxy_serv),
    #                                                   port=proxy_serv.port, mode="proxy")

    for _ in range(2000):
        # remote = rpc.connect(proxy_serv.host, proxy_serv.port, key=key)
        remote = rpc.connect("0.0.0.0", 9090, key=key)
        try:
            do_something(remote)
        except:
            print("Some error!!!!")
        remote = None  # to release connection to server


def test_with_rpc_tracker():
    """
         tracker
         /     \
    Host   --   RPC serv
    """
    tracker = Tracker(host="0.0.0.0", silent=True)
    ios_rpc_serv = RPCServerIOS.create_ios_rpc_server(key=key, address=_get_server_ip(tracker),
                                                      port=tracker.port, mode="tracker")
    tracker_connection = rpc.connect_tracker(tracker.host, tracker.port)
    # tracker_connection = rpc.connect_tracker("0.0.0.0", 9190)

    prepare_to_do_something()

    start = time.time()
    for _ in range(100):
        remote = tracker_connection.request(key)
        try:
            do_something(remote)
        except:
            print("Some error!!!!")
        remote = None  # to release connection to server

    duration = time.time() - start
    print(f"tracker : {duration:.2f}")


def test_with_rpc_tracker_via_proxy():
    """
         tracker
         /     \
    Host   --   Proxy -- RPC serv
    """
    tracker = Tracker(host="0.0.0.0", silent=True)
    ios_rpc_serv = RPCServerIOS.create_ios_rpc_server(key=key, address=_get_server_ip(tracker),
                                                      port=tracker.port, mode="tracker")
    tracker_connection = rpc.connect_tracker(tracker.host, tracker.port)

    for _ in range(2):
        remote = tracker_connection.request(key)
        do_something(remote)
        remote = None  # to release connection to server

def do_some_test_with_ios()

def test_with_pure_rpc():
    """
    Host  --  RPC serv
    """
    # ios_rpc_serv = RPCServerIOS.create_ios_rpc_server(key=key, mode="pure_server")
    prepare_to_do_something()

    for _ in range(100):
        # remote = rpc.connect(ios_rpc_serv.host, ios_rpc_serv.port, key=key)
        do_something(remote)
        remote = None  # to release connection to server

    ios_rpc_serv = RPCServerIOS.create_ios_rpc_server(key=key, mode="pure_server")
    remote = rpc.connect(ios_rpc_serv.host, ios_rpc_serv.port, key=key)

    do_some_test_with_ios(remote)


if __name__ == "__main__":
    # print("test_with_rpc_tracker")
    test_with_rpc_tracker()
    # print("test_with_rpc_tracker_via_proxy")
    test_with_rpc_tracker_via_proxy()
    # print("test_with_rpc_proxy")
    test_with_rpc_proxy()
    # print("test_with_pure_rpc")
    test_with_pure_rpc()
