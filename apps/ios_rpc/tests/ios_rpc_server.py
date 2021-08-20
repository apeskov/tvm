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

"""
iOS RPC server launcher.
"""
import subprocess
import tempfile


class RPCServerIOS:
    """
    TBD
    """
    # TODO: read that form env var
    TVM_BUILD_DIR = "/Users/apeskov/git/tvm/cmake-build-debug-ios"
    # TODO: make automatic search of debug/release iphoneos/iphinesimulator configuration
    IOS_RPC_BUNDLE_PATH = TVM_BUILD_DIR + "/apps/ios_rpc/ios_rpc/src/ios_rpc-build/" \
                                          "Debug-iphoneos/tvmrpc.app"
    # Temp directory to store caches of incremental install
    IOS_RPC_INCREMENT_CACHE_DIR = tempfile.TemporaryDirectory()

    def __init__(self, key, url="", port=9090, mode="pure_server", device_id="any_on_cable", silent=True):
        """
        TBD
        """
        if device_id in ("any", "any_on_cable", "any_on_wifi"):
            device_id = RPCServerIOS._find_suitable_device(device_id)

        if mode not in ("pure_server", "proxy", "tracker"):
            raise ValueError('mode argument should be one of "pure_server", "proxy", "tracker"')

        self.silent = silent
        self.device_id = device_id
        self.launched = False
        self.connected = False

        prc_args = f"--immediate_connect --host_url={url} --host_port={port} --key={key} --server_mode={mode}"
        self.cmd = [
            "ios-deploy",
            "--debug",
            "--bundle", RPCServerIOS.IOS_RPC_BUNDLE_PATH,
            "--id", self.device_id,
            "--no-wifi",  # do not use wifi connected devices.
            # TODO: using of noninteractive flag break transfering app output. Looks like buffered on lldb side..
            # "--noninteractive",  # print lldb and app output, but do not wait response. Auto exit in case of failure
            "--app_deltas", RPCServerIOS.IOS_RPC_INCREMENT_CACHE_DIR.name,
            "--unbuffered",
            "--args", prc_args
        ]
        self.proc = None
        self.host = None
        self.port = None
        pass

    def __del__(self):
        self.stop()

    def start(self):
        self.proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1,
                                     universal_newlines=True)

    def stop(self):
        if self.proc is not None:
            self.proc.terminate()

    def wait_launch_complete(self):
        marker_error_msg = "Error:"  # General error message from ios-deploy, like device is locked
        marker_err_2_msg = "[ !! ]"  # Other error warning messages
        marker_no_device = "Timed out waiting for device"  # error message from ios-deploy
        marker_stopped = "PROCESS_STOPPED"  # form
        marker_callstack = "First throw call stack"
        marker_connected = "[IOS-RPC] STATE: 2"  # 0 means state Tracker/Proxy is connected
        marker_server_ip = "[IOS-RPC] IP: "
        marker_server_port = "[IOS-RPC] PORT: "

        for line in self.proc.stdout:
            if self.silent is not True:
                print(line, end="")
            found = str(line).find(marker_error_msg)
            if found != -1:
                msg = str(line)[found + len(marker_error_msg):]
                raise RuntimeError("Cannot launch ios_rpc serves. Reason: " + msg)

            found = str(line).find(marker_err_2_msg)
            if found != -1:
                msg = str(line)[found + len(marker_err_2_msg):]
                raise RuntimeError("Cannot launch ios_rpc serves. Reason: " + msg)

            found = str(line).find(marker_no_device)
            if found != -1:
                raise RuntimeError("There is no attached device with UUID " + self.device_id)

            found = str(line).find(marker_stopped)
            if found != -1:
                raise RuntimeError("[ERROR] Crash during RCP server launch.. ")

            found = str(line).find(marker_callstack)
            if found != -1:
                raise RuntimeError("[ERROR] Crash during RCP server launch.. ")

            found = str(line).find(marker_server_ip)
            if found != -1:
                ip = str(line)[found + len(marker_server_ip):].rstrip("\n")
                self.host = ip

            found = str(line).find(marker_server_port)
            if found != -1:
                port = str(line)[found + len(marker_server_port):].rstrip("\n")
                self.port = int(port)

            if str(line).find(marker_connected) != -1:
                # rpc server reports that it successfully connected
                break

    @staticmethod
    def _find_suitable_device(device_type="any_on_cable"):
        # TODO: hardcoded UUID. Should be found via "xcrun simctl list" or "ios-deploy -c --timeout 1"
        return "00008101-000814213E00001E"
        # return "da967e8fa9b3df5d79d7713f6bdaee46098f9d6d"

    @staticmethod
    def create_ios_rpc_server(key, address="", port=0, mode="proxy", silent=True):
        server = RPCServerIOS(key=key, url=address, port=port, mode=mode, silent=silent)
        server.start()
        server.wait_launch_complete()
        return server
