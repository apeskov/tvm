import numpy as np
import subprocess

import tvm 
from tvm import meta_schedule as ms
from tvm import dlight as dl
from tvm import relax

from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import relax as R
from tvm.relax.backend.contrib.cublas import partition_for_cublas
from tvm.relax.backend.contrib.cutlass import partition_for_cutlass

import tvm.tir.tensor_intrin.cuda
from tvm.contrib import nvcc
from tvm.contrib import utils

TARGET = tvm.target.Target("nvidia/nvidia-a100")
DEV = tvm.cuda(0)


def make_arg(info):
    if info.dtype in ["float16", "float32"]:
        arr_np = np.random.uniform(-1, 1, size=info.shape).astype(info.dtype)
    elif info.dtype in ["int32", "uint32", "int16", "int8"]:
        arr_np = np.random.randint(0, 16, size=info.shape).astype(info.dtype)
    else:
        assert False, f"Unimplemented, dtype={info.dtype}"

    return tvm.nd.array(arr_np, device=DEV)



def get_sass(cubin):
    temp = utils.tempdir()
    temp_cubin = temp.relpath("my_kernel.cubin")
    with open(temp_cubin, "wb") as out_file:
        out_file.write(cubin)
    
    cmd = [ "nvdisasm", "-c", temp_cubin]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    (out, _) = proc.communicate()

    if proc.returncode != 0:
        msg += "\nCompilation error:\n"
        msg += out.decode("utf-8")
        raise RuntimeError(msg)

    return out.decode("utf-8")


def cuda_dump(lib, dump_path="."):
    src = lib.imported_modules[0].get_source()
    with open(f"{dump_path}/shaders.cu", "w") as f:
        print(src, file=f)

    ptx = nvcc.compile_cuda(src, target_format="ptx")
    with open(f"{dump_path}/shaders.ptx", "wb") as f:
        f.write(ptx)

    cubin = nvcc.compile_cuda(src, target_format="cubin")
    # with open(f"{dump_path}/shaders.cubin", "wb") as f:
        # f.write(cubin)

    sass = get_sass(cubin)
    with open(f"{dump_path}/shaders.sass", "w") as f:
        f.write(sass)


def linear_1_ft_gen(_N, _K): 
    @I.ir_module
    class _mod:
        @T.prim_func(private=True)
        def decode(A: T.Buffer((T.int64(4096), T.int64(11008)), "int8"), B: T.Buffer((T.int64(22016),), "float16"), decode: T.Buffer((T.int64(4096), T.int64(22016)), "float16")):
            # with T.block("root"):
            for i, j in T.grid(T.int64(4096), T.int64(22016)):
                with T.block("decode"):
                    v_i, v_j = T.axis.remap("SS", [i, j])
                    T.reads(A[v_i, v_j // T.int64(2)], B[v_j])
                    T.writes(decode[v_i, v_j])
                    decode[v_i, v_j] = T.Cast("float16", T.shift_right(T.shift_left(T.bitwise_and(T.shift_right(T.Cast("int32", A[v_i, v_j // T.int64(2)]), T.Cast("int32", v_j % T.int64(2)) * 4), 15), 28), 28)) * B[v_j]

        @R.function
        def main(A: R.Tensor((1, "n", 4096), dtype="float16"), 
                B: R.Tensor((T.int64(4096), T.int64(11008)), "int8"),
                B_SCL: R.Tensor((T.int64(22016),), "float16")):
            cls = _mod
            with R.dataflow():
                b_dec = R.call_tir(cls.decode, (B, B_SCL), out_sinfo=R.Tensor((4096, 22016), dtype="float16"))
                x = R.matmul(A, b_dec)
                R.output(x)
            return x

    def _arg_info_provider(n):
        return [
            ms.arg_info.TensorInfo("float16", [1, n, 4096]),
            ms.arg_info.TensorInfo("int8", [4096, 11008]),
            ms.arg_info.TensorInfo("float16", [22016]),
        ]

    return _mod, _arg_info_provider 


def linear_1_gen(_N, _K): 
    @I.ir_module
    class _mod:
        @R.function
        def main(A: R.Tensor((1, "n", 4096), dtype="float16"), 
                 B: R.Tensor((4096, 22016), dtype="float16")):
            with R.dataflow():
                x = R.matmul(A, B)
                R.output(x)
            return x

    def _arg_info_provider(n):
        return [
            ms.arg_info.TensorInfo("float16", [1, n, 4096]),
            ms.arg_info.TensorInfo("float16", [4096, 22016]),
        ]
    
    return _mod, _arg_info_provider


def linear_1_static_gen(_N, _K): 
    @I.ir_module
    class _mod:
        @R.function
        def main(A: R.Tensor((1, 2048, 4096), dtype="float16"), 
                 B: R.Tensor((4096, 22016), dtype="float16")):
            with R.dataflow():
                x = R.matmul(A, B)
                R.output(x)
            return x

    def _arg_info_provider(n):
        return [
            ms.arg_info.TensorInfo("float16", [1, 2048, 4096]),
            ms.arg_info.TensorInfo("float16", [4096, 22016]),
        ]
    
    return _mod, _arg_info_provider

num = None

@tvm._ffi.register_func("blabla.callback")
def tvm_callback_cuda_compile(mod, name):
    global num
    if num is not None:
        if num in [25, 26]:
        # if "T.bitwise_xor" in mod.__str__():
            print("HERE ", name, num)
        num = num + 1
        # print(f"{num} {id(mod)}")
    return None


def compile_relax(mod, with_dlight=False, with_cublas=False, with_cutlass=False, db=None):
    if with_cublas:
        mod = partition_for_cublas(mod)
        mod = relax.transform.RunCodegen()(mod)

    if with_cutlass:
        mod = partition_for_cutlass(mod)
        mod = relax.transform.RunCodegen()(mod)        

    mod = relax.pipeline.get_pipeline()(mod)
    if db is not None:
        with TARGET, db:
            mod = relax.transform.MetaScheduleApplyDatabase()(mod)

    if with_dlight:
        # "tir.use_async_copy': True
        with TARGET:
            mod = dl.ApplyDefaultSchedule(dl.gpu.Matmul())(mod)
            mod = dl.ApplyDefaultSchedule(dl.gpu.GEMV())(mod)
            mod = dl.ApplyDefaultSchedule(dl.gpu.Fallback())(mod)
    # global num
    # num = 0
    with tvm.transform.PassContext(opt_level=3, config={"relax.backend.use_cuda_graph": True, "tir.use_async_copy": True}):
        ex = relax.build(mod, TARGET)
    
    # exit()

    return ex



def main_cutlass(use_ft=False):
    N, K = 4096, 22016
    M = 2048
    
    # mod, arg_info_provider = linear_1_ft_gen(N, K) if use_ft else linear_1_gen(N, K)
    mod, arg_info_provider = linear_1_static_gen(N, K)
    # mod, arg_info_provider = linear_1_gen(N, K)

    ex = compile_relax(mod, with_cutlass=True)
    
    # cuda_dump(ex.mod.imported_modules[0])

    vm = relax.VirtualMachine(ex, DEV)

    args_info = arg_info_provider(M)
    args = [make_arg(info) for info in args_info]

    for _ in range(20):
        score = vm.time_evaluator("main", dev=DEV, number=100, repeat=3, min_repeat_ms=1000)(*args).mean
        score_us = int(float(score)*1e6)
        thrp_tmacps = M * N * K / score_us / 1e6
        print(f"{M} TIME_US {score_us} THRP {thrp_tmacps}")


def main_cublas():
    N, K = 22016, 4096
    M = 512
    mod, arg_info_provider = linear_1_gen(N, K)

    ex = compile_relax(mod, with_cublas=True)

    vm = relax.VirtualMachine(ex, DEV)

    args_info = arg_info_provider(M)
    args = [make_arg(info) for info in args_info]

    # warm up
    score = vm.time_evaluator("main", dev=DEV, number=100, repeat=1, min_repeat_ms=100)(*args).mean
    
    for _ in range(20):
        score = vm.time_evaluator("main", dev=DEV, number=100, repeat=1, min_repeat_ms=100)(*args).mean
        score_us = int(float(score)*1e6)
        thrp_tmacps = M * N * K / score_us / 1e6
        print(f"{M} TIME_US {score_us} THRP {thrp_tmacps}")


def main_dlight():
    N, K = 22016, 4096
    M = 2048
    mod, arg_info_provider = linear_1_gen(N, K)

    ex = compile_relax(mod, with_dlight=True)
    
    # dump in file format    
    ex.mod.imported_modules[0].save("tmp/orig.ll")
    ex.mod.imported_modules[0].imported_modules[0].save("tmp/orig.cubin")

    cuda_dump(ex.mod.imported_modules[0])

    vm = relax.VirtualMachine(ex, DEV)

    args_info = arg_info_provider(M)
    args = [make_arg(info) for info in args_info]

    for _ in range(10):
        score = vm.time_evaluator("main", dev=DEV, number=500, repeat=1, min_repeat_ms=1)(*args).mean
        score_us = int(float(score)*1e6)
        thrp_tmacps = M * N * K / score_us / 1e6
        print(f"{M} TIME_US {score_us} THRP {thrp_tmacps}")


def roofline():
    N, K = 22016, 4096
    mod, arg_info_provider = linear_1_gen(N, K)

    # ex = compile_relax(mod, with_cublas=True)
    # ex = compile_relax(mod, with_cutlass=True)
    ex = compile_relax(mod, with_dlight=True)

    vm = relax.VirtualMachine(ex, DEV)

    # for M in range(128, 4096, 128):
    for M in range(32, 4096+1, 32):
        args_info = arg_info_provider(M)
        args = [make_arg(info) for info in args_info]

        score = vm.time_evaluator("main", dev=DEV, number=10, repeat=1, min_repeat_ms=3000)(*args).mean
        score_us = int(float(score)*1e6)
        thrp_tmacps = M * N * K / score_us / 1e6
        print(f"{M} TIME_US {score_us} THRP {thrp_tmacps}")


# @tvm._ffi.register_func
def tvm_callback_cuda_compile(code, target):  # pylint: disable=unused-argument
    """use nvcc to generate fatbin code for better optimization"""
    ptx = nvcc.compile_cuda(code, target_format="fatbin")
    return ptx


def just_compile():
    load_cubin_rt_mod = tvm.get_global_func("runtime.module.loadfile_cubin")
    load_ll_rt_mod = tvm.get_global_func("runtime.module.loadfile_ll")
    
    def load(name):
        with open(f"{name}.cu", "r") as f:
            src = f.read()

        cubin = nvcc.compile_cuda(src, target_format="cubin")
        with open(f"{name}.cubin", "wb") as f:
            f.write(cubin)
        
        sass = get_sass(cubin)
        with open(f"{name}.sass", "w") as f:
            f.write(sass)

        rt_mod = load_ll_rt_mod(f"{name}.ll", "ll")
        rt_mod.import_module(load_cubin_rt_mod(f"{name}.cubin", "cubin"))

        return rt_mod
    
    # orig_mod = load("tmp/orig")
    # exp_mod = load("tmp/exp")
    orig_mod = load("tmp/orig_dsp")
    exp_mod = load("tmp/exp_dsp")
        
    N, K = 22016, 4096
    # M = 512
    M = 2048
    args_info = [
            ms.arg_info.TensorInfo("float16", [1, M, K]),
            ms.arg_info.TensorInfo("float16", [K, N]),
            ms.arg_info.TensorInfo("float16", [1, M, N]),
        ]
    args = [make_arg(info) for info in args_info]

    orig_mod["matmul"](*args)
    ref_out = args[-1].numpy()

    exp_mod["matmul"](*args)
    exp_out = args[-1].numpy()

    # ref via cublass
    mod, _ = linear_1_gen(N, K)
    ex = compile_relax(mod, with_cublas=True)
    vm_cublas = relax.VirtualMachine(ex, DEV)
    ref_out_2 = vm_cublas["main"](args[0], args[1]).numpy()

    mod, _ = linear_1_gen(N, K)
    ex = compile_relax(mod, with_dlight=True)
    vm_dlight = relax.VirtualMachine(ex, DEV)
    ref_out_3 = vm_dlight["main"](args[0], args[1]).numpy()
    
    assert np.allclose(ref_out, ref_out_3)
    assert np.allclose(ref_out, ref_out_2)
    assert np.allclose(ref_out, exp_out)

    for _ in range(10):
        # score = exp_mod.time_evaluator("matmul", dev=DEV, number=500, repeat=1, min_repeat_ms=500)(*args).mean
        # score = orig_mod.time_evaluator("matmul", dev=DEV, number=500, repeat=1, min_repeat_ms=500)(*args).mean
        # score = vm_cublas.time_evaluator("main", dev=DEV, number=500, repeat=1, min_repeat_ms=500)(args[0], args[1]).mean
        score = vm_dlight.time_evaluator("main", dev=DEV, number=500, repeat=1, min_repeat_ms=1)(args[0], args[1]).mean
        score_us = int(float(score)*1e6)
        thrp_tmacps = M * N * K / score_us / 1e6
        print(f"{M} TIME_US {score_us} THRP {thrp_tmacps}")

    # for M in range(128, 4096, 128): 
    #     args_info = [
    #         ms.arg_info.TensorInfo("float16", [1, M, K]),
    #         ms.arg_info.TensorInfo("float16", [K, N]),
    #         ms.arg_info.TensorInfo("float16", [1, M, N]),
    #     ]
    #     args = [make_arg(info) for info in args_info]
    #     # score = exp_mod.time_evaluator("matmul", dev=DEV, number=500, repeat=1, min_repeat_ms=500)(*args).mean
    #     score = vm_cublas.time_evaluator("main", dev=DEV, number=500, repeat=1, min_repeat_ms=500)(args[0], args[1]).mean
    #     # score = vm_dlight.time_evaluator("main", dev=DEV, number=500, repeat=1, min_repeat_ms=500)(args[0], args[1]).mean
    #     score_us = int(float(score)*1e6)
    #     thrp_tmacps = M * N * K / score_us / 1e6
    #     print(f"{M} TIME_US {score_us} THRP {thrp_tmacps}")




if __name__ == "__main__":
    # main_cutlass()
    # main_cutlass(use_ft=True)
    # main_cublas()
    main_dlight()
    # roofline()

    # just_compile()

