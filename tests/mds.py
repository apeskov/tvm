import numpy as np
import tvm.meta_schedule as ms

import tvm.tir.tensor_intrin.cuda
from mds_rule import MDS1ScheduleRule
from cutlass import make_arg, cuda_dump, linear_1_gen

from tvm import relax
from tvm.script import tir as T


TARGET = tvm.target.Target("nvidia/nvidia-a100")
DEV = tvm.cuda(0)



# @T.prim_func(private=True)
@T.prim_func
def _simple_mm(var_A: T.handle, B: T.Buffer((T.int64(4096), T.int64(22016)), "float16"), var_matmul: T.handle):
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
    n = T.int64()
    A = T.match_buffer(var_A, (T.int64(1), n, T.int64(4096)), "float16")
    matmul = T.match_buffer(var_matmul, (T.int64(1), n, T.int64(22016)), "float16")
    # with T.block("root"):
    for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(22016), T.int64(4096)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(A[v_i0, v_i1, v_k], B[v_k, v_i2])
            T.writes(matmul[v_i0, v_i1, v_i2])
            with T.init():
                matmul[v_i0, v_i1, v_i2] = T.float16(0)
            matmul[v_i0, v_i1, v_i2] = matmul[v_i0, v_i1, v_i2] + A[v_i0, v_i1, v_k] * B[v_k, v_i2]



def main():
    # M, N, K = 192*16, 16128, 4032
    M, N, K = 512, 22016, 4096
    
    mod, arg_info_provider = linear_1_gen(N, K)
    mod = relax.pipeline.get_pipeline()(mod)
    
    # Find single TIR func in module
    gvs = [gv for gv, f in mod.functions_items() if isinstance(f, tvm.tir.function.PrimFunc)]
    assert len(gvs) == 1
    gv = gvs[0]
    func = mod[gv]

    # Schedule TIR primfunc
    sch = tvm.tir.Schedule(func) 
    mds_rule = MDS1ScheduleRule(decisions={
        "m_pad": 128,
        "m_factors":   [1,2,4],
        "n_factors": [1,1,2,4],
        "k_factors":    [1,2],
    })
    sch = mds_rule.apply(sch, sch.get_block("matmul"))[0]
    mod.update_func(gv, sch.mod["main"])

    # Build RELAX
    with tvm.transform.PassContext(opt_level=3, config={"relax.backend.use_cuda_graph": True, "tir.use_async_copy": True}):
        ex = relax.build(mod, TARGET)

    cuda_dump(ex.mod.imported_modules[0])


    args_info = arg_info_provider(M)
    args = [make_arg(info) for info in args_info]
    
    vm = relax.VirtualMachine(ex, DEV)
    vm.time_evaluator("main", dev=DEV, number=1, repeat=1, min_repeat_ms=1)(*args)

    for _ in range(20):
        score = vm.time_evaluator("main", dev=DEV, number=100, repeat=1, min_repeat_ms=100)(*args).mean
        score_us = int(float(score)*1e6)
        thrp_tmacps = M * N * K / score_us / 1e6
        print(f"{M} TIME_US {score_us} THRP {thrp_tmacps}")



if __name__ == "__main__":
    main()
