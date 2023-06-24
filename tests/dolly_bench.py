import numpy as np

import tvm
from tvm.target import Target
from tvm import meta_schedule as ms
from tvm.script import tir as T

import tvm.tir.tensor_intrin.cuda

from dolly_tune import get_matmul_int4, mutate_to_dyn_m, get_matmul_int4_dyn_m, apply_trace_int_v1, apply_trace_int_v1_static


def generate_arg(info: ms.arg_info.ArgInfo, dev):
    if info.dtype == "float16":
        arr_np = np.random.uniform(-1, 1, size=info.shape).astype('float16')
    elif info.dtype == "int32":
        arr_np = np.random.randint(0, 16, size=info.shape).astype('int32')
    else:
        assert False, "Unimplemented"

    return tvm.nd.array(arr_np, device=dev)


def main():
    M, N, K, G = 32, 5120,  20480, 160
    work_dir = "dolly_tune_all_1"
    
    dev = tvm.cuda(0)
    target = Target("nvidia/nvidia-a100")
    db = ms.database.JSONDatabase(work_dir=work_dir, allow_missing=False)

    static_m_mod = get_matmul_int4(M, N, K, G)
    static_m_mod = ms.tir_integration._normalize_mod(static_m_mod)
    sch = db.query(static_m_mod, target)

    with target:
        lib = tvm.build(sch.mod["main"])

    args_info = ms.arg_info.ArgInfo.from_prim_func(static_m_mod["main"])
    args = [generate_arg(info, dev) for info in args_info]

    dur_us = lib.time_evaluator(lib.entry_name, dev=dev, number=2000, repeat=1)(*args).mean * 1e6
    print(f"Static Duration {dur_us} us")


    # dyn M version
    top_k = 40
    assert db.has_workload(static_m_mod)
    workload = db.commit_workload(static_m_mod)
    top_k_recs = db.get_top_k(workload, top_k=top_k)
    for top_idx, rec in enumerate(top_k_recs):
        dyn_m_mod = get_matmul_int4_dyn_m(N, K, G)
        dyn_m_trace = mutate_to_dyn_m(rec.trace) 
        dyn_m_sch = tvm.tir.Schedule(dyn_m_mod)
        dyn_m_trace.apply_to_schedule(dyn_m_sch, remove_postproc=False)
        
        with target:
            dyn_m_lib = tvm.build(dyn_m_sch.mod["main"])

        for M in range(1, 2):
            args_info = [
                ms.arg_info.TensorInfo(shape=[M, K], dtype="float16"),
                ms.arg_info.TensorInfo(shape=[K//8, N], dtype="int32"),
                ms.arg_info.TensorInfo(shape=[G, N], dtype="float16"),
                ms.arg_info.TensorInfo(shape=[G, N//8], dtype="int32"),
                ms.arg_info.TensorInfo(shape=[M, N], dtype="float16"),
            ]
            args = [generate_arg(info, dev) for info in args_info]

            dur_us = dyn_m_lib.time_evaluator(dyn_m_lib.entry_name, dev=dev, number=2000, repeat=1)(*args).mean * 1e6
            print(f"[TOP:{top_idx}] [M:{M}] Dyn_M Duration {dur_us} us  (declared {int(rec.run_secs[0]) * 1e6} us)")


    ######### v1 schedule dyn m
    # dyn_m_v1_sch = tvm.tir.Schedule(dyn_m_mod)
    # apply_trace_int_v1(dyn_m_v1_sch)
    # with target:
    #     dyn_m_v1_lib = tvm.build(dyn_m_v1_sch.mod["main"])

    # for M in range(1, 65, 4):
    #     args_info = [
    #         ms.arg_info.TensorInfo(shape=[M, K], dtype="float16"),
    #         ms.arg_info.TensorInfo(shape=[K//8, N], dtype="int32"),
    #         ms.arg_info.TensorInfo(shape=[G, N], dtype="float16"),
    #         ms.arg_info.TensorInfo(shape=[G, N//8], dtype="int32"),
    #         ms.arg_info.TensorInfo(shape=[M, N], dtype="float16"),
    #     ]
    #     args = [generate_arg(info, dev) for info in args_info]

    #     dur_us = dyn_m_v1_lib.time_evaluator(dyn_m_v1_lib.entry_name, dev=dev, number=2000, repeat=1)(*args).mean * 1e6
    #     print(f"[{M}] V1 Dyn_M Duration {dur_us} us")
    
    ########## v1 schedule static
    # M = 32
    # static_m_v1_sch = tvm.tir.Schedule(static_m_mod)
    # apply_trace_int_v1_static(static_m_v1_sch)

    # with target:
    #     static_m_v1_lib = tvm.build(static_m_v1_sch.mod["main"])

    # args_info = ms.arg_info.ArgInfo.from_prim_func(static_m_mod["main"])
    # args = [generate_arg(info, dev) for info in args_info]

    # dur_us = static_m_v1_lib.time_evaluator(static_m_v1_lib.entry_name, dev=dev, number=2000, repeat=1)(*args).mean * 1e6
    # print(f"[{M}] V1 Static_M Duration {dur_us} us")


def analize():
    # M, N, K, G = 32, 5120,  20480, 160
    # M, N, K, G = 32, 5120,  5120,  40
    # M, N, K, G = 32, 20480, 5120,  40

    # M, N, K, G = 1, 5120,  20480, 160 
    # work_dir = "dolly_tune_all_2"

    # M, N, K, G = 1, 20480, 5120, 40 
    # work_dir = "dolly_tune_all_3"

    # M, N, K, G = 1, 15360, 5120,  40 
    # work_dir = "dolly_tune_all_4"

    M, N, K, G = 1, 5120, 5120,  40 
    work_dir = "dolly_tune_all_5"

    dev = tvm.cuda(0)
    target = Target("nvidia/nvidia-a100")
    db = ms.database.JSONDatabase(work_dir=work_dir, allow_missing=False)

    find_best_dyn_m(M, N, K, G, db=db, top_k=1, target=target, dev=dev)


def find_best_dyn_m(M, N, K, G, db, top_k, target, dev) -> tvm.tir.schedule.Trace:
    static_m_mod = get_matmul_int4(M, N, K, G)
    static_m_mod = ms.tir_integration._normalize_mod(static_m_mod)

    assert db.has_workload(static_m_mod)
    workload = db.commit_workload(static_m_mod)

    best_score_us = float("inf")
    best_reported_static_score_us = None
    best_topk_pos = None
    best_trace = None
    
    top_k_recs = db.get_top_k(workload, top_k=top_k)
    
    for top_idx, rec in enumerate(top_k_recs):
        # dyn_m_mod = get_matmul_int4_dyn_m(N, K, G)
        # dyn_m_trace = mutate_to_dyn_m(rec.trace)
        dyn_m_mod = get_matmul_int4(M, N, K, G)
        dyn_m_trace = rec.trace 

        dyn_m_sch = tvm.tir.Schedule(dyn_m_mod)
        dyn_m_trace.apply_to_schedule(dyn_m_sch, remove_postproc=False)
        
        with target:
            dyn_m_lib = tvm.build(dyn_m_sch.mod["main"])

        print(dyn_m_sch.mod["main"])

        # M_ = 1
        M_ = M
        args_info = [
            ms.arg_info.TensorInfo(shape=[M_, K], dtype="float16"),
            ms.arg_info.TensorInfo(shape=[K//8, N], dtype="int32"),
            ms.arg_info.TensorInfo(shape=[G, N], dtype="float16"),
            ms.arg_info.TensorInfo(shape=[G, N//8], dtype="int32"),
            ms.arg_info.TensorInfo(shape=[M_, N], dtype="float16"),
        ]
        args = [generate_arg(info, dev) for info in args_info]

        score_us = dyn_m_lib.time_evaluator(dyn_m_lib.entry_name, dev=dev, number=2000, repeat=1)(*args).mean * 1e6
        print(f"[TOP:{top_idx}] [M:{M}] Dyn_M Duration {score_us} us  (declared {float(rec.run_secs[0]) * 1e6} us)")
        
        if score_us < best_score_us:
            best_score_us = score_us
            best_reported_static_score_us = float(rec.run_secs[0]) * 1e6
            best_trace = dyn_m_trace
            best_topk_pos = top_idx

    print(f"Best trace found in position {best_topk_pos}")
    print(f"      dyn_m_score  : {best_score_us} us")
    print(f"     static_score  : {best_reported_static_score_us} us")
    
    return best_trace


if __name__ == "__main__":
    analize()
    # just_print()
