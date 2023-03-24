import numpy as np

import tvm.testing
from tvm import meta_schedule as ms
from tvm import relay
from tvm.tir.tensor_intrin.hexagon import (
    VSHUFF_TRANSFORM_4x4x32_INTRIN,
    VMEM_COPY_STRIDED_DST_32_INTRIN_MAN,
    VMEM_COPY_STRIDED_SRC_32_INTRIN_MAN,
    VMEM_COPY_STRIDED_DST_8x8x32_INTRIN_MAN,
    VRMPY_u8u8i32_INTRIN,
)

from tvm.contrib.hexagon.meta_schedule import get_hexagon_local_builder, get_hexagon_rpc_runner
from tvm import te

hex_target = tvm.target.hexagon("v68")
TARGET_HEX = tvm.target.Target(hex_target, host=hex_target)

BENCHMARK_TIME = True


def ref_transform_to(src: np.ndarray, layout="NCHW8h8w32c"):
    assert len(src.shape) == 4
    assert layout in ("NCHW8h8w32c", "NHWC8h8w32c")
    N, H, W, C = src.shape
    assert H % 8 == 0 and W % 8 == 0 and C % 32 == 0
    dst = np.reshape(src, newshape=[N, H//8, 8, W//8, 8, C//32, 32])
    if layout == "NCHW8h8w32c":
        dst = np.transpose(dst, axes=[0, 5, 1, 3, 2, 4, 6])
    else:  # NHWC8h8w32c
        dst = np.transpose(dst, axes=[0, 1, 3, 5, 2, 4, 6])
    return dst


def ref_transform_from(src: np.ndarray, layout):
    assert len(src.shape) == 7
    assert layout in ("NCHW8h8w32c", "NHWC8h8w32c")
    if layout == "NCHW8h8w32c":
        N, C, H, W, h, w, c,  = src.shape
        assert h == 8 and w == 8 and c == 32
        dst = np.transpose(src, axes=[0, 2, 4, 3, 5, 1, 6])
        dst = np.reshape(dst, newshape=[N, H*h, W*w, C*c])

    else:  # NHWC8h8w32c
        N, H, W, C, h, w, c,  = src.shape
        assert h == 8 and w == 8 and c == 32
        dst = np.transpose(src, axes=[0, 1, 4, 2, 5, 3, 6])
        dst = np.reshape(dst, newshape=[N, H*h, W*w, C*c])

    return dst


def relay_layout_transform_to(shape, src_layout="NHWC", dst_layout="NCHW8h8w32c"):
    assert src_layout == "NHWC"
    x = relay.var("X", shape=shape, dtype="uint8")
    op = relay.layout_transform(x, src_layout=src_layout, dst_layout=dst_layout)
    mod = tvm.IRModule.from_expr(op)
    opt_mod, _ = relay.optimize(mod, params={}, target=TARGET_HEX)
    prim_func = relay.backend.te_compiler.lower_to_primfunc(opt_mod["main"].body.op, TARGET_HEX)

    return prim_func


def relay_layout_transform_from(shape, src_layout="NCHW8h8w32c", dst_layout="NHWC"):
    assert src_layout in ("NCHW8h8w32c", "NHWC8h8w32c")
    N, H, W, C = shape
    if src_layout == "NCHW8h8w32c":
        crout_shape = [N, C//32, H//8, W//8, 8, 8, 32]
    else:  # NHWC8h8w32c
        crout_shape = [N, H//8, W//8, C//32, 8, 8, 32]

    x = relay.var("X", shape=crout_shape, dtype="uint8")
    op = relay.layout_transform(x, src_layout=src_layout, dst_layout=dst_layout)
    mod = tvm.IRModule.from_expr(op)
    opt_mod, _ = relay.optimize(mod, params={}, target=TARGET_HEX)
    prim_func = relay.backend.te_compiler.lower_to_primfunc(opt_mod["main"].body.op, TARGET_HEX)

    return prim_func


shape = tvm.testing.parameter(
    [1, 56, 56, 256],
    [1, 8, 8, 64],
    [1, 112, 112, 64],
    [1, 56, 56, 64]
)
layout = tvm.testing.parameter(
    "NCHW8h8w32c",
    "NHWC8h8w32c"
)


@tvm.testing.requires_hexagon
def test_transform_to_croutonized_codegen(hexagon_session, shape, layout):
    np.random.seed(100)

    func = relay_layout_transform_to(shape, src_layout="NHWC", dst_layout=layout)

    args_info = ms.arg_info.ArgInfo.from_prim_func(func)
    np_input = [np.random.default_rng().integers(0, 37, list(int(d) for d in info.shape), dtype="uint8") for info in args_info]
    dev_input = [tvm.runtime.ndarray.array(arr, device=hexagon_session.device, mem_scope="global") for arr in np_input]

    sch = tvm.tir.Schedule(func)
    b1 = sch.get_block(name="T_layout_trans", func_name="main")
    n, c_o, h_o, w_o, h_i, w_i, c_i = sch.get_loops(block=b1)

    w_i_1, w_i_2 = sch.split(loop=w_i, factors=[None, 4])
    c_o_1, c_o_2 = sch.split(loop=c_o, factors=[None, 4])
    sch.reorder(n, c_o_1, h_o, w_o, h_i, w_i_1, c_o_2, w_i_2, c_i)
    sch.tensorize(c_o_2, VSHUFF_TRANSFORM_4x4x32_INTRIN)

    sch.parallel(sch.fuse(n, c_o_1, h_o, w_o))

    # Build
    dev_lib = tvm.build(sch.mod["main"], target=TARGET_HEX, name="main")
    dev_rt_mod = hexagon_session.load_module(dev_lib)
    dev_rt_mod(*dev_input)

    assert np.array_equal(dev_input[1].numpy(), ref_transform_to(np_input[0], layout))

    if BENCHMARK_TIME:
        timer = dev_rt_mod.time_evaluator("__tvm_main__", hexagon_session.device, number=100, repeat=1)
        dur_sec = timer(*dev_input).mean
        print(f"{dur_sec*1000:0.3f} ms")

    print(dev_lib.get_source("asm"))


@tvm.testing.requires_hexagon
def test_transform_to_croutonized(hexagon_session, shape, layout):
    np.random.seed(100)

    func = relay_layout_transform_to(shape, src_layout="NHWC", dst_layout=layout)

    args_info = ms.arg_info.ArgInfo.from_prim_func(func)
    np_input = [np.random.default_rng().integers(0, 37, list(int(d) for d in info.shape), dtype="uint8") for info in args_info]
    dev_input = [tvm.runtime.ndarray.array(arr, device=hexagon_session.device, mem_scope="global") for arr in np_input]

    sch = tvm.tir.Schedule(func)
    b1 = sch.get_block(name="T_layout_trans", func_name="main")
    n, c_o, h_o, w_o, h_i, w_i, c_i = sch.get_loops(block=b1)

    w_i_1, w_i_2 = sch.split(loop=w_i, factors=[None, 4])
    sch.tensorize(w_i_2, VMEM_COPY_STRIDED_SRC_32_INTRIN_MAN)

    sch.parallel(sch.fuse(n, c_o, h_o, w_o))

    # Build
    dev_lib = tvm.build(sch.mod["main"], target=TARGET_HEX, name="main")
    dev_rt_mod = hexagon_session.load_module(dev_lib)
    dev_rt_mod(*dev_input)

    assert np.array_equal(dev_input[1].numpy(), ref_transform_to(np_input[0], layout))

    if BENCHMARK_TIME:
        timer = dev_rt_mod.time_evaluator("__tvm_main__", hexagon_session.device, number=100, repeat=1)
        dur_sec = timer(*dev_input).mean
        print(f"{dur_sec*1000:0.3f} ms")


@tvm.testing.requires_hexagon
def test_transform_from_croutonized(hexagon_session, shape, layout):
    np.random.seed(100)

    func = relay_layout_transform_from(shape, src_layout=layout, dst_layout="NHWC")

    args_info = ms.arg_info.ArgInfo.from_prim_func(func)
    np_input = [np.random.default_rng().integers(0, 37, list(int(d) for d in info.shape), dtype="uint8") for info in args_info]
    dev_input = [tvm.runtime.ndarray.array(arr, device=hexagon_session.device, mem_scope="global") for arr in np_input]

    sch = tvm.tir.Schedule(func)
    b1 = sch.get_block(name="T_layout_trans", func_name="main")
    n, h, w, c = sch.get_loops(block=b1)

    c_1, c_2 = sch.split(loop=c, factors=[None, 32])
    w_1, w_2 = sch.split(loop=w, factors=[None, 4])
    sch.reorder(w_1, c_1, w_2, c_2)
    sch.tensorize(w_2, VMEM_COPY_STRIDED_DST_32_INTRIN_MAN)

    sch.parallel(sch.fuse(n, h, w_1))

    # Build
    dev_lib = tvm.build(sch.mod["main"], target=TARGET_HEX, name="main")
    dev_rt_mod = hexagon_session.load_module(dev_lib)
    dev_rt_mod(*dev_input)

    assert np.array_equal(dev_input[1].numpy(), ref_transform_from(np_input[0], layout))

    if BENCHMARK_TIME:
        timer = dev_rt_mod.time_evaluator("__tvm_main__", hexagon_session.device, number=100, repeat=1)
        dur_sec = timer(*dev_input).mean
        print(f"{dur_sec*1000:0.3f} ms")

    print(dev_lib.get_source("asm"))


@tvm.testing.requires_hexagon
def test_transform_from_croutonized_blocked(hexagon_session, shape, layout):
    np.random.seed(100)

    func = relay_layout_transform_from(shape, src_layout=layout, dst_layout="NHWC")

    args_info = ms.arg_info.ArgInfo.from_prim_func(func)
    np_input = [np.random.default_rng().integers(0, 37, list(int(d) for d in info.shape), dtype="uint8") for info in args_info]
    dev_input = [tvm.runtime.ndarray.array(arr, device=hexagon_session.device, mem_scope="global") for arr in np_input]

    sch = tvm.tir.Schedule(func)
    b1 = sch.get_block(name="T_layout_trans", func_name="main")
    n, h, w, c = sch.get_loops(block=b1)

    c_1, c_2 = sch.split(loop=c, factors=[None, 32])
    w_1, w_2 = sch.split(loop=w, factors=[None, 8])
    h_1, h_2 = sch.split(loop=h, factors=[None, 8])
    sch.reorder(h_1, w_1, c_1, h_2, w_2, c_2)
    sch.tensorize(h_2, VMEM_COPY_STRIDED_DST_8x8x32_INTRIN_MAN)

    sch.parallel(sch.fuse(n, h_1, w_1))

    # Build
    dev_lib = tvm.build(sch.mod["main"], target=TARGET_HEX, name="main")
    dev_rt_mod = hexagon_session.load_module(dev_lib)
    dev_rt_mod(*dev_input)

    assert np.array_equal(dev_input[1].numpy(), ref_transform_from(np_input[0], layout))

    if BENCHMARK_TIME:
        timer = dev_rt_mod.time_evaluator("__tvm_main__", hexagon_session.device, number=100, repeat=1)
        dur_sec = timer(*dev_input).mean
        print(f"{dur_sec*1000:0.3f} ms")

    print(dev_lib.get_source("asm"))


@tvm.testing.requires_hexagon
def test_tuning_transform_with_intrinsic(hexagon_launcher):
    np.random.seed(100)

    func = relay_layout_transform_from(shape=[1, 56, 56, 256], src_layout="NCHW8h8w32c", dst_layout="NHWC")

    space = ms.space_generator.PostOrderApply(
        sch_rules=[
            ms.schedule_rule.ApplyCustomRule(),
            ms.schedule_rule.AutoInline(
                into_producer=False,
                into_consumer=True,
                inline_const_tensor=True,
                disallow_if_then_else=True,
                require_injective=True,
                require_ordered=True,
                disallow_op=["tir.exp"],
            ),
            ms.schedule_rule.MultiLevelTilingWithIntrin(
                VMEM_COPY_STRIDED_DST_8x8x32_INTRIN_MAN,
                structure="SSS",
                tile_binds=None,
                max_innermost_factor=64,  # ???
                vector_load_lens=None,
                reuse_read=None,
                reuse_write=None,
            ),
            ms.schedule_rule.ParallelizeVectorizeUnroll(
                max_jobs_per_core=16,
                max_vectorize_extent=128,
                unroll_max_steps=[0, 16, 64, 512],
                unroll_explicit=True,
            ),
        ],
        postprocs=[
            ms.postproc.RewriteParallelVectorizeUnroll(),
            ms.postproc.RewriteReductionBlock(),
            ms.postproc.RewriteTensorize(vectorize_init_loop=True),
        ],
        mutator_probs={},
    )

    database = ms.tir_integration.tune_tir(
        mod=func,
        target=TARGET_HEX,
        work_dir=f"tune_transform_tmp",
        max_trials_global=512,
        num_trials_per_iter=32,
        cost_model="random",  # "xgb"
        space=space,
        strategy="replay-trace",
        builder=get_hexagon_local_builder(),
        runner=get_hexagon_rpc_runner(hexagon_launcher, number=10),
    )


@tvm.testing.requires_hexagon
def test_tuning_transform_to_with_intrinsic(hexagon_launcher):
    np.random.seed(100)
    layout = "NCHW8h8w32c"
    func = relay_layout_transform_to(shape=[1, 56, 56, 256], src_layout="NHWC", dst_layout=layout)

    space = ms.space_generator.PostOrderApply(
        sch_rules=[
            ms.schedule_rule.ApplyCustomRule(),
            ms.schedule_rule.AutoInline(
                into_producer=False,
                into_consumer=True,
                inline_const_tensor=True,
                disallow_if_then_else=True,
                require_injective=True,
                require_ordered=True,
                disallow_op=["tir.exp"],
            ),
            ms.schedule_rule.MultiLevelTilingWithIntrin(
                VSHUFF_TRANSFORM_4x4x32_INTRIN,
                structure="SSS",
                tile_binds=None,
                max_innermost_factor=64,  # ???
                vector_load_lens=None,
                reuse_read=None,
                reuse_write=None,
            ),
            ms.schedule_rule.ParallelizeVectorizeUnroll(
                max_jobs_per_core=16,
                max_vectorize_extent=128,
                unroll_max_steps=[0, 16, 64, 512],
                unroll_explicit=True,
            ),
        ],
        postprocs=[
            ms.postproc.RewriteParallelVectorizeUnroll(),
            ms.postproc.RewriteReductionBlock(),
            ms.postproc.RewriteTensorize(vectorize_init_loop=True),
        ],
        mutator_probs={},
    )

    database = ms.tir_integration.tune_tir(
        mod=func,
        target=TARGET_HEX,
        work_dir=f"tune_transform_tmp",
        max_trials_global=512,
        num_trials_per_iter=32,
        # cost_model="random",
        cost_model="xgb",
        space=space,
        strategy="replay-trace",
        builder=get_hexagon_local_builder(),
        runner=get_hexagon_rpc_runner(hexagon_launcher, number=10),
    )


def test_check_tuning_result(hexagon_session):
    np.random.seed(100)

    layout = "NCHW8h8w32c"
    func = relay_layout_transform_from(shape=[1, 56, 56, 256], src_layout=layout, dst_layout="NHWC")

    args_info = ms.arg_info.ArgInfo.from_prim_func(func)
    np_input = [np.random.default_rng().integers(0, 37, list(int(d) for d in info.shape), dtype="uint8") for info in args_info]
    dev_input = [tvm.runtime.ndarray.array(arr, device=hexagon_session.device, mem_scope="global") for arr in np_input]

    # Manual Schedule
    # sch = tvm.tir.Schedule(func)
    # b1 = sch.get_block(name="T_layout_trans", func_name="main")
    # n, h, w, c = sch.get_loops(block=b1)
    #
    # c_1, c_2 = sch.split(loop=c, factors=[None, 32])
    # w_1, w_2 = sch.split(loop=w, factors=[None, 8])
    # h_1, h_2 = sch.split(loop=h, factors=[None, 8])
    # sch.reorder(h_1, w_1, c_1, h_2, w_2, c_2)
    # sch.tensorize(h_2, VMEM_COPY_STRIDED_DST_8x8x32_INTRIN_MAN)
    #
    # sch.parallel(sch.fuse(n, h_1, w_1))

    # From DB
    db = ms.database.JSONDatabase(work_dir=f"tune_transform_tmp")
    sch = ms.tir_integration.compile_tir(db, func, TARGET_HEX)

    # Build
    dev_lib = tvm.build(sch.mod["main"], target=TARGET_HEX, name="main")
    dev_rt_mod = hexagon_session.load_module(dev_lib)
    dev_rt_mod(*dev_input)

    assert np.array_equal(dev_input[1].numpy(), ref_transform_from(np_input[0], layout))

    if BENCHMARK_TIME:
        timer = dev_rt_mod.time_evaluator("__tvm_main__", hexagon_session.device, number=100, repeat=1)
        dur_sec = timer(*dev_input).mean
        print(f"{dur_sec*1000:0.3f} ms")

    print(dev_lib.get_source("asm"))


@tvm.testing.requires_hexagon
def test_tuning_dense_with_intrinsic(hexagon_launcher):
    np.random.seed(100)

    def dense_compute(m, n, k):
        """dense compute"""
        X = te.placeholder((m, k), name="X", dtype="uint8")
        packed_width = te.placeholder((n // 32, k // 4, 32, 4), name="packed_width", dtype="uint8")

        axis_k = te.reduce_axis((0, k), name="k")
        out = te.compute(
            (m, n),
            lambda i, j: te.sum(
                X[i, axis_k].astype("int32")
                * packed_width[
                    tvm.tir.indexdiv(j, 32), tvm.tir.indexdiv(axis_k, 4), j % 32, axis_k % 4
                ].astype("int32"),
                axis=axis_k,
                ),
            name="compute",
        )
        return te.create_prim_func([X, packed_width, out])

    func = dense_compute(56*56, 256, 64)

    space = ms.space_generator.PostOrderApply(
        sch_rules=[
            ms.schedule_rule.ApplyCustomRule(),
            # ms.schedule_rule.AutoInline(
            #     into_producer=False,
            #     into_consumer=True,
            #     inline_const_tensor=True,
            #     disallow_if_then_else=True,
            #     require_injective=True,
            #     require_ordered=True,
            #     disallow_op=["tir.exp"],
            # ),
            ms.schedule_rule.MultiLevelTilingWithIntrin(
                VRMPY_u8u8i32_INTRIN,
                structure="SRSRS",
                tile_binds=None,
                max_innermost_factor=64,
                vector_load_lens=None,
                reuse_read=None,
                reuse_write=ms.schedule_rule.ReuseType(
                    req="may",
                    levels=[1, 2],
                    scope="global",
                ),
            ),            # ms.schedule_rule.ParallelizeVectorizeUnroll(
            #     max_jobs_per_core=16,
            #     max_vectorize_extent=128,
            #     unroll_max_steps=[0, 16, 64, 512],
            #     unroll_explicit=True,
            # ),
        ],
        postprocs=[
            ms.postproc.RewriteParallelVectorizeUnroll(),
            ms.postproc.RewriteReductionBlock(),
            ms.postproc.RewriteTensorize(vectorize_init_loop=True),
        ],
        mutator_probs={},
    )

    database = ms.tir_integration.tune_tir(
        mod=func,
        target=TARGET_HEX,
        work_dir=f"tune_transform_tmp",
        max_trials_global=1,
        num_trials_per_iter=1,
        space=space,
        strategy="replay-trace",
        builder=get_hexagon_local_builder(),
        runner=get_hexagon_rpc_runner(hexagon_launcher, number=10),
    )
