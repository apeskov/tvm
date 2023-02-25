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
# pylint: disable=invalid-name,missing-function-docstring
"""Intrinsics for Hexagon tensorization."""
from tvm.script import tir as T
from .. import TensorIntrin


def generate_dma_load_intrin(
    size: int,
    dtype: str,
):
    """Generator of dma_load intrins"""

    @T.prim_func
    def sync_dma_load_desc(a: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (size), dtype, offset_factor=1, scope="global")
        C = T.match_buffer(c, (size), dtype, offset_factor=1, scope="global.vtcm")
        with T.block("root"):
            T.reads(A[0:size])
            T.writes(C[0:size])
            for i in T.serial(size):
                with T.block("load"):
                    vii = T.axis.remap("S", [i])
                    C[vii] = A[vii]

    @T.prim_func
    def sync_dma_load_impl(a: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (size), dtype, offset_factor=1, scope="global")
        C = T.match_buffer(c, (size), dtype, offset_factor=1, scope="global.vtcm")
        with T.block("root"):
            T.reads(A[0:size])
            T.writes(C[0:size])
            T.evaluate(
                T.tvm_call_packed(
                    "device_api.hexagon.dma_copy_dltensor",
                    T.tvm_stack_make_array(
                        T.address_of(C[0], dtype="handle"),
                        T.tvm_stack_make_shape(size, dtype="handle"),
                        0,
                        1,
                        C.dtype,
                        0,
                        dtype="handle",
                    ),
                    T.tvm_stack_make_array(
                        T.address_of(A[0], dtype="handle"),
                        T.tvm_stack_make_shape(size, dtype="handle"),
                        0,
                        1,
                        A.dtype,
                        0,
                        dtype="handle",
                    ),
                    T.cast(size, dtype="int"),
                    False,  # Do not use experimental bypass mode.
                    dtype="int32",
                )
            )

    return sync_dma_load_desc, sync_dma_load_impl


def generate_dot_product_32x4_u8u8i32(mem_scope="global"):
    @T.prim_func
    def dot_product_32x4_u8u8i32_desc(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (4,), "uint8", offset_factor=1, scope=mem_scope)
        B = T.match_buffer(b, (32, 4), "uint8", offset_factor=1, scope=mem_scope)
        C = T.match_buffer(c, (32,), "int32", offset_factor=1, scope=mem_scope)
        with T.block("root"):
            T.reads(C[0:32], A[0:4], B[0:32, 0:4])
            T.writes(C[0:32])
            for i in T.serial(0, 32):
                for k in T.serial(0, 4):
                    with T.block("update"):
                        vi, vk = T.axis.remap("SR", [i, k])
                        C[vi] = C[vi] + T.cast(A[vk], "int32") * T.cast(B[vi, vk], "int32")

    @T.prim_func
    def dot_product_32x4_u8u8i32_vrmpy(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (4,), "uint8", offset_factor=1, scope=mem_scope)
        B = T.match_buffer(b, (32, 4), "uint8", offset_factor=1, scope=mem_scope)
        C = T.match_buffer(c, (32,), "int32", offset_factor=1, scope=mem_scope)
        with T.block("root"):
            T.reads(C[0:32], A[0:4], B[0:32, 0:4])
            T.writes(C[0:32])

            A_u8x4 = A.vload([0], "uint8x4")
            A_i32 = T.reinterpret(A_u8x4, dtype="int32")

            B_i8x128 = B.vload([0, 0], dtype="uint8x128")
            B_i32x32 = T.reinterpret(B_i8x128, dtype="int32x32")

            C[T.ramp(T.int32(0), 1, 32)] = T.call_llvm_pure_intrin(
                T.llvm_lookup_intrinsic_id("llvm.hexagon.V6.vrmpyub.acc.128B"),
                T.uint32(3),
                C[T.ramp(T.int32(0), 1, 32)],
                B_i32x32,
                A_i32,
                dtype="int32x32",
            )

    return dot_product_32x4_u8u8i32_desc, dot_product_32x4_u8u8i32_vrmpy


def generate_dot_product_32x4_u8i8i32(mem_scope="global"):
    @T.prim_func
    def dot_product_32x4_u8i8i32_desc(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (4,), "uint8", offset_factor=1, scope=mem_scope)
        B = T.match_buffer(b, (32, 4), "int8", offset_factor=1, scope=mem_scope)
        C = T.match_buffer(c, (32,), "int32", offset_factor=1, scope=mem_scope)
        with T.block("root"):
            T.reads(C[0:32], A[0:4], B[0:32, 0:4])
            T.writes(C[0:32])
            for i in T.serial(0, 32):
                for k in T.serial(0, 4):
                    with T.block("update"):
                        vi, vk = T.axis.remap("SR", [i, k])
                        C[vi] = C[vi] + T.cast(A[vk], "int32") * T.cast(B[vi, vk], "int32")

    @T.prim_func
    def dot_product_32x4_u8i8i32_vrmpy(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (4,), "uint8", offset_factor=1, scope=mem_scope)
        B = T.match_buffer(b, (32, 4), "int8", offset_factor=1, scope=mem_scope)
        C = T.match_buffer(c, (32,), "int32", offset_factor=1, scope=mem_scope)
        with T.block("root"):
            T.reads(C[0:32], A[0:4], B[0:32, 0:4])
            T.writes(C[0:32])

            A_u8x4 = A.vload([0], "uint8x4")
            A_i32 = T.reinterpret(A_u8x4, dtype="int32")

            B_i8x128 = B.vload([0, 0], dtype="int8x128")
            B_i32x32 = T.reinterpret(B_i8x128, dtype="int32x32")

            C[T.ramp(T.int32(0), 1, 32)] = T.call_llvm_pure_intrin(
                T.llvm_lookup_intrinsic_id("llvm.hexagon.V6.vrmpybusv.acc.128B"),
                T.uint32(3),
                C[T.ramp(T.int32(0), 1, 32)],
                T.broadcast(A_i32, 32),
                B_i32x32,
                dtype="int32x32",
            )

    return dot_product_32x4_u8i8i32_desc, dot_product_32x4_u8i8i32_vrmpy


def generate_dot_product_32x2_i16i16i32(mem_scope="global"):
    @T.prim_func
    def dot_product_32x2_i16i16i32_desc(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (2,), "int16", offset_factor=1, scope=mem_scope)
        B = T.match_buffer(b, (32, 2), "int16", offset_factor=1, scope=mem_scope)
        C = T.match_buffer(c, (32,), "int32", offset_factor=1, scope=mem_scope)
        with T.block("root"):
            T.reads(C[0:32], A[0:2], B[0:32, 0:2])
            T.writes(C[0:32])
            for i in T.serial(0, 32):
                for k in T.serial(0, 2):
                    with T.block("update"):
                        vi, vk = T.axis.remap("SR", [i, k])
                        C[vi] = C[vi] + T.cast(A[vk], "int32") * T.cast(B[vi, vk], "int32")

    @T.prim_func
    def dot_product_32x2_i16i16i32_vdmpy(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (2,), "int16", offset_factor=1, scope=mem_scope)
        B = T.match_buffer(b, (32, 2), "int16", offset_factor=1, scope=mem_scope)
        C = T.match_buffer(c, (32,), "int32", offset_factor=1, scope=mem_scope)
        with T.block("root"):
            T.reads(C[0:32], A[0:2], B[0:32, 0:2])
            T.writes(C[0:32])

            A_i16x2 = A.vload([0], "int16x2")
            A_i32 = T.reinterpret(A_i16x2, dtype="int32")

            B_i16x64 = B.vload([0, 0], dtype="int16x64")
            B_i32x32 = T.reinterpret(B_i16x64, dtype="int32x32")

            C[T.ramp(T.int32(0), 1, 32)] = T.call_llvm_pure_intrin(
                T.llvm_lookup_intrinsic_id("llvm.hexagon.V6.vdmpyhvsat.acc.128B"),
                T.uint32(3),
                C[T.ramp(T.int32(0), 1, 32)],
                T.Broadcast(A_i32, 32),
                B_i32x32,
                dtype="int32x32",
            )

    return dot_product_32x2_i16i16i32_desc, dot_product_32x2_i16i16i32_vdmpy


def generate_copy_strided_src_32_man(mem_scope="global"):
    @T.prim_func
    def intrin_desc(a: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (4, 32), "uint8", offset_factor=1, scope=mem_scope)
        C = T.match_buffer(c, (4, 32), "uint8", offset_factor=1, scope=mem_scope)
        with T.block("root"):
            T.reads(A[0:4, 0:32])
            T.writes(C[0:4, 0:32])
            for w, c_i in T.grid(4, 32):
                with T.block("update"):
                    w_, c_i_ = T.axis.remap("SS", [w, c_i])
                    C[w_, c_i_] = A[w_, c_i_]  # Just copy, but A has not trivial stride

    @T.prim_func
    def intrin_impl(a: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (4, 32), "uint8", offset_factor=1, scope=mem_scope)
        C = T.match_buffer(c, (4, 32), "uint8", offset_factor=1, scope=mem_scope)
        with T.block("root"):
            T.reads(A[0:4, 0:32])
            T.writes(C[0:4, 0:32])

            src: T.handle = T.address_of(A[0, 0], dtype="handle")
            dst: T.handle = T.address_of(C[0, 0], dtype="handle")

            src_1: T.handle = T.address_of(A[1, 0], dtype="handle")
            src_stride = T.reinterpret(src_1, dtype="int32") - T.reinterpret(src, dtype="int32")

            T.call_extern("HexagonCopyStridedSrc", src, dst, src_stride, dtype="")

    return intrin_desc, intrin_impl


def generate_copy_strided_dst_32_man(mem_scope="global"):
    @T.prim_func
    def intrin_desc(a: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (4, 32), "uint8", offset_factor=1, scope=mem_scope)
        C = T.match_buffer(c, (4, 32), "uint8", offset_factor=1, scope=mem_scope)
        with T.block("root"):
            T.reads(A[0:4, 0:32])
            T.writes(C[0:4, 0:32])
            for w, c_i in T.grid(4, 32):
                with T.block("update"):
                    w_, c_i_ = T.axis.remap("SS", [w, c_i])
                    C[w_, c_i_] = A[w_, c_i_]  # Just copy, but C has not trivial stride

    @T.prim_func
    def intrin_impl(a: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (4, 32), "uint8", offset_factor=1, scope=mem_scope)
        C = T.match_buffer(c, (4, 32), "uint8", offset_factor=1, scope=mem_scope)
        with T.block("root"):
            T.reads(A[0:4, 0:32])
            T.writes(C[0:4, 0:32])

            src: T.handle = T.address_of(A[0, 0], dtype="handle")
            dst: T.handle = T.address_of(C[0, 0], dtype="handle")

            dst_1: T.handle = T.address_of(C[1, 0], dtype="handle")
            dst_stride = T.reinterpret(dst_1, dtype="int32") - T.reinterpret(dst, dtype="int32")

            T.call_extern("HexagonCopyStridedDst", src, dst, dst_stride, dtype="")

    return intrin_desc, intrin_impl


def generate_copy_strided_dst_8x8x32_man(mem_scope="global"):
    @T.prim_func
    def intrin_desc(a: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (8, 8, 32), "uint8", offset_factor=1, scope=mem_scope)
        C = T.match_buffer(c, (8, 8, 32), "uint8", offset_factor=1, scope=mem_scope)
        with T.block("root"):
            T.reads(A[0:8, 0:8, 0:32])
            T.writes(C[0:8, 0:8, 0:32])
            for h, w, c_i in T.grid(8, 8, 32):
                with T.block("update"):
                    h_, w_, c_i_ = T.axis.remap("SSS", [h, w, c_i])
                    C[h_, w_, c_i_] = A[h_, w_, c_i_]  # Just copy, but C has not trivial stride

    @T.prim_func
    def intrin_impl(a: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (8, 8, 32), "uint8", offset_factor=1, scope=mem_scope)
        C = T.match_buffer(c, (8, 8, 32), "uint8", offset_factor=1, scope=mem_scope)
        with T.block("root"):
            T.reads(A[0:8, 0:8, 0:32])
            T.writes(C[0:8, 0:8, 0:32])

            src: T.handle = T.address_of(A[0, 0, 0], dtype="handle")
            dst: T.handle = T.address_of(C[0, 0, 0], dtype="handle")

            dst_w_1: T.handle = T.address_of(C[0, 1, 0], dtype="handle")
            dst_w_stride = T.reinterpret(dst_w_1, dtype="int32") - T.reinterpret(dst, dtype="int32")

            dst_h_1: T.handle = T.address_of(C[1, 0, 0], dtype="handle")
            dst_h_stride = T.reinterpret(dst_h_1, dtype="int32") - T.reinterpret(dst, dtype="int32")

            T.call_extern("HexagonCroutonBlock_8x8x32_intin", src, dst, dst_w_stride, dst_h_stride, dtype="")

    return intrin_desc, intrin_impl


def generate_trans_4x4x32_u8_man(mem_scope="global"):
    @T.prim_func
    def intrin_desc(a: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (4, 128), "uint8", offset_factor=1, scope=mem_scope)
        C = T.match_buffer(c, (4, 1, 1, 1, 4, 32), "uint8", offset_factor=1, scope=mem_scope)
        with T.block("root"):
            T.reads(A[0:4, 0:128])
            T.writes(C[0:4, 1, 1, 1, 0:4, 0:32])
            for c_o, w, c_i in T.grid(4, 4, 32):
                with T.block("update"):
                    c_o_, w_, c_i_ = T.axis.remap("SSS", [c_o, w, c_i])
                    C[c_o_, 0, 0, 0, w_, c_i_] = A[w_, c_o_ * 32 + c_i_]

    @T.prim_func
    def intrin_impl(a: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (4, 128), "uint8", offset_factor=1, scope=mem_scope)
        C = T.match_buffer(c, (4, 1, 1, 1, 4, 32), "uint8", offset_factor=1, scope=mem_scope)

        with T.block("root"):
            T.reads(A[0:4, 0:128])
            T.writes(C[0:4, 1, 1, 1, 0:4, 0:32])

            src: T.handle = T.address_of(A[0, 0], dtype="handle")
            dst: T.handle = T.address_of(C[0, 0, 0, 0, 0, 0], dtype="handle")

            # Works
            src_1: T.handle = T.address_of(A[1, 0], dtype="handle")
            dst_1: T.handle = T.address_of(C[1, 0, 0, 0, 0, 0], dtype="handle")

            src_stride = T.reinterpret(src_1, dtype="int32") - T.reinterpret(src, dtype="int32")
            dst_stride = T.reinterpret(dst_1, dtype="int32") - T.reinterpret(dst, dtype="int32")

            T.call_extern("Hexagon4x4x32Transform", src, dst, src_stride, dst_stride, dtype="")

    return intrin_desc, intrin_impl


def generate_trans_4x4x32_u8(mem_scope="global"):
    @T.prim_func
    def intrin_desc(a: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (4, 128), "uint8", offset_factor=1, scope=mem_scope)
        C = T.match_buffer(c, (4, 1, 1, 1, 4, 32), "uint8", offset_factor=1, scope=mem_scope)
        with T.block("root"):
            T.reads(A[0:4, 0:128])
            T.writes(C[0:4, 1, 1, 1, 0:4, 0:32])
            for c_o, w, c_i in T.grid(4, 4, 32):
                with T.block("update"):
                    c_o_, w_, c_i_ = T.axis.remap("SSS", [c_o, w, c_i])
                    C[c_o_, 0, 0, 0, w_, c_i_] = A[w_, c_o_ * 32 + c_i_]

    @T.prim_func
    def intrin_impl(a: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (4, 128), "uint8", offset_factor=1, scope=mem_scope)
        C = T.match_buffer(c, (4, 1, 1, 1, 4, 32), "uint8", offset_factor=1, scope=mem_scope)
        with T.block("root"):
            T.reads(A[0:4, 0:128])
            T.writes(C[0:4, 1, 1, 1, 0:4, 0:32])

            vshuff_intr = T.llvm_lookup_intrinsic_id("llvm.hexagon.V6.vshuffvdd.128B")

            v0: T.int32x32 = T.reinterpret(A[0, T.ramp(0, 1, 128)], dtype="int32x32")
            v1: T.int32x32 = T.reinterpret(A[1, T.ramp(0, 1, 128)], dtype="int32x32")
            v2: T.int32x32 = T.reinterpret(A[2, T.ramp(0, 1, 128)], dtype="int32x32")
            v3: T.int32x32 = T.reinterpret(A[3, T.ramp(0, 1, 128)], dtype="int32x32")

            v10 = T.call_llvm_pure_intrin(vshuff_intr, T.uint32(3), v1, v0, 32, dtype="int32x64")
            v32 = T.call_llvm_pure_intrin(vshuff_intr, T.uint32(3), v3, v2, 32, dtype="int32x64")

            v0_ = T.vectorlow(v10, dtype="int32x32")
            v1_ = T.vectorhigh(v10, dtype="int32x32")
            v2_ = T.vectorlow(v32, dtype="int32x32")
            v3_ = T.vectorhigh(v32, dtype="int32x32")

            v20 = T.call_llvm_pure_intrin(vshuff_intr, T.uint32(3), v2_, v0_, 64, dtype="int32x64")
            v31 = T.call_llvm_pure_intrin(vshuff_intr, T.uint32(3), v3_, v1_, 64, dtype="int32x64")

            C[0, 0, 0, 0, 0, T.ramp(0, 1, 128)] = T.reinterpret(T.vectorlow(v20, dtype="int32x32"), dtype="uint8x128")
            C[1, 0, 0, 0, 0, T.ramp(0, 1, 128)] = T.reinterpret(T.vectorlow(v31, dtype="int32x32"), dtype="uint8x128")
            C[2, 0, 0, 0, 0, T.ramp(0, 1, 128)] = T.reinterpret(T.vectorhigh(v20, dtype="int32x32"), dtype="uint8x128")
            C[3, 0, 0, 0, 0, T.ramp(0, 1, 128)] = T.reinterpret(T.vectorhigh(v31, dtype="int32x32"), dtype="uint8x128")

    return intrin_desc, intrin_impl


VRMPY_u8u8i32_INTRIN = "dot_32x4_u8u8i32_vrmpy"

TensorIntrin.register(VRMPY_u8u8i32_INTRIN, *generate_dot_product_32x4_u8u8i32())

VRMPY_u8i8i32_INTRIN = "dot_32x4_u8i8i32_vrmpy"

TensorIntrin.register(VRMPY_u8i8i32_INTRIN, *generate_dot_product_32x4_u8i8i32())

VDMPY_i16i16i32_INTRIN = "dot_product_32x2_i16i16i32_vdmpy"

TensorIntrin.register(VDMPY_i16i16i32_INTRIN, *generate_dot_product_32x2_i16i16i32())

VRMPY_u8u8i32_VTCM_INTRIN = "dot_32x4_u8u8i32_vtcm_vrmpy"
TensorIntrin.register(VRMPY_u8u8i32_VTCM_INTRIN, *generate_dot_product_32x4_u8u8i32("global.vtcm"))

VRMPY_u8i8i32_VTCM_INTRIN = "dot_32x4_u8i8i32_vtcm_vrmpy"
TensorIntrin.register(VRMPY_u8i8i32_VTCM_INTRIN, *generate_dot_product_32x4_u8i8i32("global.vtcm"))

DMA_READ_128_u8 = "dma_read_128_u8"
TensorIntrin.register(DMA_READ_128_u8, *generate_dma_load_intrin(128, "uint8"))

DMA_READ_128_i8 = "dma_read_128_i8"
TensorIntrin.register(DMA_READ_128_i8, *generate_dma_load_intrin(128, "int8"))

VSHUFF_TRANSFORM_4x4x32_INTRIN = "trans_4x4x32_u8"
TensorIntrin.register(VSHUFF_TRANSFORM_4x4x32_INTRIN, *generate_trans_4x4x32_u8())

VSHUFF_TRANSFORM_4x4x32_INTRIN_MAN = "trans_4x4x32_u8_man"
TensorIntrin.register(VSHUFF_TRANSFORM_4x4x32_INTRIN_MAN, *generate_trans_4x4x32_u8_man())

VMEM_COPY_STRIDED_SRC_32_INTRIN_MAN = "vmem_copy_strided_src_32_man"
TensorIntrin.register(VMEM_COPY_STRIDED_SRC_32_INTRIN_MAN, *generate_copy_strided_src_32_man())

VMEM_COPY_STRIDED_DST_32_INTRIN_MAN = "vmem_copy_strided_dst_32_man"
TensorIntrin.register(VMEM_COPY_STRIDED_DST_32_INTRIN_MAN, *generate_copy_strided_dst_32_man())

VMEM_COPY_STRIDED_DST_8x8x32_INTRIN_MAN = "vmem_copy_strided_dst_8x8x32_man"
TensorIntrin.register(VMEM_COPY_STRIDED_DST_8x8x32_INTRIN_MAN, *generate_copy_strided_dst_8x8x32_man())
