import tempfile

import tvm
from tvm.target import Target
from tvm import meta_schedule as ms
from tvm.script import tir as T

import tvm.tir.tensor_intrin.cuda

def get_matmul(M, N, K):
    @T.prim_func
    def matmul(
            A: T.Buffer((T.int64(M), T.int64(K)), "float16"),
            B: T.Buffer((T.int64(K), T.int64(N)), "float16"),
            C: T.Buffer((T.int64(M), T.int64(N)), "float16")
    ):
        T.func_attr({"global_symbol": "matmul", "tir.noalias": True})
        for i, j, k in T.grid(T.int64(M), T.int64(N), T.int64(K)):
            with T.block("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = T.float16(0)
                C[vi, vj] += A[vi, vk] * B[vk, vj]

    return matmul


def get_matmul_int4(M, N, K, G):
    assert K % 8 == 0,  "must be packable as a whole byte"

    # We need to calculate the group size based on the fact ceil(K / group_size) == G
    # we assume K is divisible by G, otherwise we need to
    # pass in group_size explicitly
    assert K % G == 0
    group_size = K // G

    func_name = f"matmul_{M}_{N}_{K}_{G}"

    @T.prim_func
    def matmul(
            A: T.Buffer((T.int64(M), T.int64(K)), "float16"),
            B_pack: T.Buffer((T.int64(K//8), T.int64(N)), "int32"),
            scales: T.Buffer((T.int64(G), T.int64(N)), "float16"),
            zeros_pack: T.Buffer((T.int64(G), T.int64(N//8),), "int32"),
            C: T.Buffer((T.int64(M), T.int64(N)), "float16"),
    ):
        B = T.alloc_buffer((T.int64(K), T.int64(N)), dtype="float16")

        T.func_attr({"global_symbol": func_name, "tir.noalias": True})

        zeros = T.alloc_buffer((T.int64(G), T.int64(N)), dtype="int32")
        for g, n in T.grid(T.int64(G), T.int64(N)):
            with T.block("zeros_decode"):
                vg, vn = T.axis.remap("SS", [g, n])
                zeros[vg, vn] = (zeros_pack[vg, vn // 8] >> (vn % 8 * 4) & 0xF) + T.int32(1)

        for k, n in T.grid(T.int64(K), T.int64(N)):
            with T.block("B_decode"):
                vk, vn= T.axis.remap("SS", [k, n])
                B[vk, vn] = T.cast((B_pack[vk // 8, vn] >> (vk % 8 * 4) & 0xF) - zeros[vk // group_size, vn], "float16") * scales[vk // group_size, vn]

        for i, j, k in T.grid(T.int64(M), T.int64(N), T.int64(K)):
            with T.block("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = T.float16(0)
                C[vi, vj] += A[vi, vk] * B[vk, vj]

    return matmul

def get_matmul_int4_dyn_m(N, K, G):
    assert K % 8 == 0,  "must be packable as a whole byte"

    # We need to calculate the group size based on the fact ceil(K / group_size) == G
    # we assume K is divisible by G, otherwise we need to
    # pass in group_size explicitly
    assert K % G == 0
    group_size = K // G
    
    func_name = f"matmul_dynm_{N}_{K}_{G}"

    @T.prim_func
    def matmul(
            a: T.handle,
            B_pack: T.Buffer((T.int64(K//8), T.int64(N)), "int32"),
            scales: T.Buffer((T.int64(G), T.int64(N)), "float16"),
            zeros_pack: T.Buffer((T.int64(G), T.int64(N//8),), "int32"),
            c: T.handle,
    ):
        m = T.int64()
        A = T.match_buffer(a, (m, T.int64(K)), "float16")
        C = T.match_buffer(c, (m, T.int64(N)), "float16")
        B = T.alloc_buffer((T.int64(K), T.int64(N)), dtype="float16")

        T.func_attr({"global_symbol": func_name, "tir.noalias": True})

        zeros = T.alloc_buffer((T.int64(G), T.int64(N)), dtype="int32")
        for g, n in T.grid(T.int64(G), T.int64(N)):
            with T.block("zeros_decode"):
                vg, vn = T.axis.remap("SS", [g, n])
                zeros[vg, vn] = (zeros_pack[vg, vn // 8] >> (vn % 8 * 4) & 0xF) + T.int32(1)

        for k, n in T.grid(T.int64(K), T.int64(N)):
            with T.block("B_decode"):
                vk, vn= T.axis.remap("SS", [k, n])
                B[vk, vn] = T.cast((B_pack[vk // 8, vn] >> (vk % 8 * 4) & 0xF) - zeros[vk // group_size, vn], "float16") * scales[vk // group_size, vn]

        for i, j, k in T.grid(T.int64(m), T.int64(N), T.int64(K)):
            with T.block("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = T.float16(0)
                C[vi, vj] += A[vi, vk] * B[vk, vj]

    return matmul


def apply_trace_int_v1(sch: tvm.tir.Schedule) -> None:
  """Tuned with relax. Dynamic M"""
  b0 = sch.get_block(name="zeros_decode", func_name="main")                                                                                                             
  b1 = sch.get_block(name="B_decode", func_name="main")                                                                                                                 
  b2 = sch.get_block(name="matmul", func_name="main")                                                                                                                   
  b3 = sch.get_block(name="root", func_name="main")                                                                                                                     
  sch.annotate(block_or_loop=b2, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")                                                                          
  b4 = sch.reindex(block=b2, buffer=("write", 0))                                                                                                                       
  b5 = sch.reindex(block=b2, buffer=("read", 0))                                                                                                                        
  b6 = sch.reindex(block=b2, buffer=("read", 1))                                                                                                                        
  sch.transform_layout(block=b2, buffer=("read", 0), index_map=lambda vi, vk: (vi, vk,), pad_value=None, assume_injective_transform=False)                              
  sch.transform_layout(block=b2, buffer=("read", 1), index_map=lambda vj, vk: (vk, vj,), pad_value=None, assume_injective_transform=False)                              
  sch.transform_layout(block=b2, buffer=("write", 0), index_map=lambda vi, vj: (vi, vj,), pad_value=None, assume_injective_transform=False)                             
  sch.transform_block_layout(block=b4, index_map=lambda vi, vj: (vi, vj,))                                                                                              
  sch.transform_block_layout(block=b5, index_map=lambda vi, vk: (vi, vk,))                                                                                              
  sch.transform_block_layout(block=b6, index_map=lambda vj, vk: (vk, vj,))                                                                                              
  sch.transform_block_layout(block=b2, index_map=lambda vi, vj, vk: (vi, vj, vk,))                                                                                      
  
  sch.pad_einsum(b2, (32, 16, 16))   # <= should be [16, 16, 16]
  l7, l8, l9 = sch.get_loops(block=b2)                                                                                                                                  
  l10, l11 = sch.split(loop=l9, factors=[None, 16], preserve_unit_iters=True)                                                                                           
  l12, l13 = sch.split(loop=l8, factors=[None, 16], preserve_unit_iters=True)                                                                                           
  l14, l15 = sch.split(loop=l7, factors=[None, 16], preserve_unit_iters=True)                                                                                           
  l16, l17, l18, l19, l20, l21 = sch.get_loops(block=b2)                                                                                                                
  sch.reorder(l18, l20, l15, l13, l11)                                                                                                                                  
  b22 = sch.blockize(target=l15, preserve_unit_iters=True)                                                                                                              
  sch.annotate(block_or_loop=b22, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_sync_16x16x16_f16f16f16")                                                       
  sch.annotate(block_or_loop=b22, ann_key="meta_schedule.auto_tensorize_init", ann_val="wmma_fill_16x16x16_f16")                                                        
  sch.annotate(block_or_loop=b22, ann_key="warp_execution", ann_val=1)        
  l23, l24, l25 = sch.get_loops(block=b22)                                                                                                                              
#   v26, v27, v28, v29, v30 = sch.sample_perfect_tile(loop=l23, n=5, max_innermost_factor=4, decision=[1, 1, 1, 2, 1])                                                    
#   l31, l32, l33, l34, l35 = sch.split(loop=l23, factors=[v26, v27, v28, v29, v30], preserve_unit_iters=True)
  l31, l32, l33, l34, l35 = sch.split(loop=l23, factors=[None, 1, 1, 2, 1], preserve_unit_iters=True)
  v36, v37, v38, v39, v40 = sch.sample_perfect_tile(loop=l24, n=5, max_innermost_factor=4, decision=[10, 8, 4, 1, 1])                                                   
  l41, l42, l43, l44, l45 = sch.split(loop=l24, factors=[v36, v37, v38, v39, v40], preserve_unit_iters=True)                                                            
  v46, v47, v48 = sch.sample_perfect_tile(loop=l25, n=3, max_innermost_factor=4, decision=[320, 1, 4])                                                                  
  l49, l50, l51 = sch.split(loop=l25, factors=[v46, v47, v48], preserve_unit_iters=True)                                                                                
  sch.reorder(l31, l41, l32, l42, l33, l43, l49, l50, l34, l44, l51, l35, l45)                                                                                          
  l52 = sch.fuse(l31, l41, preserve_unit_iters=True)                                                                                                                    
  sch.bind(loop=l52, thread_axis="blockIdx.y")                                                                                                                          
  l53 = sch.fuse(l32, l42, preserve_unit_iters=True)                                                                                                                    
  sch.bind(loop=l53, thread_axis="blockIdx.x")                                                                                                                          
  l54 = sch.fuse(l33, l43, preserve_unit_iters=True)                                                                                                                    
  sch.bind(loop=l54, thread_axis="threadIdx.y")                                                                                                                         
  sch.annotate(block_or_loop=b22, ann_key="meta_schedule.thread_extent_low_inclusive", ann_val=32)                                                                      
  sch.annotate(block_or_loop=b22, ann_key="meta_schedule.thread_extent_high_inclusive", ann_val=1024)                                                                   
  b55 = sch.cache_write(block=b22, write_buffer_index=0, storage_scope="shared.dyn")                                                                                    
  sch.reverse_compute_at(block=b55, loop=l53, preserve_unit_loops=True, index=-1)                                                                                       
  b56 = sch.cache_write(block=b22, write_buffer_index=0, storage_scope="wmma.accumulator")                                                                              
  sch.reverse_compute_at(block=b56, loop=l54, preserve_unit_loops=True, index=-1)                                                                                       
  l57, l58, l59, l60 = sch.get_loops(block=b55)                                                                                                                         
  l61 = sch.fuse(l59, l60, preserve_unit_iters=True)                                                                                                                    
  v62 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=2)                                                                     
  sch.annotate(block_or_loop=b55, ann_key="meta_schedule.cooperative_fetch", ann_val=v62)                                                                               
  sch.reverse_compute_inline(block=b4)                                                                                                                                  
  l63, l64, l65, l66, l67 = sch.get_loops(block=b56)                                                                                                                    
  l68, l69 = sch.split(loop=l67, factors=[None, 16], preserve_unit_iters=True)                                                                                          
  l70, l71 = sch.split(loop=l66, factors=[None, 16], preserve_unit_iters=True)                                                                                          
  l72, l73, l74, l75, l76, l77, l78 = sch.get_loops(block=b56)                                                                                                          
  sch.reorder(l77, l71, l69)                                                                                                                                            
  b79 = sch.blockize(target=l71, preserve_unit_iters=True)                                                                                                              
  sch.annotate(block_or_loop=b79, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_store_16x16x16_f16_shared_dyn")                                                 
  b80 = sch.cache_read(block=b22, read_buffer_index=0, storage_scope="shared.dyn", consumer_blocks=[b22])                                                               
  sch.compute_at(block=b80, loop=l49, preserve_unit_loops=True, index=-1)                                                                                               
  l81, l82, l83, l84, l85, l86 = sch.get_loops(block=b80)                                                                                                               
  l87 = sch.fuse(l85, l86, preserve_unit_iters=True)                                                                                                                    
  v88 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=2)                                                                     
  sch.annotate(block_or_loop=b80, ann_key="meta_schedule.cooperative_fetch", ann_val=v88)                                                                               
  b89 = sch.cache_read(block=b22, read_buffer_index=1, storage_scope="shared.dyn", consumer_blocks=[b22])                                                               
  sch.compute_at(block=b89, loop=l49, preserve_unit_loops=True, index=-1)                                                                                               
  l90, l91, l92, l93, l94, l95 = sch.get_loops(block=b89)                                                                                                               
  l96 = sch.fuse(l94, l95, preserve_unit_iters=True)                                                                                                                    
  v97 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=1)                                                                     
  sch.annotate(block_or_loop=b89, ann_key="meta_schedule.cooperative_fetch", ann_val=v97)                                                                               
  b98 = sch.cache_read(block=b22, read_buffer_index=0, storage_scope="wmma.matrix_a")                                                                                   
  sch.compute_at(block=b98, loop=l50, preserve_unit_loops=True, index=-1)                                                                                               
  l99, l100, l101, l102, l103, l104, l105 = sch.get_loops(block=b98)                                                                                                    
  l106, l107 = sch.split(loop=l105, factors=[None, 16], preserve_unit_iters=True)                                                                                       
  l108, l109 = sch.split(loop=l104, factors=[None, 16], preserve_unit_iters=True)                                                                                       
  l110, l111, l112, l113, l114, l115, l116, l117, l118 = sch.get_loops(block=b98)                                                                                       
  sch.reorder(l117, l109, l107)                                                                                                                                         
  b119 = sch.blockize(target=l109, preserve_unit_iters=True)                                                                                                            
  sch.annotate(block_or_loop=b119, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_load_16x16x16_f16_a_shared_dyn")                                               
  b120 = sch.cache_read(block=b22, read_buffer_index=1, storage_scope="wmma.matrix_b")                                                                                  
  sch.compute_at(block=b120, loop=l50, preserve_unit_loops=True, index=-1)                                                                                              
  l121, l122, l123, l124, l125, l126, l127 = sch.get_loops(block=b120)                                                                                                  
  l128, l129 = sch.split(loop=l127, factors=[None, 16], preserve_unit_iters=True)                                                                                       
  l130, l131 = sch.split(loop=l126, factors=[None, 16], preserve_unit_iters=True)                                                                                       
  l132, l133, l134, l135, l136, l137, l138, l139, l140 = sch.get_loops(block=b120)                                                                                      
  sch.reorder(l139, l131, l129)                                                                                                                                         
  b141 = sch.blockize(target=l131, preserve_unit_iters=True)    
  sch.annotate(block_or_loop=b141, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_load_16x16x16_f16_b_shared_dyn")
  sch.compute_inline(block=b5)
  sch.compute_inline(sch.get_block("A_reindex_pad"))  # inline pad A, new one
  sch.compute_inline(block=b6)
  sch.storage_align(block=b80, buffer_index=0, axis=-2, factor=32, offset=8)
  sch.storage_align(block=b89, buffer_index=0, axis=-2, factor=32, offset=8)
  sch.compute_inline(block=b1)
  sch.compute_inline(block=b0)
  sch.reverse_compute_inline(sch.get_block("C_reindex_pad"))  # inline pad A, new one
  v142 = sch.sample_categorical(candidates=[0, 16, 64, 512, 1024], probs=[0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001], decision=2)
  sch.annotate(block_or_loop=b3, ann_key="meta_schedule.unroll_explicit", ann_val=v142)
  sch.enter_postproc()
  sch.unannotate(block_or_loop=b55, ann_key="meta_schedule.cooperative_fetch")
  l143, l144, l145 = sch.get_loops(block=b55)
  l146, l147, l148, l149 = sch.split(loop=l145, factors=[None, 4, 32, 4], preserve_unit_iters=True)
  sch.vectorize(loop=l149)
  sch.bind(loop=l148, thread_axis="threadIdx.x")
  sch.bind(loop=l147, thread_axis="threadIdx.y")
  sch.unannotate(block_or_loop=b80, ann_key="meta_schedule.cooperative_fetch")
  l150, l151, l152, l153, l154 = sch.get_loops(block=b80)
  l155, l156, l157, l158 = sch.split(loop=l154, factors=[None, 4, 32, 4], preserve_unit_iters=True)
  sch.vectorize(loop=l158)
  sch.bind(loop=l157, thread_axis="threadIdx.x")
  sch.bind(loop=l156, thread_axis="threadIdx.y")
  sch.unannotate(block_or_loop=b89, ann_key="meta_schedule.cooperative_fetch")
  l159, l160, l161, l162, l163 = sch.get_loops(block=b89)
  l164, l165, l166 = sch.split(loop=l163, factors=[None, 4, 32], preserve_unit_iters=True)
  sch.bind(loop=l166, thread_axis="threadIdx.x")
  sch.bind(loop=l165, thread_axis="threadIdx.y")
  b167 = sch.get_block(name="root", func_name="main")
  sch.unannotate(block_or_loop=b167, ann_key="meta_schedule.unroll_explicit")
  b168, b169, b170, b171, b172, b173, b174 = sch.get_child_blocks(b167)
  l175, l176, l177, l178, l179, l180, l181, l182 = sch.get_loops(block=b168)
  l183, l184, l185, l186, l187, l188, l189 = sch.get_loops(block=b169)
  l190, l191, l192, l193, l194, l195, l196 = sch.get_loops(block=b170)
  l197, l198, l199, l200, l201, l202, l203 = sch.get_loops(block=b171)
  l204, l205, l206, l207, l208, l209, l210, l211, l212, l213 = sch.get_loops(block=b172)
  sch.annotate(block_or_loop=l204, ann_key="pragma_auto_unroll_max_step", ann_val=64)
  sch.annotate(block_or_loop=l204, ann_key="pragma_unroll_explicit", ann_val=1)
  l214, l215, l216, l217, l218 = sch.get_loops(block=b173)
  l219, l220, l221, l222, l223, l224 = sch.get_loops(block=b174)
  b225 = sch.get_block(name="matmul_o", func_name="main")
  l226, l227, l228, l229, l230, l231, l232, l233, l234, l235 = sch.get_loops(block=b225)
  b236 = sch.decompose_reduction(block=b225, loop=l229)
  sch.unannotate(block_or_loop=b236, ann_key="meta_schedule.auto_tensorize")
  sch.annotate(block_or_loop=b236, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_fill_16x16x16_f16")
  sch.unannotate(block_or_loop=b225, ann_key="meta_schedule.auto_tensorize_init")
  sch.unannotate(block_or_loop=b236, ann_key="meta_schedule.auto_tensorize_init")
  b237 = sch.get_block(name="matmul_o_init", func_name="main")
  sch.unannotate(block_or_loop=b237, ann_key="meta_schedule.auto_tensorize")
  sch.tensorize(block_or_loop=b237, tensor_intrin="wmma_fill_16x16x16_f16", preserve_unit_iters=True)
  b238 = sch.get_block(name="A_reindex_pad_shared.dyn_wmma.matrix_a_o", func_name="main")  # new name
  sch.unannotate(block_or_loop=b238, ann_key="meta_schedule.auto_tensorize")
  sch.tensorize(block_or_loop=b238, tensor_intrin="wmma_load_16x16x16_f16_a_shared_dyn", preserve_unit_iters=True)
  b239 = sch.get_block(name="B_reindex_shared.dyn_wmma.matrix_b_o", func_name="main")
  sch.unannotate(block_or_loop=b239, ann_key="meta_schedule.auto_tensorize")
  sch.tensorize(block_or_loop=b239, tensor_intrin="wmma_load_16x16x16_f16_b_shared_dyn", preserve_unit_iters=True)
  b240 = sch.get_block(name="matmul_o_update", func_name="main")
  sch.unannotate(block_or_loop=b240, ann_key="meta_schedule.auto_tensorize")
  sch.tensorize(block_or_loop=b240, tensor_intrin="wmma_sync_16x16x16_f16f16f16", preserve_unit_iters=True)
  b241 = sch.get_block(name="C_reindex_pad_shared.dyn_wmma.accumulator_o", func_name="main")   # new name
  sch.unannotate(block_or_loop=b241, ann_key="meta_schedule.auto_tensorize")
  sch.tensorize(block_or_loop=b241, tensor_intrin="wmma_store_16x16x16_f16_shared_dyn", preserve_unit_iters=True)


def apply_trace_int_v1_static(sch: tvm.tir.Schedule) -> None:
  """Tuned with relax. Dynamic M"""
  b0 = sch.get_block(name="zeros_decode", func_name="main")                                                                                                             
  b1 = sch.get_block(name="B_decode", func_name="main")                                                                                                                 
  b2 = sch.get_block(name="matmul", func_name="main")                                                                                                                   
  b3 = sch.get_block(name="root", func_name="main")                                                                                                                     
  sch.annotate(block_or_loop=b2, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")                                                                          
  b4 = sch.reindex(block=b2, buffer=("write", 0))                                                                                                                       
  b5 = sch.reindex(block=b2, buffer=("read", 0))                                                                                                                        
  b6 = sch.reindex(block=b2, buffer=("read", 1))                                                                                                                        
  sch.transform_layout(block=b2, buffer=("read", 0), index_map=lambda vi, vk: (vi, vk,), pad_value=None, assume_injective_transform=False)                              
  sch.transform_layout(block=b2, buffer=("read", 1), index_map=lambda vj, vk: (vk, vj,), pad_value=None, assume_injective_transform=False)                              
  sch.transform_layout(block=b2, buffer=("write", 0), index_map=lambda vi, vj: (vi, vj,), pad_value=None, assume_injective_transform=False)                             
  sch.transform_block_layout(block=b4, index_map=lambda vi, vj: (vi, vj,))                                                                                              
  sch.transform_block_layout(block=b5, index_map=lambda vi, vk: (vi, vk,))                                                                                              
  sch.transform_block_layout(block=b6, index_map=lambda vj, vk: (vk, vj,))                                                                                              
  sch.transform_block_layout(block=b2, index_map=lambda vi, vj, vk: (vi, vj, vk,))                                                                                      
  
#   sch.pad_einsum(b2, (32, 16, 16))   # <= should be [16, 16, 16]
  l7, l8, l9 = sch.get_loops(block=b2)                                                                                                                                  
  l10, l11 = sch.split(loop=l9, factors=[None, 16], preserve_unit_iters=True)                                                                                           
  l12, l13 = sch.split(loop=l8, factors=[None, 16], preserve_unit_iters=True)                                                                                           
  l14, l15 = sch.split(loop=l7, factors=[None, 16], preserve_unit_iters=True)                                                                                           
  l16, l17, l18, l19, l20, l21 = sch.get_loops(block=b2)                                                                                                                
  sch.reorder(l18, l20, l15, l13, l11)                                                                                                                                  
  b22 = sch.blockize(target=l15, preserve_unit_iters=True)                                                                                                              
  sch.annotate(block_or_loop=b22, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_sync_16x16x16_f16f16f16")                                                       
  sch.annotate(block_or_loop=b22, ann_key="meta_schedule.auto_tensorize_init", ann_val="wmma_fill_16x16x16_f16")                                                        
  sch.annotate(block_or_loop=b22, ann_key="warp_execution", ann_val=1)        
  l23, l24, l25 = sch.get_loops(block=b22)                                                                                                                              
  v26, v27, v28, v29, v30 = sch.sample_perfect_tile(loop=l23, n=5, max_innermost_factor=4, decision=[1, 1, 1, 2, 1])                                                    
  l31, l32, l33, l34, l35 = sch.split(loop=l23, factors=[v26, v27, v28, v29, v30], preserve_unit_iters=True)
#   l31, l32, l33, l34, l35 = sch.split(loop=l23, factors=[None, 1, 1, 2, 1], preserve_unit_iters=True)
  v36, v37, v38, v39, v40 = sch.sample_perfect_tile(loop=l24, n=5, max_innermost_factor=4, decision=[10, 8, 4, 1, 1])                                                   
  l41, l42, l43, l44, l45 = sch.split(loop=l24, factors=[v36, v37, v38, v39, v40], preserve_unit_iters=True)                                                            
  v46, v47, v48 = sch.sample_perfect_tile(loop=l25, n=3, max_innermost_factor=4, decision=[320, 1, 4])                                                                  
  l49, l50, l51 = sch.split(loop=l25, factors=[v46, v47, v48], preserve_unit_iters=True)                                                                                
  sch.reorder(l31, l41, l32, l42, l33, l43, l49, l50, l34, l44, l51, l35, l45)                                                                                          
  l52 = sch.fuse(l31, l41, preserve_unit_iters=True)                                                                                                                    
  sch.bind(loop=l52, thread_axis="blockIdx.y")                                                                                                                          
  l53 = sch.fuse(l32, l42, preserve_unit_iters=True)                                                                                                                    
  sch.bind(loop=l53, thread_axis="blockIdx.x")                                                                                                                          
  l54 = sch.fuse(l33, l43, preserve_unit_iters=True)                                                                                                                    
  sch.bind(loop=l54, thread_axis="threadIdx.y")                                                                                                                         
  sch.annotate(block_or_loop=b22, ann_key="meta_schedule.thread_extent_low_inclusive", ann_val=32)                                                                      
  sch.annotate(block_or_loop=b22, ann_key="meta_schedule.thread_extent_high_inclusive", ann_val=1024)                                                                   
  b55 = sch.cache_write(block=b22, write_buffer_index=0, storage_scope="shared.dyn")                                                                                    
  sch.reverse_compute_at(block=b55, loop=l53, preserve_unit_loops=True, index=-1)                                                                                       
  b56 = sch.cache_write(block=b22, write_buffer_index=0, storage_scope="wmma.accumulator")                                                                              
  sch.reverse_compute_at(block=b56, loop=l54, preserve_unit_loops=True, index=-1)                                                                                       
  l57, l58, l59, l60 = sch.get_loops(block=b55)                                                                                                                         
  l61 = sch.fuse(l59, l60, preserve_unit_iters=True)                                                                                                                    
  v62 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=2)                                                                     
  sch.annotate(block_or_loop=b55, ann_key="meta_schedule.cooperative_fetch", ann_val=v62)                                                                               
  sch.reverse_compute_inline(block=b4)                                                                                                                                  
  l63, l64, l65, l66, l67 = sch.get_loops(block=b56)                                                                                                                    
  l68, l69 = sch.split(loop=l67, factors=[None, 16], preserve_unit_iters=True)                                                                                          
  l70, l71 = sch.split(loop=l66, factors=[None, 16], preserve_unit_iters=True)                                                                                          
  l72, l73, l74, l75, l76, l77, l78 = sch.get_loops(block=b56)                                                                                                          
  sch.reorder(l77, l71, l69)                                                                                                                                            
  b79 = sch.blockize(target=l71, preserve_unit_iters=True)                                                                                                              
  sch.annotate(block_or_loop=b79, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_store_16x16x16_f16_shared_dyn")                                                 
  b80 = sch.cache_read(block=b22, read_buffer_index=0, storage_scope="shared.dyn", consumer_blocks=[b22])                                                               
  sch.compute_at(block=b80, loop=l49, preserve_unit_loops=True, index=-1)                                                                                               
  l81, l82, l83, l84, l85, l86 = sch.get_loops(block=b80)                                                                                                               
  l87 = sch.fuse(l85, l86, preserve_unit_iters=True)                                                                                                                    
  v88 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=2)                                                                     
  sch.annotate(block_or_loop=b80, ann_key="meta_schedule.cooperative_fetch", ann_val=v88)                                                                               
  b89 = sch.cache_read(block=b22, read_buffer_index=1, storage_scope="shared.dyn", consumer_blocks=[b22])                                                               
  sch.compute_at(block=b89, loop=l49, preserve_unit_loops=True, index=-1)                                                                                               
  l90, l91, l92, l93, l94, l95 = sch.get_loops(block=b89)                                                                                                               
  l96 = sch.fuse(l94, l95, preserve_unit_iters=True)                                                                                                                    
  v97 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=1)                                                                     
  sch.annotate(block_or_loop=b89, ann_key="meta_schedule.cooperative_fetch", ann_val=v97)                                                                               
  b98 = sch.cache_read(block=b22, read_buffer_index=0, storage_scope="wmma.matrix_a")                                                                                   
  sch.compute_at(block=b98, loop=l50, preserve_unit_loops=True, index=-1)                                                                                               
  l99, l100, l101, l102, l103, l104, l105 = sch.get_loops(block=b98)                                                                                                    
  l106, l107 = sch.split(loop=l105, factors=[None, 16], preserve_unit_iters=True)                                                                                       
  l108, l109 = sch.split(loop=l104, factors=[None, 16], preserve_unit_iters=True)                                                                                       
  l110, l111, l112, l113, l114, l115, l116, l117, l118 = sch.get_loops(block=b98)                                                                                       
  sch.reorder(l117, l109, l107)                                                                                                                                         
  b119 = sch.blockize(target=l109, preserve_unit_iters=True)                                                                                                            
  sch.annotate(block_or_loop=b119, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_load_16x16x16_f16_a_shared_dyn")                                               
  b120 = sch.cache_read(block=b22, read_buffer_index=1, storage_scope="wmma.matrix_b")                                                                                  
  sch.compute_at(block=b120, loop=l50, preserve_unit_loops=True, index=-1)                                                                                              
  l121, l122, l123, l124, l125, l126, l127 = sch.get_loops(block=b120)                                                                                                  
  l128, l129 = sch.split(loop=l127, factors=[None, 16], preserve_unit_iters=True)                                                                                       
  l130, l131 = sch.split(loop=l126, factors=[None, 16], preserve_unit_iters=True)                                                                                       
  l132, l133, l134, l135, l136, l137, l138, l139, l140 = sch.get_loops(block=b120)                                                                                      
  sch.reorder(l139, l131, l129)                                                                                                                                         
  b141 = sch.blockize(target=l131, preserve_unit_iters=True)    
  sch.annotate(block_or_loop=b141, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_load_16x16x16_f16_b_shared_dyn")
  sch.compute_inline(block=b5)
#   sch.compute_inline(sch.get_block("A_reindex_pad"))  # inline pad A, new one
  sch.compute_inline(block=b6)
  sch.storage_align(block=b80, buffer_index=0, axis=-2, factor=32, offset=8)
  sch.storage_align(block=b89, buffer_index=0, axis=-2, factor=32, offset=8)
  sch.compute_inline(block=b1)
  sch.compute_inline(block=b0)
#   sch.reverse_compute_inline(sch.get_block("C_reindex_pad"))  # inline pad A, new one
  v142 = sch.sample_categorical(candidates=[0, 16, 64, 512, 1024], probs=[0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001], decision=2)
  sch.annotate(block_or_loop=b3, ann_key="meta_schedule.unroll_explicit", ann_val=v142)
  sch.enter_postproc()
  sch.unannotate(block_or_loop=b55, ann_key="meta_schedule.cooperative_fetch")
  l143, l144, l145 = sch.get_loops(block=b55)
  l146, l147, l148, l149 = sch.split(loop=l145, factors=[None, 4, 32, 4], preserve_unit_iters=True)
  sch.vectorize(loop=l149)
  sch.bind(loop=l148, thread_axis="threadIdx.x")
  sch.bind(loop=l147, thread_axis="threadIdx.y")
  sch.unannotate(block_or_loop=b80, ann_key="meta_schedule.cooperative_fetch")
  l150, l151, l152, l153, l154 = sch.get_loops(block=b80)
  l155, l156, l157, l158 = sch.split(loop=l154, factors=[None, 4, 32, 4], preserve_unit_iters=True)
  sch.vectorize(loop=l158)
  sch.bind(loop=l157, thread_axis="threadIdx.x")
  sch.bind(loop=l156, thread_axis="threadIdx.y")
  sch.unannotate(block_or_loop=b89, ann_key="meta_schedule.cooperative_fetch")
  l159, l160, l161, l162, l163 = sch.get_loops(block=b89)
  l164, l165, l166 = sch.split(loop=l163, factors=[None, 4, 32], preserve_unit_iters=True)
  sch.bind(loop=l166, thread_axis="threadIdx.x")
  sch.bind(loop=l165, thread_axis="threadIdx.y")
  b167 = sch.get_block(name="root", func_name="main")
  sch.unannotate(block_or_loop=b167, ann_key="meta_schedule.unroll_explicit")
  b168, b169, b170, b171, b172, b173, b174 = sch.get_child_blocks(b167)
  l175, l176, l177, l178, l179, l180, l181, l182 = sch.get_loops(block=b168)
  l183, l184, l185, l186, l187, l188, l189 = sch.get_loops(block=b169)
  l190, l191, l192, l193, l194, l195, l196 = sch.get_loops(block=b170)
  l197, l198, l199, l200, l201, l202, l203 = sch.get_loops(block=b171)
  l204, l205, l206, l207, l208, l209, l210, l211, l212, l213 = sch.get_loops(block=b172)
  sch.annotate(block_or_loop=l204, ann_key="pragma_auto_unroll_max_step", ann_val=64)
  sch.annotate(block_or_loop=l204, ann_key="pragma_unroll_explicit", ann_val=1)
  l214, l215, l216, l217, l218 = sch.get_loops(block=b173)
  l219, l220, l221, l222, l223, l224 = sch.get_loops(block=b174)
  b225 = sch.get_block(name="matmul_o", func_name="main")
  l226, l227, l228, l229, l230, l231, l232, l233, l234, l235 = sch.get_loops(block=b225)
  b236 = sch.decompose_reduction(block=b225, loop=l229)
  sch.unannotate(block_or_loop=b236, ann_key="meta_schedule.auto_tensorize")
  sch.annotate(block_or_loop=b236, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_fill_16x16x16_f16")
  sch.unannotate(block_or_loop=b225, ann_key="meta_schedule.auto_tensorize_init")
  sch.unannotate(block_or_loop=b236, ann_key="meta_schedule.auto_tensorize_init")
  b237 = sch.get_block(name="matmul_o_init", func_name="main")
  sch.unannotate(block_or_loop=b237, ann_key="meta_schedule.auto_tensorize")
  sch.tensorize(block_or_loop=b237, tensor_intrin="wmma_fill_16x16x16_f16", preserve_unit_iters=True)
#   b238 = sch.get_block(name="A_reindex_pad_shared.dyn_wmma.matrix_a_o", func_name="main")  # new name
  b238 = sch.get_block(name="A_reindex_shared.dyn_wmma.matrix_a_o", func_name="main")
  sch.unannotate(block_or_loop=b238, ann_key="meta_schedule.auto_tensorize")
  sch.tensorize(block_or_loop=b238, tensor_intrin="wmma_load_16x16x16_f16_a_shared_dyn", preserve_unit_iters=True)
  b239 = sch.get_block(name="B_reindex_shared.dyn_wmma.matrix_b_o", func_name="main")
  sch.unannotate(block_or_loop=b239, ann_key="meta_schedule.auto_tensorize")
  sch.tensorize(block_or_loop=b239, tensor_intrin="wmma_load_16x16x16_f16_b_shared_dyn", preserve_unit_iters=True)
  b240 = sch.get_block(name="matmul_o_update", func_name="main")
  sch.unannotate(block_or_loop=b240, ann_key="meta_schedule.auto_tensorize")
  sch.tensorize(block_or_loop=b240, tensor_intrin="wmma_sync_16x16x16_f16f16f16", preserve_unit_iters=True)
#   b241 = sch.get_block(name="C_reindex_pad_shared.dyn_wmma.accumulator_o", func_name="main")   # new name
  b241 = sch.get_block(name="C_reindex_shared.dyn_wmma.accumulator_o", func_name="main")   # new name
  sch.unannotate(block_or_loop=b241, ann_key="meta_schedule.auto_tensorize")
  sch.tensorize(block_or_loop=b241, tensor_intrin="wmma_store_16x16x16_f16_shared_dyn", preserve_unit_iters=True)


def main():
    configs = [
    #    M   N      K      G
        # (1, 15360, 5120,  40),
        (1, 5120,  5120,  40),
        # (1, 20480, 5120,  40),
        # (1, 5120,  20480, 160),
    ]
    target = Target("nvidia/nvidia-a100")
    work_dir = "dolly_tune_all_5"

    # ms_rule_type = "cuda"
    ms_rule_type = "cuda-tensorcore"
    
    funcs = {}
    for M, N, K, G in configs:
        func = get_matmul_int4(M, N, K, G)
        name = func.attrs["global_symbol"]
        funcs[name] = func

    database = ms.tir_integration.tune_tir(
        mod=tvm.ir.IRModule(funcs),
        target=target,
        work_dir=work_dir,
        max_trials_global=100500,
        max_trials_per_task=2048,
        num_trials_per_iter=32,
        space=ms.space_generator.PostOrderApply(                
                sch_rules=ms_rule_type,
                postprocs=ms_rule_type,
                mutator_probs=ms_rule_type,
            ),
    )



def mutate_to_dyn_m(trace: tvm.tir.schedule.Trace) -> tvm.tir.schedule.Trace:
    rv_map = {}
    def map_to(arg):
        return rv_map[arg] if arg in rv_map else arg
    
    def map_attrs_to(attr):
        if isinstance(attr, tvm.runtime.container.String) and attr == "A_reindex_shared.dyn_wmma.matrix_a_o":
            return "A_reindex_pad_shared.dyn_wmma.matrix_a_o"
        if isinstance(attr, tvm.runtime.container.String) and attr == "C_reindex_shared.dyn_wmma.accumulator_o":
            return "C_reindex_pad_shared.dyn_wmma.accumulator_o"
        return attr

    def just_copy(inst):
        return tvm.tir.schedule.Instruction(
            inst.kind,
            [map_to(inp) for inp in inst.inputs],
            [map_attrs_to(attr) for attr in inst.attrs],
            [map_to(outp) for outp in inst.outputs]
        )

    def process_SampleCategorical(inst: tvm.tir.schedule.Instruction):
        decision = int(trace.decisions[inst])
        val = inst.attrs[0][decision]
        rv_map[inst.outputs[0]] = val
        return []

    first_SamplePerfectTile = True 
    def process_SamplePerfectTile(inst: tvm.tir.schedule.Instruction):
        decision = [int(des) for des in trace.decisions[inst]]

        nonlocal first_SamplePerfectTile
        if first_SamplePerfectTile:
            first_SamplePerfectTile = False
            decision[0] = None
        
        for rv, val in zip(inst.outputs, decision):
            rv_map[rv] = T.int64(val) if val is not None else None

        return []

    rv_matmul = None
    def process_GetBlock(inst: tvm.tir.schedule.Instruction):
        nonlocal rv_matmul
        if rv_matmul is None and inst.attrs[0] == "matmul":
            rv_matmul = inst.outputs[0]
        
        return [just_copy(inst)]
    
    def process_GetLoops(inst: tvm.tir.schedule.Instruction):
        nonlocal rv_matmul
        if inst.inputs[0] == rv_matmul and len(inst.outputs) == 3:
            pad = tvm.tir.schedule.Instruction(
                tvm.tir.schedule.InstructionKind.get("PadEinsum"),
                [rv_matmul], [[T.int64(32), T.int64(16), T.int64(16)]], []
            )
            return [pad, just_copy(inst)]
        else:
            return [just_copy(inst)]
    
    first_ComputeInline = True 
    def process_ComputeInline(inst: tvm.tir.schedule.Instruction):
        nonlocal first_ComputeInline
        if first_ComputeInline:
            first_ComputeInline = False
            get_a = tvm.tir.schedule.Instruction(
                tvm.tir.schedule.InstructionKind.get("GetBlock"),
                [], ["A_reindex_pad", "main"], [tvm.tir.schedule.BlockRV()]
            )
            inlibe_a = tvm.tir.schedule.Instruction(
                tvm.tir.schedule.InstructionKind.get("ComputeInline"),
                [get_a.outputs[0]], [], []
            )
            get_b = tvm.tir.schedule.Instruction(
                tvm.tir.schedule.InstructionKind.get("GetBlock"),
                [], ["C_reindex_pad", "main"], [tvm.tir.schedule.BlockRV()]
            )
            inlibe_b = tvm.tir.schedule.Instruction(
                tvm.tir.schedule.InstructionKind.get("ReverseComputeInline"),
                [get_b.outputs[0]], [], []
            )
            return [get_a, inlibe_a, get_b, inlibe_b, just_copy(inst)]

        return [just_copy(inst)]

    processing_funcs ={
        "SamplePerfectTile": process_SamplePerfectTile,
        "SampleCategorical": process_SampleCategorical,
        "GetBlock": process_GetBlock,
        "GetLoops": process_GetLoops,
        "ComputeInline": process_ComputeInline,
    }

    new_insts = []
    for inst in trace.insts:
        if inst.kind.name in processing_funcs:
            for inst_ in processing_funcs[inst.kind.name](inst):
                new_insts.append(inst_)
        else:
            new_insts.append(just_copy(inst))

    return tvm.tir.schedule.Trace(new_insts, {})


if __name__ == "__main__":
    main()
