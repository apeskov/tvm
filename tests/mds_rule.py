from tvm import meta_schedule as ms 
from tvm.tir.schedule import Schedule, BlockRV

@ms.utils.derived_object
class MDS1ScheduleRule(ms.schedule_rule.PyScheduleRule):
    def __init__(self, decisions = {}) -> None:
        super().__init__()
        self._decisions = decisions

    def _initialize_with_tune_context(self, context) -> None:
        pass
    
    def is_acceptable(self, sch: Schedule, block):
        """Check if provided block is gemm
        Trifial implementation. Check blpck name ends with "_matmul"
        Is not correct for general cases. 
        """
        b = sch.get(block)
        return "matmul" in b.name_hint

    def deduce_mpad_value(self, sch: Schedule):
        """
        Define m pad value will be used in schedule.
        Read proper hint attribute or provide some heuristic value.
        """
        func = sch.mod[sch.func_working_on]
        if "metaschedule.hint.m_pad_value" in func.attrs.keys():
            m_pad_value = func.attrs["metaschedule.hint.m_pad_value"]
            if isinstance(m_pad_value, tvm.tir.IntImm):
                m_pad_value = m_pad_value.value
            return m_pad_value

        if "m_pad" in self._decisions:
            return self._decisions["m_pad"]
        
        return 64  # default value 

    def apply(self, sch: Schedule, block: BlockRV):
        if not self.is_acceptable(sch, block):
            return [sch]

        m_pad = self.deduce_mpad_value(sch)
        sch = sch.copy()

        # padding prolog
        # assume order of loops is : B M N K
        sch.pad_einsum(block, padding=[1, m_pad, 16, 16])
        b_pad_a = sch.get_producers(block)[0]
        b_pad_o = sch.get_consumers(block)[0]

        # NB! Do not use transform_layout(block, buffer=("read", 1), ...)
        #     it may change layout of input argument if block is the first one.
        #
        # To respect weights layout
        is_trans = "NT_" in sch.get(block).name_hint

        # block 16x16x16
        lb, lm, ln, lk = sch.get_loops(block)
        lm, lm_b = sch.split(lm, factors=[None, 16])
        ln, ln_b = sch.split(ln, factors=[None, 16])
        lk, lk_b = sch.split(lk, factors=[None, 16])
        sch.reorder(lm, ln, lk, lm_b, ln_b, lk_b)
        b_wmma = sch.blockize(lm_b)

        m_decisions = self._decisions["m_factors"] if "m_factors" in self._decisions else None
        n_decisions = self._decisions["n_factors"] if "n_factors" in self._decisions else None
        k_decisions = self._decisions["k_factors"] if "k_factors" in self._decisions else None

        lm_4, lm = sch.split(lm, factors=[None, m_pad//16])
        lm_factors = sch.sample_perfect_tile(loop=lm, n=3, max_innermost_factor=4, decision=m_decisions)
        lm_3, lm_2, lm_1 = sch.split(lm, factors=lm_factors)
        ln_factors = sch.sample_perfect_tile(loop=ln, n=4, max_innermost_factor=4, decision=n_decisions)
        ln_4, ln_3, ln_2, ln_1 = sch.split(ln, factors=ln_factors)
        lk_factors = sch.sample_perfect_tile(loop=lk, n=2, max_innermost_factor=4, decision=k_decisions)
        lk_2, lk_1 = sch.split(lk, factors=lk_factors)
        sch.reorder(lm_4, ln_4, lm_3, ln_3, lm_2, ln_2, lk_2, lk_1, lm_1, ln_1)
        lnm_by = sch.fuse(lm_4, ln_4)
        sch.bind(lnm_by, thread_axis="blockIdx.y")
        lnm_bx = sch.fuse(lm_3, ln_3)
        sch.bind(lnm_bx, thread_axis="blockIdx.x")
        lnm_ty = sch.fuse(lm_2, ln_2)
        sch.bind(lnm_ty, thread_axis="threadIdx.y")


        # copy from/to shared on level of L1 block
        b_o_shared = sch.cache_write(b_wmma, write_buffer_index=0, storage_scope="shared.dyn")
        b_o_wmma = sch.cache_write(b_wmma, write_buffer_index=0, storage_scope="wmma.accumulator")
        sch.reverse_compute_at(b_o_wmma, loop=lnm_ty, preserve_unit_loops=True, index=-1)
        sch.reverse_compute_at(b_o_shared, loop=lnm_ty, preserve_unit_loops=True, index=-1)
        
        b_a_shared = sch.cache_read(b_wmma, read_buffer_index=0, storage_scope="shared.dyn")
        b_a_wmma = sch.cache_read(b_wmma, read_buffer_index=0, storage_scope="wmma.matrix_a")
        sch.compute_at(b_a_wmma, loop=lk_1, preserve_unit_loops=True, index=-1)
        sch.compute_at(b_a_shared, loop=lk_2, preserve_unit_loops=True, index=-1)

        b_b_shared = sch.cache_read(b_wmma, read_buffer_index=1, storage_scope="shared.dyn")
        b_b_wmma = sch.cache_read(b_wmma, read_buffer_index=1, storage_scope="wmma.matrix_b")
        sch.compute_at(b_b_wmma, loop=lk_1, preserve_unit_loops=True, index=-1)
        sch.compute_at(b_b_shared, loop=lk_2, preserve_unit_loops=True, index=-1)

        b_wmma_init = sch.decompose_reduction(block=b_wmma, loop=lk_2)

        # tensozise helper
        def blk_tensorize(blk, intrin_name):
            *_, lm, ln = sch.get_loops(blk)
            lm, lm_b = sch.split(lm, factors=[None, 16])
            ln, ln_b = sch.split(ln, factors=[None, 16])
            sch.reorder(lm, ln, lm_b, ln_b)
            blk_16x16 = sch.blockize(lm_b)
            # TODO: add bind to Ty???
            sch.tensorize(blk_16x16, intrin_name)

        # vectorize helper
        def blk_vectorize(blk, vec_size=4, cooperative=True):
            # 16x16 4*32*Ty
            # Ideally it should be 8 (128bit register containd 8 half floats) 
            ty_size = (lm_factors[-2] * ln_factors[-2])  # TODO: error "Stringifying is not supported for type: tir.Mul"
            tx_size = 32
            *_, lm, ln = sch.get_loops(blk) 
            lmn = sch.fuse(lm, ln)
            # lmn, lmn_ty, lmn_tx, lmn_v = sch.split(lmn, factors=[None, ty_size, tx_size, vec_size])
            lmn, lm_ty, ln_ty_2, lmn_tx, lmn_v = sch.split(lmn, factors=[None, lm_factors[-2], ln_factors[-2], tx_size, vec_size])
            sch.bind(lmn_tx, thread_axis="threadIdx.x")
            if cooperative:
                sch.bind(sch.fuse(lm_ty, ln_ty_2), thread_axis="threadIdx.y")
            sch.vectorize(lmn_v)

            # NB! significant impact. Looks like bank conflict. "buffer_index=0" for cache write, is it correct? 
            # sch.storage_align(block=blk, buffer_index=0, axis=-2, factor=16, offset=8)   
        
        # tensorize compute
        sch.tensorize(b_wmma, "wmma_sync_16x16x16_f16f16f16_trans" if is_trans else "wmma_sync_16x16x16_f16f16f16")
        sch.tensorize(b_wmma_init, "wmma_fill_16x16x16_f16")

        # tensorize load/store WMMA regs
        blk_tensorize(b_o_wmma, "wmma_store_16x16x16_f16_shared_dyn")
        blk_tensorize(b_a_wmma, "wmma_load_16x16x16_f16_a_shared_dyn")
        blk_tensorize(b_b_wmma, "wmma_load_16x16x16_f16_b_trans_shared_dyn" if is_trans else "wmma_load_16x16x16_f16_b_shared_dyn")

        # vectorize load/store smem
        blk_vectorize(b_a_shared, vec_size=4)
        blk_vectorize(b_b_shared, vec_size=4)
        blk_vectorize(b_o_shared, vec_size=4, cooperative=False)

        # Padding epilog
        sch.compute_inline(b_pad_a)
        sch.reverse_compute_inline(b_pad_o)

        # TBR
        # sch.reverse_compute_at(sch.get_block("var_matmul_intermediate_pad_shared.dyn"), sch.get_loops("var_matmul_intermediate_pad_shared.dyn_wmma.accumulator_o")[5])
        # TBR

        return [sch]


    def clone(self) -> ms.schedule_rule.ScheduleRule:
        return MDS1ScheduleRule()

