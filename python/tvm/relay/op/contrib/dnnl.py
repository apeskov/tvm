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
# pylint: disable=invalid-name, unused-argument
"""DNNL library supported operators.
There are two ways to registering a function for an op to indicate if it is
supported by DNNL.

- The first and simplest way is to use the helper so that
users only need to provide the operator name and a boolean value to indicate if
it is supported. For example:

    .. code-block:: python

      add = _register_external_op_helper("add")
      add = _register_external_op_helper("add", True)
      add = _register_external_op_helper("add", False)

- The other way is to implement the function by themselves to
check the attributes of the op and decide if it should be offloaded to DNNL.
"""
import tvm.ir
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name
from ...dataflow_pattern import wildcard, is_op, is_constant, is_expr
from .register import register_pattern_table, get_pattern_table
from tvm.relay.expr import const
import math
def partition_for_dnnl(mod, params=None):
    if params:
        mod["main"] = bind_params_by_name(mod["main"], params)

    seq = tvm.transform.Sequential(
        [
            transform.InferType(),
            transform.FoldConstant(),
            transform.FoldScaleAxis(),
            transform.DynamicToStatic(),
            transform.AlterOpLayout(),
            transform.FoldConstant(),
            transform.MergeComposite(get_pattern_table("dnnl")),
            transform.AnnotateTarget("dnnl"),
            #   If you no need in per layer performance statistic you can
            #   uncomment next line
            # transform.MergeCompilerRegions(),
            transform.PartitionGraph(),
        ]
    )

    return seq(mod)


def _register_external_op_helper(op_name, supported=True):
    """The helper function to indicate that a given operator can be supported
    by DNNL.

    Parameters
    ----------
    op_name : Str
        The name of operator that will be registered.

    Returns
    -------
    f : callable
        A function that returns if the operator is supported by DNNL.
    """

    @tvm.ir.register_op_attr(op_name, "target.dnnl")
    def _func_wrapper(expr):
        return supported

    return _func_wrapper


_register_external_op_helper("nn.batch_norm")
_register_external_op_helper("nn.conv2d")
_register_external_op_helper("nn.dense")
_register_external_op_helper("nn.relu")
# _register_external_op_helper("add")
# _register_external_op_helper("subtract")
# _register_external_op_helper("multiply")
_register_external_op_helper("qnn.batch_matmul")
_register_external_op_helper("qnn.dense")

def make_pattern(with_bias=True):
    data = wildcard()
    weight = wildcard()
    bias = wildcard()
    conv = is_op("nn.conv2d")(data, weight)
    if with_bias:
        conv_out = is_op("add")(conv, bias)
    else:
        conv_out = conv
    return is_op("nn.relu")(conv_out)


def make_pattern_qnn_conv2d(with_sum=False):
    weight = wildcard()
    bias = wildcard()

    # TODO(@apeskov): additional check applicability of this pattern
    pat = wildcard()
    pat = is_op("qnn.conv2d")(pat, weight, wildcard(), wildcard(), wildcard(), wildcard())
    pat = is_op("add")(pat, bias)
    pat = is_op("qnn.requantize")(pat, wildcard(), wildcard(), wildcard(), wildcard())
    pat = is_op("clip")(pat)
    pat = is_op("cast")(pat)
    if with_sum is True:
        pat = is_op("qnn.add")(pat, wildcard(), wildcard(), wildcard(), wildcard(),
                               wildcard(), wildcard(), wildcard())
        pat = is_op("clip")(pat)

    return pat


def make_pattern_qnn_dense(with_sum=False):
    weight = wildcard()
    bias = wildcard()

    # TODO(@apeskov): additional check applicability of this pattern
    pat = wildcard()
    pat = is_op("qnn.dense")(pat, weight, wildcard(), wildcard(), wildcard(), wildcard())
    pat = is_op("add")(pat, bias)
    pat = is_op("qnn.requantize")(pat, wildcard(), wildcard(), wildcard(), wildcard())
    pat = is_op("clip")(pat)
    pat = is_op("cast")(pat)
    if with_sum is True:
        pat = is_op("qnn.add")(pat, wildcard(), wildcard(), wildcard(), wildcard(),
                               wildcard(), wildcard(), wildcard())
        pat = is_op("clip")(pat)

    return pat

def make_pattern_qnn_batch_matmul():
    pat = wildcard()
    weight = wildcard()
    pat = is_op("nn.batch_matmul")(pat, weight, wildcard(), wildcard(), wildcard(), wildcard())
    return pat

def make_pattern_qnn_batch_matmul_reshape_dequantize():
    pat = wildcard()
    weight = wildcard()
    pat = is_op("qnn.batch_matmul")(pat, weight, wildcard(), wildcard(), wildcard(), wildcard())
    pat = is_op("reshape")(pat)
    pat = is_op("qnn.dequantize")(pat, wildcard(), wildcard())
    return pat

def make_pattern_gelu(inpt=None):
    if inpt == None:
        inpt = wildcard()
    div = is_op("divide")(inpt, is_constant())
    pat = is_op("erf")(div)
    mul = is_op("multiply")(inpt, is_constant())
    pat = is_op("add")(pat, is_expr(const(1, dtype="float32")))
    pat = is_op("multiply")(mul, pat)
    return pat

def make_pattern_qnn_dense_reshape_dequantize_add_quantize():
    pat = wildcard()
    weight = wildcard()
    bias = wildcard()
    pat = is_op("qnn.dense")(pat, weight, wildcard(), wildcard(), wildcard(), wildcard())
    pat = is_op("reshape")(pat)
    pat = is_op("qnn.dequantize")(pat, wildcard(), wildcard())
    biasVal = is_op("cast")(bias)
    pat = is_op("add")(pat, bias)
    pat = is_op("qnn.quantize")(pat, wildcard(), wildcard())
    return pat

def make_pattern_qnn_dense_reshape_dequantize():
    pat = wildcard()
    weight = wildcard()
    bias = wildcard()
    pat = is_op("qnn.dense")(pat, weight, wildcard(), wildcard(), wildcard(), wildcard())
    pat = is_op("reshape")(pat)
    pat = is_op("qnn.dequantize")(pat, wildcard(), wildcard())
    return pat

def  make_pattern_qnn_batch_matmul_reshape_requantize():
    pat     = wildcard()
    weights = wildcard()
    pat = is_op("qnn.batch_matmul")(pat, weights, wildcard(), wildcard(), wildcard(), wildcard())
    pat = is_op("reshape")(pat)
    pat = is_op("transpose")(pat)
    pat = is_op("reshape")(pat)
    pat = is_op("qnn.requantize")(pat, wildcard(), wildcard(), wildcard(), wildcard())
    return pat

def make_pattern_qnn_dense_reshape_dequantize_gelu():
    pat = wildcard()
    weight = wildcard()
    bias = wildcard()
    pat = is_op("qnn.dense")(pat, weight, wildcard(), wildcard(), wildcard(), wildcard())
    pat = is_op("reshape")(pat)
    pat = is_op("qnn.dequantize")(pat, wildcard(), wildcard())
    biasVal = is_op("cast")(bias)
    pat = is_op("add")(pat, bias)
    pat = make_pattern_gelu(pat)
    pat = is_op("qnn.quantize")(pat, wildcard(), wildcard())
    return pat

def make_pattern_batchnorm():
    pat  = wildcard()
    mean = is_op("mean")(pat)
    subt = is_op("subtract")(pat, mean)
    pow  = is_op("power")(subt, is_expr(const(2, dtype="float32")))
    pat  = is_op("mean")(pow)
    pat  = is_op("add")(pat, wildcard())
    pat  = is_op("sqrt")(pat)
    pat  = is_op("divide")(subt, pat)
    pat  = is_op("multiply")(pat, wildcard())
    pat  = is_op("add")(pat, wildcard())
    return pat

def make_pattern_softmax():
    pat = wildcard()
    max = is_op("max")(pat)
    pat = is_op("subtract")(pat, max)
    exp = is_op("exp")(pat)
    sum = is_op("sum")(exp)
    pat = is_op("divide")(exp, sum)
    return pat

@register_pattern_table("dnnl")
def pattern_table():
    conv2d_bias_relu_pat = ("dnnl.conv2d_bias_relu", make_pattern(with_bias=True))
    conv2d_relu_pat = ("dnnl.conv2d_relu", make_pattern(with_bias=False))
    conv2d_qnn_sum_pat = ("dnnl.qnn.conv2d_sum", make_pattern_qnn_conv2d(with_sum=True))
    conv2d_qnn_pat = ("dnnl.qnn.conv2d", make_pattern_qnn_conv2d())
    dense_qnn_pat = ("dnnl.qnn.dense", make_pattern_qnn_dense())
    batch_matmul_pat = ("dnnl.qnn.batch_matmul", make_pattern_qnn_batch_matmul())
    batch_matmul_reshape_dequantize_pat = ("dnnl.qnn.batch_matmul_dequantize", make_pattern_qnn_batch_matmul_reshape_dequantize())
    dense_reshape_dequantize_gelu_pat = ("dnnl.qnn.dense_dequantize_gelu", make_pattern_qnn_dense_reshape_dequantize_gelu())
    dense_reshape_dequantize_pat = ("dnnl.qnn.dense_dequantize_add_quantize", make_pattern_qnn_dense_reshape_dequantize_add_quantize())
    batch_matmul_reshape_requantize_pat = ("dnnl.qnn.batch_matmul_requantize", make_pattern_qnn_batch_matmul_reshape_requantize())
    gelu_pat = ("dnnl.qnn.gelu", make_pattern_gelu())
    batchnorm_pat = ("dnnl.qnn.batchnorm", make_pattern_batchnorm())
    softmax_pat   = ("dnnl.qnn.softmax", make_pattern_softmax())
    dense_dequantize_pat = ("dnnl.qnn.dense_dequantize", make_pattern_qnn_dense_reshape_dequantize())
    dnnl_patterns = [conv2d_bias_relu_pat,
                     conv2d_relu_pat,
                     conv2d_qnn_sum_pat,
                     conv2d_qnn_pat,
                    #  dense_qnn_pat,
                     dense_reshape_dequantize_gelu_pat,
                     #gelu_pat,
                    #  batch_matmul_reshape_dequantize_pat,
                    #  batch_matmul_reshape_requantize_pat,
                     batch_matmul_pat,
                     dense_reshape_dequantize_pat,
                     dense_dequantize_pat,
                    # batchnorm_pat,
                     softmax_pat
                    ]
    return dnnl_patterns

