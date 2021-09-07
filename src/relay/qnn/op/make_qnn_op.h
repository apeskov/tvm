/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *
 * \file tvm/relay/op/make_op.h
 * \brief Header of internal operator functions
 * to assist in creating ops in C++
 */
#ifndef TVM_RELAY_QNN_OP_MAKE_QNN_OP_H_
#define TVM_RELAY_QNN_OP_MAKE_QNN_OP_H_

#include <tvm/relay/expr.h>
#include <tvm/relay/op.h>

namespace tvm {
namespace relay {
namespace qnn {

Expr MakeDequantize(Expr data, Expr input_scale, Expr input_zero_point, int axis);

}  // namespace qnn
}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_QNN_OP_MAKE_QNN_OP_H_
