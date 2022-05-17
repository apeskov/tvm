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
 * \file src/runtime/contrib/dnnl/dnnl_utils.cc
 * \brief Some DNNL specific utility functions
 */

#ifndef TVM_RUNTIME_CONTRIB_DNNL_DNNL_UTILS_H_
#define TVM_RUNTIME_CONTRIB_DNNL_DNNL_UTILS_H_

#include <algorithm>
#include <cstdint>
#include <ostream>
#include <string>
#include <type_traits>
#include <vector>

// TODO(@apeskov): Have to mute warning from dnnl headers.
//  -Wzero-as-null-pointer-constant and -Wdocumentation-unknown-command
#include <dnnl.hpp>

#include <tvm/runtime/data_type.h>

#include "../json/json_node.h"
#include "tvm/runtime/logging.h"

namespace tvm {
namespace runtime {
namespace contrib {

/*!
 * \brief Convert a DLPack data type to a DNNL data type.
 * \param dltype The DLPack data type.
 * \return The corresponding DNNL data type.
 */
dnnl::memory::data_type dtype_dl2dnnl(DLDataType dltype);

namespace utils {

/*! \brief Pretty printer util */
inline std::ostream& operator<<(std::ostream& o, const dnnl::memory::dims& dims) {
  o << "[";
  auto d = dims.begin();
  if (d != dims.end()) o << *d++;
  while (d != dims.end()) o << "," << *d++;
  o << "]";
  return o;
}

/*!
 * \brief Extract rank from layout string descriptor (num of capital letters).
 * Example: NCWHT4n8c -> 5, ACB8c4a -> 3
 */
inline int GetLayoutRank(const std::string& layout) {
  int rank = 0;
  for (auto it = layout.begin(); it != layout.end();) {
    auto start = it;
    while (std::isdigit(*it)) it++;
    if (start == it) rank++;  // no digits only letter
    it++;
  }
  return rank;
}

/*! \brief Define which logical dims ordering is default for particular layout string. */
inline std::string DefaultLogicLayoutFor(const std::string& layout) {
  int rank = GetLayoutRank(layout);
  static const std::vector<std::string> sparse_dims = {"W", "HW", "DHW"};
  if (layout.find("N") != std::string::npos) return "NC" + sparse_dims[rank - 3];
  if (layout.find("G") != std::string::npos) return "GOI" + sparse_dims[rank - 4];
  if (layout.find("O") != std::string::npos) return "OI" + sparse_dims[rank - 3];

  LOG(FATAL) << "Unknown layout " << layout << "There is no default scheme to handle it";
  return {};
}

/*! \brief Generator of dnnl format_tag for plain version of tensor with provided rank. */
inline static dnnl::memory::format_tag PlainLayout(uint32_t rank) {
  switch (rank) {
    case 0:
    case 1:
      return dnnl::memory::format_tag::a;
    case 2:
      return dnnl::memory::format_tag::ab;
    case 3:
      return dnnl::memory::format_tag::abc;
    case 4:
      return dnnl::memory::format_tag::abcd;
    case 5:
      return dnnl::memory::format_tag::abcde;
    case 6:
      return dnnl::memory::format_tag::abcdef;
    case 7:
      return dnnl::memory::format_tag::abcdefg;
    default:
      LOG(FATAL) << "Unsupported data tensor rank: " << rank;
      break;
  }
  return dnnl::memory::format_tag::undef;
}

/*! \brief Converter helper for data type objects */
inline static dnnl::memory::data_type Convert2Dnnl(DLDataType dtype) {
  if (dtype.code == DLDataTypeCode::kDLInt) {
    if (dtype.bits == 8) return dnnl::memory::data_type::s8;
    if (dtype.bits == 32) return dnnl::memory::data_type::s32;
  } else if (dtype.code == DLDataTypeCode::kDLUInt) {
    if (dtype.bits == 8) return dnnl::memory::data_type::u8;
  } else if (dtype.code == DLDataTypeCode::kDLFloat) {
    if (dtype.bits == 16) return dnnl::memory::data_type::f16;
    if (dtype.bits == 32) return dnnl::memory::data_type::f32;
  } else if (dtype.code == DLDataTypeCode::kDLBfloat) {
    if (dtype.bits == 16) return dnnl::memory::data_type::bf16;
  }
  LOG(FATAL) << "Data type is not supported";
  return dnnl::memory::data_type::undef;
}

/*! \brief Converter helper for shape objects */
inline static dnnl::memory::dims Convert2Dnnl(std::vector<int64_t> shape) {
  if (shape.empty()) return {1};  // DNNL scalar representation
  return shape;
}

/*! \brief Converter data type template arg to runtime object */
template <typename T>
inline dnnl::memory::data_type DnnlDType();

template <>
inline dnnl::memory::data_type DnnlDType<int>() {
  return dnnl::memory::data_type::s32;
}

template <>
inline dnnl::memory::data_type DnnlDType<float>() {
  return dnnl::memory::data_type::f32;
}

template <typename T, std::enable_if_t<std::is_integral<T>::value, int> = 0>
T AttrConvert(std::vector<std::string> val) {
  ICHECK_EQ(val.size(), 1);
  return std::stol(val[0]);
}

template <typename T, std::enable_if_t<std::is_floating_point<T>::value, int> = 0>
T AttrConvert(std::vector<std::string> val) {
  ICHECK_EQ(val.size(), 1);
  return std::stof(val[0]);
}

template <typename T, std::enable_if_t<std::is_same<T, std::string>::value, int> = 0>
T AttrConvert(std::vector<std::string> val) {
  ICHECK_EQ(val.size(), 1);
  return val[0];
}

template <typename T,
          std::enable_if_t<std::is_same<T, std::vector<typename T::value_type>>::value, int> = 0>
T AttrConvert(std::vector<std::string> val) {
  T res;
  for (const auto& el : val) res.push_back(AttrConvert<typename T::value_type>({el}));
  return res;
}

/*! \brief Attribute extractor helper. */
template <typename T>
const T GetAttr(const json::JSONGraphNode& node, std::string name,
                std::vector<std::string> def = {}) {
  auto attr = node.HasAttr(name) ? node.GetAttr<std::vector<std::string>>(name) : def;
  return AttrConvert<T>(attr);
}

}  // namespace utils
}  // namespace contrib
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_CONTRIB_DNNL_DNNL_UTILS_H_
