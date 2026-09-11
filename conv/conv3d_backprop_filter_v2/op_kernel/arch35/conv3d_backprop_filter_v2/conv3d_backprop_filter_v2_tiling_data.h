/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file conv3d_backprop_filter_v2_tiling_data.h
 * \brief
 */
#ifndef CONV3D_BACKPROP_FILTER_V2_TILING_DATA_H
#define CONV3D_BACKPROP_FILTER_V2_TILING_DATA_H

namespace AscendC {
namespace conv_bp_v2_kernel {
struct TConv3DDwTiling {
    uint32_t batch = 1;
    uint32_t cin = 1;
    uint32_t cout = 1;
    uint32_t cin1G = 1;
    uint32_t cout1G = 1;
    uint32_t dout = 1;
    uint32_t ho = 1;
    uint32_t wo = 1;
    uint32_t di = 1;
    uint32_t hi = 1;
    uint32_t wi = 1;
    uint32_t dk = 1;
    uint32_t hk = 1;
    uint32_t wk = 1;
    uint32_t group = 1;
    uint32_t realGroup = 1;
    uint32_t strideD = 1;
    uint32_t strideH = 1;
    uint32_t strideW = 1;
    uint32_t padFront = 1;
    uint32_t padBack = 1;
    uint32_t padUp = 1;
    uint32_t padDown = 1;
    uint32_t padLeft = 1;
    uint32_t padRight = 1;
    uint32_t dilationD = 1;
    uint32_t dilationH = 1;
    uint32_t dilationW = 1;
    uint32_t channelSize = 1;
    uint32_t al0Pbuffer = 1;
    uint32_t bl0Pbuffer = 1;
    uint32_t cl0Pbuffer = 1;
    uint32_t al1Pbuffer = 1;
    uint32_t bl1Pbuffer = 1;
    uint32_t baseM = 1;
    uint32_t baseK = 1;
    uint32_t baseN = 1;
    uint32_t m0 = 1;
    uint32_t k0 = 1;
    uint32_t n0 = 1;
    uint32_t stepKa = 1;
    uint32_t stepKb = 1;
    uint32_t iterateOrder = 1;
    uint32_t bl1Bound = 1;
    uint32_t al1Bound = 1;
    uint32_t hf32Flag = 1;
    uint32_t singleCoreDk = 1;
    uint32_t singleCoreGroup = 1;
    uint32_t singleCoreCout = 1;
    uint32_t singleCoreHo = 1;
    uint32_t splitWo = 128;
    uint32_t streamkType = 1;
    uint32_t usedCoreNum = 1;
    uint32_t singleCoreM = 1;
    uint32_t singleCoreN = 1;
    uint32_t singleCoreK = 1;
    uint64_t singleCoreBatch = 1;
    uint64_t singleCoreCin = 1;
    uint64_t singleCoreBatchDout = 1;
};

// fmap_resident 场景私有子结构（W4 尾部 append-only：既有 59 字段零位移，kb/binary 兼容）
struct TFmapResidentTiling {
    uint32_t mLoad = 32;       // dedy 单次载入 M 宽（默认值仅为结构体兜底——既有 UT 全量 tiling 期望串
                               // 按此默认值生成[P-10]；场景 host 容量门产出恒 RES_M=64[方案 0]并覆写）
    uint32_t batchExtent = 1;  // dedy 单次载入 batch 数（case1=1 / case2=8）
    uint32_t reserved0 = 0;    // 8 字节对齐预留
    uint32_t reserved1 = 0;
};

struct Conv3DBackpropFilterV2TilingData {
    TConv3DDwTiling dwTiling;
    TFmapResidentTiling fmapResidentTiling;
};
} // namespace conv_bp_v2_kernel
} // namespace AscendC
#endif // CONV3D_BACKPROP_FILTER_V2_TILING_DATA_H
