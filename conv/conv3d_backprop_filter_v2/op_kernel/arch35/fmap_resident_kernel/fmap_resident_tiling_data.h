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
 * \file fmap_resident_tiling_data.h
 * \brief TilingData 场景镜像：结构定义单源在 kernel/host 共享文件
 *        conv3d_backprop_filter_v2_tiling_data.h（C1 结构性零漂移），本文件仅提供提取 helper。
 */
#ifndef FMAP_RESIDENT_TILING_DATA_H
#define FMAP_RESIDENT_TILING_DATA_H

#include "../conv3d_backprop_filter_v2/conv3d_backprop_filter_v2_tiling_data.h"

namespace AscendC {
namespace conv_bp_v2_kernel {
// fmap_resident 场景参数提取（mLoad/batchExtent 由 host 容量门产出：case1 (32,1) / case2 (128,8)）
struct FrTilingParams {
    uint32_t mLoad;
    uint32_t batchExtent;
};

__aicore__ inline FrTilingParams FrGetTilingParams(const Conv3DBackpropFilterV2TilingData* tilingData)
{
    FrTilingParams p;
    p.mLoad = tilingData->fmapResidentTiling.mLoad;
    p.batchExtent = tilingData->fmapResidentTiling.batchExtent;
    return p;
}
} // namespace conv_bp_v2_kernel
} // namespace AscendC

#endif // FMAP_RESIDENT_TILING_DATA_H
