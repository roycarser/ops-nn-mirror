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
 * \file fmap_resident_block_ctx.h
 * \brief 块上下文：chunk 走位游标 + 驻留租约状态载体（字段集见 FrBlockCtx，common.h）。
 *        本文件提供 per-tile 解码结果与驻留平面寻址参数的聚合结构。
 */
#ifndef FMAP_RESIDENT_BLOCK_CTX_H
#define FMAP_RESIDENT_BLOCK_CTX_H

#include "fmap_resident_common.h"

namespace ConvolutionBackprop {

// 单 tile 解码结果：t = g×mCnt + m（N-major 全局序，design §3.2.1）
struct FrTileDesc {
    uint64_t g; // cin 组号 [0, nCnt)
    uint64_t m; // cout 块号 [0, mCnt)
};

// 驻留平面寻址参数（Compute 钩子内逐 batchDout 现算，S6 偏移三式之 offsetB 场景版）
struct FrPlaneAddr {
    int32_t batchIdx;
    int32_t doutIdx;
    int32_t dinReal;   // 真实 fmap din 平面号 ∈ [0, di)（pad 补偿后）
    uint64_t planeOff; // 驻留区内平面元素偏移 = (batchIdx×di + dinReal)×planeElems + groupBase
};

} // namespace ConvolutionBackprop

#endif // FMAP_RESIDENT_BLOCK_CTX_H
