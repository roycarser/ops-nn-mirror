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
 * \file fmap_resident_pipeline.h
 * \brief 组装与入口（design §2.2 四阶段编排：Stage0/1/3 缺省，仅 Stage2 主计算）。
 *        Init：GM 绑定 + 引擎 Init（场景三钩子经 Config 扩展点生效）+ 全量 SetFmap/SetOutBackprop
 *        （场景地址计算基于全量张量 + curMIdx_，无需逐块张量切片）；
 *        Process：IterateAll（tile-chunk 驱动全部逻辑）+ End（引擎默认：FreeAllEvent + hf32 关闭）。
 */
#ifndef FMAP_RESIDENT_PIPELINE_H
#define FMAP_RESIDENT_PIPELINE_H

#include "fmap_resident_processor.h"
#include "../conv3d_backprop_filter_v2/conv3d_backprop_filter_v2.h"

namespace AscendC {

template <typename xType, int xFormat, typename dedyType, int dedyFormat, typename yType, int yFormat>
class Conv3dBpFmapResident {
public:
    __aicore__ inline Conv3dBpFmapResident(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR dedy, GM_ADDR y, GM_ADDR workSpace,
                                const conv_bp_v2_kernel::Conv3DBackpropFilterV2TilingData* tilingData)
    {
        xGm_.SetGlobalBuffer((__gm__ xType*)x);
        dedyGm_.SetGlobalBuffer((__gm__ dedyType*)dedy);
        yGm_.SetGlobalBuffer((__gm__ yType*)y);
        dw_.Init(&(tilingData->dwTiling));
        dw_.SetFmap(xGm_);
        dw_.SetOutBackprop(dedyGm_);
    }

    __aicore__ inline void Process()
    {
        if (block_idx >= dw_.ctx.tiling_->usedCoreNum) {
            return;
        }
        dw_.IterateAll(yGm_, 0); // → FrIterateAll（AIV 早退于钩子内）
        dw_.End();               // 引擎默认：FreeAllEvent + SetHF32Mode(false)
    }

private:
    static constexpr ConvolutionBackprop::CubeFormat xCubeFormat = GetFormat(xFormat);
    static constexpr ConvolutionBackprop::CubeFormat dedyCubeFormat = GetFormat(dedyFormat);
    static constexpr ConvolutionBackprop::CubeFormat yCubeFormat = GetFormat(yFormat);
    using xDwType = ConvolutionBackprop::ConvType<TPosition::GM, xCubeFormat, xType>;
    using filterSizeDwType = ConvolutionBackprop::ConvType<TPosition::GM, ConvolutionBackprop::CubeFormat::ND, int32_t>;
    using dedyDwType = ConvolutionBackprop::ConvType<TPosition::GM, dedyCubeFormat, dedyType>;
    using yDwType = ConvolutionBackprop::ConvType<TPosition::GM, yCubeFormat, yType>;
    static constexpr Conv3ddwConfig frConfig = {false, false}; // 非拆分/非扩维（本域 groups=1）
    ConvolutionBackprop::Conv3dBpFmapResidentEngine<xDwType, filterSizeDwType, dedyDwType, yDwType, frConfig> dw_;
    GlobalTensor<xType> xGm_;
    GlobalTensor<dedyType> dedyGm_;
    GlobalTensor<yType> yGm_;
};

} // namespace AscendC

#endif // FMAP_RESIDENT_PIPELINE_H
