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
 * \file conv2d_dw_winograd.h
 * \brief
 */

#ifndef CONV2D_BACKPROP_FILTER_WINOGRAD_H
#define CONV2D_BACKPROP_FILTER_WINOGRAD_H
#endif

#include "basic_api/kernel_basic_intf.h"
#include "conv3d_backprop_filter_v2_tiling_data.h"
#include "../conv3d_backprop/conv_bp_winograd.h"

using namespace AscendC ;

template <typename xType, typename dyType>
class Conv2dDwWinograd {
public:
    __aicore__ inline void Init(
        TPipe* pipe,
        GM_ADDR x,
        GM_ADDR dy,
        GM_ADDR dw,
        const conv_bp_v2_kernel::Conv2DBackpropFilterWinogradTilingData* tiling)
        : fmapTransformer(
            pipe, x,
            tiling->cin,
            tiling->hi,
            tiling->wi,
            tiling->padH,
            tiling->padW,
            tiling->singleShapeCin,
            tiling->singleShapeK)
    {
        this->dyGm_.SetGlobalBuffer(dy);
        this->dwGm_.SetGlobalBuffer(dw);
    }

    __aicore__ inline void Process()
    {
        typename WinoFmapTransformer<xType>::TileKIter kIter;
        fmapTransformer.Transform(0, 0, &kIter);

    }

private:
    WinoFmapTransformer<xType> fmapTransformer;
    GlobalTensor<dyType> dyGm_;
    GlobalTensor<float> dwGm_;
};