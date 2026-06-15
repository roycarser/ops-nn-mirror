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

#include "conv3d_backprop_filter_v2_tiling_data.h"
#include "../conv3d_backprop/conv_bp_wino.h"

using namespace AscendC ;

template <typename SrcT,typename DstT, uint32_t WinoTilingFlag>
class Conv2dDwWinograd {
public:
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR dedy, GM_ADDR y, GM_ADDR workspace,
        conv_bp_v2_kernel::Conv3DBackpropFilterV2TilingData* tilingData)
    {
        x_ = reinterpret_cast<__gm__ SrcT*>(x);
        dy_ = reinterpret_cast<__gm__ SrcT*>(dedy);
        workspace_ = reinterpret_cast<__gm__ SrcT*>(workspace);
        y_ = reinterpret_cast<__gm__ DstT*>(y);
        tilingData_ = &tilingData->dwTiling;
    }

    __aicore__ inline void Process()
    {
        uint32_t batch = tilingData_->batch;
        uint32_t cout = tilingData_->cout;
        uint32_t cin = tilingData_->cin;
        uint32_t padH = tilingData_->padUp;
        uint32_t padW = tilingData_->padLeft;
        uint32_t fmapH = tilingData_->hi;
        uint32_t fmapW = tilingData_->wi;
        uint32_t dyH = tilingData_->ho;
        uint32_t dyW = tilingData_->wo;
        bool hf32 = tilingData_->hf32Flag;

        using TilingT = decltype( BuildTilingType());
        WinoFmapFwdTransformer<SrcT, TilingT> fmapFwd(x_, fmapH, fmapW, cin, padH, padW);
        WinoDyFwdTransformer<SrcT, TilingT> dyFwd(dy_, dyH, dyW, cout, 0, 0);
        WinoMMAD<SrcT, TilingT> winoMmad(hf32);

        uint32_t tileH = WinoDyFwdTransformer<SrcT, TilingT>::SlideWin::SrcLength2Tiles(dyH);
        uint32_t tileW = WinoDyFwdTransformer<SrcT, TilingT>::SlideWin::SrcLength2Tiles(dyW);

        ConvBackpropFilterWinograd<SrcT, DstT, TilingT> winograd(
            fmapFwd, dyFwd,
            workspace_, y_,
            winoMmad,
            tileH, tileW,
            batch);

        winograd.Init();
        winograd.IterateAll();
        winograd.End();
    }

private:
    static __aicore__ inline constexpr auto BuildTilingType()
    {
        constexpr uint32_t singleShapeCout = 64;
        constexpr uint32_t singleShapeCin = 64;
        constexpr uint32_t singleShapeTransformC = 16;
        constexpr uint32_t singleShapeResidentC = 32;
        constexpr uint32_t fwdBufCnt = 5;
        constexpr uint32_t invTransBufCnt = 4;
        constexpr uint32_t invTransCout = 8;

        //TODO set by dtype and tilingFlag
        constexpr uint32_t singleShapeTileH = 2;
        constexpr uint32_t singleShapeTileW = 32;

        return BlockConfig::Tiling<singleShapeCout,
            singleShapeCin,
            singleShapeTransformC / C0<SrcT>(),
            singleShapeTileH,
            singleShapeTileW,
            fwdBufCnt,
            singleShapeResidentC,
            BlockConfig::InputTensor::FMAP,
            invTransBufCnt,
            invTransCout>{};
    }

    __gm__ SrcT* x_ = nullptr;
    __gm__ SrcT* dy_ = nullptr;
    __gm__ SrcT* workspace_ = nullptr;
    __gm__ DstT* y_ = nullptr;
    AscendC::conv_bp_v2_kernel::TConv3DDwTiling* tilingData_ = nullptr;
};