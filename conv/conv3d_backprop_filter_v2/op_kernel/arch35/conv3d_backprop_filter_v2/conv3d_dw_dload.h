/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file conv3d_dw_dload.h
 * \brief conv3d dw DLoad 模板（d 轴 MTE2 载入）kernel 入口——Conv2DDwWinograd
 *        （conv2d_dw_winograd.h）同构形态：Init（GM + TilingData 装配）→ Process（整算）
 *        → End（串联类内部收尾）。tiling 映射（dwTiling → DLoadTiling）：
 *        singleShapeAligned16Cin ← baseN/hkwk（baseN = mmad N 轴 = cin×hkwk → 反解）、
 *        singleShapeAligned16Cout ← baseM（cout 块宽/M 侧）、kl0HoWo ← baseK（howo 窗宽/
 *        K 侧）、dout/ho/wo ← dwTiling 直取（howodout 权威值）、hf32Flag ← hf32Flag。
 *        blockNum 生产路径 0 = GetBlockNum()（串联类内部处理，AIV 侧 Process 直接跳出）
 */

#ifndef CONV3D_DW_DLOAD_H
#define CONV3D_DW_DLOAD_H

#include "conv3d_backprop_filter_v2_tiling_data.h"
#include "../conv3d_backprop/dload/conv_bp_dload.h"

using namespace AscendC;

template <typename SrcT>
class Conv3DDwDLoad {
public:
    __aicore__ inline void Init(GM_ADDR fmap, GM_ADDR dy, GM_ADDR y, GM_ADDR workspace,
                                const conv_bp_v2_kernel::Conv3DBackpropFilterV2TilingData* tilingData)
    {
        // dwTiling 即用即读（不设冗余 shape 成员）；cin = baseN/hkwk
        // （baseN = mmad N 轴 = cin×hkwk → 反解 cin）
        const conv_bp_v2_kernel::TConv3DDwTiling& dw = tilingData->dwTiling;
        config_.shape.hk = static_cast<uint16_t>(dw.hk);
        config_.shape.wk = static_cast<uint16_t>(dw.wk);
        config_.shape.dk = static_cast<uint16_t>(dw.dk);
        config_.shape.hPad = static_cast<uint16_t>(dw.padUp);
        config_.shape.wPad = static_cast<uint16_t>(dw.padLeft);
        config_.shape.dPad = static_cast<uint16_t>(dw.padFront);
        config_.shape.batch = dw.batch;
        config_.shape.cout = dw.cout;
        config_.shape.cin = dw.cin;
        config_.shape.din = dw.di;
        config_.shape.hin = dw.hi;
        config_.shape.win = dw.wi;
        // dout/ho/wo 直取上层 dwTiling（输出 shape 权威值，含 stride≠1 等上层语义）
        config_.shape.dout = dw.dout;
        config_.shape.hout = dw.ho;
        config_.shape.wout = dw.wo;
        // tiling 映射：baseM = mmad M 轴（cout 块宽）、baseK = howo 窗宽；
        // baseN = mmad N 轴 = cin×hkwk → cin 块宽 = baseN/hkwk（host 侧保证整除）
        config_.tiling.singleShapeAligned16Cin = static_cast<uint16_t>(dw.baseN / (dw.hk * dw.wk));
        config_.tiling.singleShapeAligned16Cout = static_cast<uint16_t>(dw.baseM);
        config_.tiling.kl0HoWo = static_cast<uint16_t>(dw.baseK);
        config_.tiling.hf32Flag = dw.hf32Flag != 0;

        fmap_ = reinterpret_cast<__gm__ SrcT*>(fmap);
        dy_ = reinterpret_cast<__gm__ SrcT*>(dy);
        y_ = reinterpret_cast<__gm__ SrcT*>(y);
    }

    __aicore__ inline void Process()
    {
        // 纯 Cube 模板：AIV 核直接跳出（引擎 if ASCEND_IS_AIV 先例形态——与串联类
        // Process 内判别同款双保险）
        if ASCEND_IS_AIV {
            return;
        }
        op_.Init(reinterpret_cast<GM_ADDR>(fmap_), reinterpret_cast<GM_ADDR>(dy_), config_);
        op_.Process(reinterpret_cast<GM_ADDR>(y_));
        op_.End();
    }

private:
    BpDLoad::ConvBackpropFilterDLoad<SrcT> op_;
    BpDLoad::DLoadConfig config_ = {};

    __gm__ SrcT* fmap_ = nullptr;
    __gm__ SrcT* dy_ = nullptr;
    __gm__ SrcT* y_ = nullptr;
};

#endif // CONV3D_DW_DLOAD_H
