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
 * \file fmap_resident_tile_ops.h
 * \brief impl 薄包装（design §1.3 Tile 层决策）：
 *   - 驻留装载：LoadToB1Dn2Nz AS-IS 调用（每平面一次，dnNum=1 形态——48/24 次，绕过
 *     LoadToB1/CalB1HiCopyParams GM 外壳；pad 交 load3d padList）
 *   - 直出 Fixpipe：引擎 GetTensorC 默认（LoadL0c2GmNormal dk 折叠 + nSize 恒真实值 32[HR-5]），
 *     场景侧仅构造 y 偏移（TileOffsetC 的 g 项经张量平移注入）
 *   T6 ctx 依赖：LoadToB1Dn2Nz 消费 ctx.{isSplitWo_, curSingleCoreDk_, bL1cin1CopyLen, hwI_, dhwI_,
 *   tiling_->wi/k0}；本场景 curSingleCoreDk_≡1、bL1cin1CopyLen≡32（FrCompute 现算，与现状
 *   CalOut2B1Params 结果等价——同名异义字段已在调用点显式设值，假复用防线）。
 */
#ifndef FMAP_RESIDENT_TILE_OPS_H
#define FMAP_RESIDENT_TILE_OPS_H

#include "fmap_resident_scheduler.h"
#include "../conv3d_backprop/impl/conv_bp_filter_sub_func.h"

namespace ConvolutionBackprop {
namespace FmapResident {

// 驻留组切片装载：组 g 全部 (batch×di) 平面一次装载，全程驻留（NEW ① 租约的装载动作）。
// 驻留布局 [dn=batch×di][hi×wi][32cin] = 现状单平面 Dn2Nz 布局的平面堆叠（2.1 §5.3 证据 4 同构）。
template <class Intf>
__aicore__ inline void ResidentLoadGroup(Intf* self, typename Intf::Config::ContextData& ctxData, uint64_t g,
                                         int32_t slot)
{
    auto& ctx = self->ctx;
    const auto* t = ctx.tiling_;
    const uint64_t hwI = ctx.hwI_;
    LocalTensor<typename Intf::SrcT> slice =
        ctxData.frCtx.residentBuf[static_cast<int64_t>(slot) * static_cast<int64_t>(ctxData.frCtx.groupSlice)];

    // LoadToB1Dn2Nz 消费字段前置（首组装载时 bL1cin1CopyLen 仍为 InitParamsPart2 归零值 → dValue=0
    // 触发 MTE 异常；此处显式钉死本场景常量值）
    self->ctx.bL1cin1CopyLen = FR_RES_CIN;
    self->ctx.curSingleCoreDk_ = 1;
    self->ctx.isSplitWo_ = false;

    ConvolutionBackpropFunc::Out2L1ScalarParams params; // isSplitKo_=false 分支不消费 params 字段
    for (int32_t b = 0; b < t->batch; ++b) {
        for (uint32_t d = 0; d < t->di; ++d) {
            // GM 平面基址（NCDHW fmap: [batch][cin][di][hi][wi]）
            const uint64_t gmOff = static_cast<uint64_t>(b) * t->cin * ctx.dhwI_ +
                                   g * FR_RES_CIN * ctx.dhwI_ + d * hwI;
            LocalTensor<typename Intf::SrcT> planeDst =
                slice[(b * t->di + d) * ctxData.frCtx.planeElems];
            ConvolutionBackpropFunc::LoadToB1Dn2Nz<Intf>(self, t->hi, gmOff, params, planeDst);
        }
    }
    ctxData.frCtx.curGroup = static_cast<int32_t>(g);
    ctxData.frCtx.curGroupSlot = slot;
    }

} // namespace FmapResident
} // namespace ConvolutionBackprop

#endif // FMAP_RESIDENT_TILE_OPS_H
