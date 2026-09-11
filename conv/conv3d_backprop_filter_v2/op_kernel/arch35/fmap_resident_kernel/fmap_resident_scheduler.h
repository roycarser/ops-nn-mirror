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
 * \file fmap_resident_scheduler.h
 * \brief N 亲和 chunk 走位（NEW ②，design §3.2）。只算不搬（架构层职责契约）。
 *        全局 N-major tile 序 t = g×mCnt + m（同组 tile 连续 → fmap 驻留亲和）；
 *        chunk 连续段分配：核 c 认领 [⌊T·c/usedCore⌋, ⌊T·(c+1)/usedCore⌋)——首尾相接精确分割 [0,T)
 *        → 每 tile 恰一核 → 每输出元素恰一核一次写全（HR-2 证据 3 / HR-5 前提）。
 *        host 侧 CalcSpanMax 同式（契约 新5）。
 */
#ifndef FMAP_RESIDENT_SCHEDULER_H
#define FMAP_RESIDENT_SCHEDULER_H

#include "fmap_resident_block_ctx.h"

namespace ConvolutionBackprop {
namespace FmapResident {

// chunk 解码：填充 FrBlockCtx 走位字段。返回 false = 本核无 tile（超用核/余核）
template <typename SrcT>
__aicore__ inline bool DecodeChunk(FrBlockCtx<SrcT>& ctx, const AscendC::conv_bp_v2_kernel::TConv3DDwTiling* t,
                                   uint64_t blockIdx)
{
    ctx.mCnt = t->cout / FR_RES_M; // M 粒度 64（方案 0：FR_RES_M，host CalcSpanMax 同式）
    ctx.nCnt = t->cin / FR_RES_CIN;
    const uint64_t total = ctx.mCnt * ctx.nCnt;
    if (blockIdx >= t->usedCoreNum || total == 0) {
        return false;
    }
    ctx.tileStart = total * blockIdx / t->usedCoreNum;
    ctx.tileEnd = total * (blockIdx + 1) / t->usedCoreNum;
    return ctx.tileStart < ctx.tileEnd;
}

// tile 解码：t = g×mCnt + m
__aicore__ inline FrTileDesc DecodeTile(uint64_t tileIdx, uint64_t mCnt)
{
    FrTileDesc d;
    d.g = tileIdx / mCnt;
    d.m = tileIdx % mCnt;
    return d;
}

// offsetC 场景版（S6 偏移三式之一，tile 粒度）：y[cout,cin,dk,hk,wk] 直出基址
//   = m×64×cin×dkhkwk + g×(32×dkhkwk)（M 项 FR_RES_M 方案 0；g 项 FR_RES_CIN cin 侧不变）；
//   dk 段内偏移由引擎 GetTensorC（LoadL0c2GmNormal dk 折叠）承担。
__aicore__ inline uint64_t TileOffsetC(const FrTileDesc& tile, const AscendC::conv_bp_v2_kernel::TConv3DDwTiling* t)
{
    const uint64_t dkhkwk = static_cast<uint64_t>(t->dk) * t->hk * t->wk;
    return tile.m * FR_RES_M * t->cin * dkhkwk + tile.g * (FR_RES_CIN * dkhkwk);
}

// offsetB 场景版（驻留平面寻址）：din = dout×strideD + dk×dilationD − padFront（本域 s=d=1）
__aicore__ inline bool CalcPlaneAddr(FrPlaneAddr& addr, const AscendC::conv_bp_v2_kernel::TConv3DDwTiling* t,
                                     uint64_t batchDoutIdx, int32_t dkIdx, uint64_t groupBase, uint64_t planeElems)
{
    addr.batchIdx = static_cast<int32_t>(batchDoutIdx / t->dout);
    addr.doutIdx = static_cast<int32_t>(batchDoutIdx % t->dout);
    const int32_t dinPad = addr.doutIdx * t->strideD + dkIdx * t->dilationD; // pad 域平面号
    if (dinPad < t->padFront || dinPad >= t->di + t->padFront) {
        return false; // 全 pad 平面：跳过本 batchDout（对 Mmad 结果贡献为 0）
    }
    addr.dinReal = dinPad - t->padFront;
    addr.planeOff = groupBase +
                    (static_cast<uint64_t>(addr.batchIdx) * t->di + static_cast<uint64_t>(addr.dinReal)) * planeElems;
    return true;
}

} // namespace FmapResident
} // namespace ConvolutionBackprop

#endif // FMAP_RESIDENT_SCHEDULER_H
