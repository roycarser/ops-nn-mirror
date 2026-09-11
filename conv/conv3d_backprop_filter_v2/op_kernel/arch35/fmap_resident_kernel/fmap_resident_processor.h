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
 * \file fmap_resident_processor.h
 * \brief 三钩子实现体（common.h Cfg 委托至此）：
 *   FrInitTque  —— L1 三区布局（B1 驻留 2-slice @0 → A1 ping/pong；C3 新契约与 host 容量门同式两份）
 *   FrCompute   —— ComputeNormal 场景化：batchDout 循环（K 全载单 L0C 链）+ 驻留平面选择
 *                  （cacheB1BufPing_ = 驻留切片视图，bL1 窗口参数逐 batchDout 现算）+ 引擎
 *                  ComputeLoop（A1 GM 流式/装载 A1→L0A→Mmad 九步流水全继承；B1 GM→L1 路径不触发）
 *   FrIterateAll —— tile-chunk 循环 + 驻留租约 + 逐 dk 段 Compute/GetTensorC 直出
 */
#ifndef FMAP_RESIDENT_PROCESSOR_H
#define FMAP_RESIDENT_PROCESSOR_H

#include "fmap_resident_tile_ops.h"
#include "../conv3d_backprop/impl/conv_bp_filter_sub_func.h"

namespace ConvolutionBackprop {
namespace FmapResident {

constexpr uint32_t L1_ALIGN_32 = 32;

// ---------------- Init 钩子实现：L1 布局（C3，与 host CheckCapacity 同式） ----------------
// 布局（A1 前置——引擎 InitTque 硬件惯例：A1 必须位于 L1 前部 bank 区，B1 驻留区放后）：
//   [A1 ping][A1 pong][b1Ping_ 清零通道][B1 驻留区 2×groupSliceBytes(raw)]
// al1Pbuffer=2（A1 流式乒乓）：引擎 LoadToA1 按 (curMKL1Idx+kaStepIdx+1)&1 交替 ping/pong，
// End() 亦触碰 a1Pong_——两侧都必须 InitBuffer（实测教训：漏配 pong = AllocTensor 于未初始化
// TQue → 野地址，P-12）
template <class Intf>
__aicore__ inline void FrInitTque(Intf* self)
{
    auto& ctx = self->ctx;
    auto& frCtx = self->ctx.frCtx;
    const auto* t = ctx.tiling_;

    if ASCEND_IS_AIV {
        // AIV 分支对齐引擎默认 InitTque（3510）：vecBuf_ InitBuffer 使 pipe_ 底层资源就绪——
        // 缺失时 AIV 侧 End()->FreeAllEvent() 触碰未初始化 TQue → aivec trap（实测教训）
        ctx.pipe_.InitBuffer(ctx.vecBuf_, AscendC::TOTAL_UB_SIZE);
        return;
    }
    if ASCEND_IS_AIC {
        frCtx.planeElems = ctx.hwI_ * FR_RES_CIN;
        frCtx.groupSlice = static_cast<uint64_t>(t->batch) * t->di * frCtx.planeElems;

        // a1Bytes = al1Bound×4：host al1Bound=FR_RES_M×stepKa×baseK（方案 0 M=64）自动跟随，此处零改动
        const uint32_t a1Bytes = static_cast<uint32_t>(ctx.tiling_->al1Bound * sizeof(typename Intf::SrcT));
        const uint32_t a1BytesAligned = (a1Bytes + L1_ALIGN_32 - 1) / L1_ALIGN_32 * L1_ALIGN_32;
        // 清零通道（ClearBaseMNL0C 全 pad dk 段零填充路径 + End FreeAllEvent 兼容）：[baseN×k0] fp32
        const uint32_t b1ClearBytes = static_cast<uint32_t>(ctx.tiling_->baseN * ctx.tiling_->k0 *
                                                            sizeof(typename Intf::SrcT));
        const uint32_t b1ClearAligned = (b1ClearBytes + L1_ALIGN_32 - 1) / L1_ALIGN_32 * L1_ALIGN_32;

        uint32_t offset = 0;
        ctx.pipe_.InitBuffer(ctx.a1Ping_, Std::make_tuple(offset, a1BytesAligned)); // A1 ping @ L1 头部
        offset += a1BytesAligned;
        if (ctx.tiling_->al1Pbuffer > 1) {
            ctx.pipe_.InitBuffer(ctx.a1Pong_, Std::make_tuple(offset, a1BytesAligned)); // A1 pong 紧随
            offset += a1BytesAligned;
        }
        ctx.pipe_.InitBuffer(ctx.b1Ping_, Std::make_tuple(offset, b1ClearAligned));
        offset += b1ClearAligned;
        // B1 驻留区（TQue 显式地址 + AllocTensor 单次持久租约——设计偏差声明：raw LocalTensor
        // [dx SmallKernel 先例]设备侧合法，但 CPU mock 模型内存不合法[load3d padUp 窗口读注册
        // 区外 SIGSEGV，2026-09-10 实测]；TQue 载体设备语义等价[引擎 a1Ping_ 同款]，双槽位）
        const uint64_t residentElems = frCtx.groupSlice * 2;
        const uint32_t residentBytes = static_cast<uint32_t>(residentElems * sizeof(typename Intf::SrcT));
        ctx.pipe_.InitBuffer(frCtx.residentQue_, Std::make_tuple(offset, residentBytes));
        frCtx.residentBuf = frCtx.residentQue_.template AllocTensor<typename Intf::SrcT>();

        const uint32_t cMatrixByteSize = ctx.baseMN_ * sizeof(typename Intf::L0cT);
        ctx.pipe_.InitBuffer(ctx.l0cPing_, 1, cMatrixByteSize);
        ctx.pipe_.InitBuffer(ctx.l0cPong_, 1, cMatrixByteSize); // cl0Pbuffer=2：dk 段间 Fixpipe/Mmad 重叠
        ctx.pipe_.InitBuffer(ctx.l0aBuf_, TOTAL_L0A_SIZE);
        ctx.pipe_.InitBuffer(ctx.l0bBuf_, TOTAL_L0B_SIZE);
    }
}

// ---------------- Compute 钩子实现：batchDout 循环 + 驻留平面选择 + 引擎 ComputeLoop ----------------
template <class Intf>
__aicore__ inline void FrCompute(Intf* self, ConvolutionBackpropFunc::Out2L1ScalarParams& out2L1Params)
{
    if ASCEND_IS_AIV {
        return; // VecRole=None 早退（AIC 侧零跨核 Wait → 安全前提 2.1 §7.1）
    }
    auto& ctx = self->ctx;
    auto& frCtx = ctx.frCtx;
    const auto* t = ctx.tiling_;
    LocalTensor<typename Intf::SrcT> l0a;
    LocalTensor<typename Intf::SrcT> l0b;
    LocalTensor<typename Intf::L0cT> l0c;

    // ---- ComputeInit 场景版（引擎 ComputeInit 的驻留改造：isSplitKernelHW 恒 false 不清 L0C） ----
    if (ctx.l0cPingPongFlag_) {
        l0c = ctx.l0cPing_.template AllocTensor<typename Intf::L0cT>();
    } else {
        l0c = ctx.l0cPong_.template AllocTensor<typename Intf::L0cT>();
    }
    // 引擎 Compute::call 入场语义（conv_bp_func_arch35.h:117-121 逐字对齐）：baseUseM_/baseUseN_
    // 逐 tile 现算。GetTensorC 在 Compute 返回后消费 baseUseM_（Fixpipe dnNum/srcStride）——
    // 若不设或 save/restore 恢复成 InitParams 的 0，Fixpipe dnNum=0/srcStride=0 退化 →
    // L0C→GM 野写（P-12 真根因；本场景 M/N 均单块恒满：baseUseM_=tailM_=64[方案 0 FR_RES_M]/
    // baseUseN_=tailN_=288）
    ctx.baseUseM_ = (ctx.curMIdx_ + 1 == ctx.mIter_) ? ctx.tailM_ : ctx.tiling_->baseM;
    ctx.baseUseN_ = ((ctx.curNIdx_ + 1) % ctx.cinHkWkLoop_ == 0) ? ctx.tailN_ : ctx.tiling_->baseN;
    const uint32_t baseUseMBak = ctx.baseUseM_;
    ConvolutionBackpropFunc::CalcParamsL12L0a<Intf>(self);
    ConvolutionBackpropFunc::CalcParamsL12L0b<Intf>(self);
    ConvolutionBackpropFunc::CalcParamsMmad<Intf>(self);

    // ---- 场景参数（引擎 CalOut2L1ScalarParams 的 B 侧分支在 isLoad2L1B=false 下不执行，此处等值补齐） ----
    out2L1Params.isLoad2L1A = true;  // A1 GM 流式（引擎 ComputeLoadToA1 路径原样）
    out2L1Params.isFreeAL1 = true;
    out2L1Params.isLoad2L1B = false; // B1 驻留：GM→L1 路径不触发，ExecuteMTE1L0b 不 DeQue/不 Free
    out2L1Params.isFreeBL1 = false;
    out2L1Params.isLastMAL1 = (ctx.mIter_ - 1) == ctx.curMIdx_;
    if constexpr (Intf::Config::cType::format == ConvolutionBackprop::CubeFormat::NCDHW) {
        out2L1Params.out2A1SrcAddr = static_cast<uint64_t>(ctx.curMIdx_) * ctx.tiling_->baseM *
                                     ctx.tiling_->dout * ctx.hwO_;
    } else {
        out2L1Params.out2A1SrcAddr = static_cast<uint64_t>(ctx.curMIdx_) * ctx.tiling_->baseM;
    }
    int32_t woIterateTimes = 1;
    ConvolutionBackpropFunc::calculateWoIterTimes<Intf>(self, woIterateTimes, ctx.tiling_->splitWo);
    ConvolutionBackpropFunc::updateSingleShapeWoI<Intf>(self, out2L1Params, 1, 0, ctx.tiling_->splitWo);
    const uint64_t singleShapeHi = ctx.singleShapeHo_ * ctx.tiling_->strideH + ctx.strideKernelDilationH;
    out2L1Params.singleShapeHi = ctx.tiling_->hi > singleShapeHi ? singleShapeHi : ctx.tiling_->hi;
    ctx.bL1cin1CopyLen = ctx.singleShapeCin_; // = 32（与引擎 CalOut2B1Params 在本折叠下的结果等值）

    // ---- batchDout 循环：K 全载（batch×dout 全走完，单 L0C 累加链 → bit 级确定性，HR-2 §8.1） ----
    bool isFirstMmad = true;
    const uint64_t batchDoutEnd = ctx.batchDoutStartIdx_ + ctx.singleShapeBatch_;
    for (uint64_t bd = ctx.batchDoutStartIdx_; bd < batchDoutEnd; bd++) {
        bool skipCurrentDinCompute = false;
        ConvolutionBackpropFunc::UpdateSrcAddrBaseOnBatchDoutIdx<Intf>(self, bd, out2L1Params,
                                                                       skipCurrentDinCompute);
        if (skipCurrentDinCompute) {
            continue; // 全 pad din 平面：贡献为 0，跳过（游标/地址照常推进——引擎纪律）
        }
        // 驻留平面选择（load3d 无 D 轴参数的根因对策：D 轴选择落在 L1 源平面偏移）
        FrPlaneAddr addr;
        if (!FmapResident::CalcPlaneAddr(addr, t, bd, ctx.dkStartIdx_,
                                         static_cast<uint64_t>(frCtx.curGroupSlot) * frCtx.groupSlice,
                                         frCtx.planeElems)) {
            continue;
        }
        // B 窗口参数：驻留平面为整 fmap din 平面 [hi][wi][32cin]（ResidentLoadGroup 装载粒度），
        // load3d 窗口须覆盖全平面行且 padUp=fmap H 向前 pad（streamK CalB1HiCopyParams 同语义：
        // 全 K 窗 b1SrcHiGm=-padUp<0 → padUp=padUp/l1H=hi）——padUp=0 会使窗口行错位一格，
        // 边界窗读到 NZ 分形未写尾部（nValue=36 非整分形，n=36..47 未初始化）→ L0B 垃圾
        ctx.bL1HiCopyLenPing = static_cast<uint32_t>(t->hi);
        ctx.bL1PadUpPing = t->padUp;
        ctx.cacheB1BufPing_ = frCtx.residentBuf[static_cast<int64_t>(addr.planeOff)];

        // 引擎九步流水（k 循环 + A1 GM 流式 + L0A/L0B 装载 + Mmad；B1 GM 装载路径不触发）
        ConvolutionBackpropFunc::ComputeLoop<Intf>(self, out2L1Params, l0a, l0b, l0c, isFirstMmad, 0, 0, 0);
    }
    if (isFirstMmad) {
        // 全 dk 段全 pad：零填充（y 该 dk 段必须为 0 而非脏数据；b1Ping_/a1Ping_ 清零通道已 InitBuffer）
        ConvolutionBackpropFunc::ClearBaseMNL0C<Intf>(self, l0c);
    }
    ctx.baseUseM_ = baseUseMBak;
    if (ctx.l0cPingPongFlag_) {
        ctx.l0cPing_.EnQue(l0c);
    } else {
        ctx.l0cPong_.EnQue(l0c);
    }
}

// ---------------- IterateAll 钩子实现：tile-chunk 循环 + 驻留租约 + 逐 dk 段直出 ----------------
template <class Intf>
__aicore__ inline void FrIterateAll(Intf* self, const GlobalTensor<typename Intf::DstT>& output, uint8_t enAtomic)
{
    if ASCEND_IS_AIV {
        return;
    }
    auto& ctx = self->ctx;
    auto& frCtx = ctx.frCtx;
    const auto* t = ctx.tiling_;

    if (!FmapResident::DecodeChunk(frCtx, t, block_idx)) {
        return; // 超用核/余核（dw End 无跨核 Wait，直接返回安全——MNStreamK :57-60 先例）
    }
    enAtomic = 0; // 全覆写直出（每元素恰一核一写，HR-2 证据 4/HR-5）
    // dedy 基线锚定（真根因#3 修复）：SetOutBackprop 覆写 ctx.outBackPropGlobal_——逐 tile 平移
    // 必须从本基线出发（原实现 ctx.outBackPropGlobal_[m×…] 在已平移张量上累加 → 累计漂移 →
    // batch 步进叠加后 A 侧 GM 越界；块内首 tile 前的 ctx 值即 Init 传入的原始张量）
    frCtx.dedyBase = ctx.outBackPropGlobal_;

    for (uint64_t tt = frCtx.tileStart; tt < frCtx.tileEnd; tt++) {
        const FrTileDesc tile = FmapResident::DecodeTile(tt, frCtx.mCnt);
        // 驻留租约：组切换时装载新槽位（chunk 内组号单调 → 至多一次；首组装 slot 0）
        if (static_cast<int32_t>(tile.g) != frCtx.curGroup) {
            const int32_t slot = (frCtx.curGroupSlot == -1) ? 0 : (frCtx.curGroupSlot == 0 ? 1 : 0);
            FmapResident::ResidentLoadGroup<Intf>(self, ctx, tile.g, slot);
        }
        ctx.curMIdx_ = 0; // tile M=64=baseM 单块（mIter_=1）；tile.m 项经 y/dedy 基址注入
        ctx.curNIdx_ = 0; // N=288=baseN 单段；g/dk 项经 y 基址注入
        // dedy tile.m 段切片（NCDHW cout 段）：A1 装载地址经张量平移携带 m 项（FR_RES_M=64，方案 0），
        // out2A1SrcAddr 保持相对 0（真根因#3：平移基线化——从 frCtx.dedyBase 出发而非
        // ctx.outBackPropGlobal_ 自身）
        self->SetOutBackprop(frCtx.dedyBase[static_cast<int64_t>(tile.m * FR_RES_M * t->dout *
                                                                 t->ho * t->wo)]);
        // 段形状（引擎语义对齐 streamK 单基本块）：M=64（FR_RES_M）；N=32cin×hkwk=288（单 dk 段宽）；
        // K=ho×wo（单 batchDout 窗口）；batch=batch×dout（K 全载：FrCompute 内全 batchDout 循环）
        self->SetSingleShape(FR_RES_M, static_cast<uint64_t>(FR_RES_CIN) * t->hk * t->wk,
                             static_cast<uint64_t>(t->ho) * t->wo, FR_RES_CIN,
                             static_cast<uint64_t>(t->batch) * t->dout);
        // y 段基址（S6 offsetC tile 项）：m×64×cin×dkhkwk + g×(32×dkhkwk)；dk 项逐段累加
        const uint64_t dkhkwk = static_cast<uint64_t>(t->dk) * t->hk * t->wk;
        int64_t dstBase = static_cast<int64_t>(tile.m * FR_RES_M * t->cin * dkhkwk +
                                               tile.g * (FR_RES_CIN * dkhkwk));
        for (uint32_t dkIdx = 0; dkIdx < t->dk; ++dkIdx) { // dk 外层循环（逐段直出）
            self->SetStartIdx(0, 0, static_cast<int32_t>(dkIdx)); // batchDout 起点=0，dkStartIdx_=本段
            ConvolutionBackpropFunc::Out2L1ScalarParams params;
            self->Compute(params); // → FrCompute（K 全载单链累加至 L0C [32,288]）
            // 直出：引擎 GetTensorC（nSize=32 真实值 [HR-5]；Fixpipe 内部偏移项 curMIdx_/curNIdx_ 均为 0）
            self->GetTensorC(output[dstBase + static_cast<int64_t>(dkIdx * t->hk * t->wk)], enAtomic);
        }
        ctx.isFirstIter_ = true;
    }
}

} // namespace FmapResident
} // namespace ConvolutionBackprop

#endif // FMAP_RESIDENT_PROCESSOR_H
