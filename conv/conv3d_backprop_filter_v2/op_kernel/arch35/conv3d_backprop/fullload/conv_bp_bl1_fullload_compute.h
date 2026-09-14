/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software and/or modify it under the terms of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file conv_bp_bl1_fullload_compute.h
 * \brief BL1 全载 L0 搬运与 Mmad 计算循环（板测版：camodel 30/30 case 验证终态语义）
 *
 * ★语义结论（cannsim/camodel 全量回归钉死，过程记录见 .cannbot/.../sim/验证记录.md）：
 *   S1. fp32 L0A/L0B/L0C 线性布局（M=mmad.m, N=mmad.n, K=mmad.k）：
 *       addrA(m,k) = (k/8)*(8M) + 8m + (k%8)        // L0A: [k/8 块][m(步长8)][k%8]
 *       addrB(k,n) = (k/8)*(8N) + 8n + (k%8)        // L0B: [k/8 块][n(步长8)][k%8]
 *       addrC(m,n) = (n/16)*(16M) + 16m + (n%16)    // L0C: [n/16 块][m(步长16)][n%16]
 *   S2. B1(fmap) 合轴驻留 [batch][c1g][d][h][w][c0]（C1 外 D 内，dhw 合轴连续）：
 *       addr(b,ci,d,hw) = b*(dhwin*alignedCin) + (ci/C0)*(dhwin*C0) + d*(hwIn*C0) + hw*C0 + (ci%C0)
 *       LoadL1Fmap = 单条 Dn2Nz{dnNum=batch, dValue=cinLength, nValue=dhwin, …} 全载驻留，
 *       loadedCinIdx_ 缓存使 B1 仅在 cin 块切换时重装
 *   S3. A1(dy) 合轴驻留 [batch][co1g][dhowo][co0]（D 在 C1 内，dhowo 合轴连续）：
 *       addr(b,co,pt) = b*(dhowo*alignedCout) + (co/C0)*(dhowo*C0) + pt*C0 + (co%C0)
 *       LoadA1Dy = 每 batch 一条 Dn2Nz{dnNum=co1g 段数, dValue=C0}（dn 维堆叠承载 co1g 段距）
 *   S4. load3d(B1→L0B) 合轴 tile（第十一轮）：每 (窗,dk,tap) 一条命令（kExt=C0、
 *       kStartPt=tap*C0）+ rt=2*cin16G（配对 (2g,2g+1) 落块 g、+16 槽连续铺块）、
 *       rs=din*hkwk、ds=nExt/16、dst=l0b[128*(cin16G*(dk*hkwk+tap))]（含 dk 段偏移）；
 *       块序 B = cin16G*(dk*hkwk+tap) + g → 与 L0C 全量 dhw n 序一致（S6/S8）
 *   S5. load2d(A1→L0A) 单命令大范围转置（winograd fp32 先例参数式）：mStep=Ceil(窗宽,16)、
 *       kStep=M/8（偶数，M 恒 16 对齐）、srcStride=A1 行距/16（m1 组距，单位 128 元素；A1
 *       howo 行距 16 对齐 padding 保证整除）、dstStride=kStep/2；一条命令覆盖整窗 [M][16]
 *   S6. Mmad（第十一轮 dk 内移）：A(m=co,k=howo) × B(k=howo,n=ci 展开) → C(m=co,n)；
 *       m=alignedCout、n=段级 nExt=16*cin16G*hkwk、k=howoLen；dk 段 L0C/L0B n 区不相交
 *       （段基址 16M*cin16G*dk*hkwk 元素），dk 间免背压；cmatrixInitVal=每 dk 段首 Mmad
 *       true（段首=批0/该dk首有效dout/窗0）；garbage n0 槽同 S6 旧例（fixpipe nSize
 *       只读真实行）
 *   S7. Mmad 目的必须是 TPosition::CO1；LocalTensor(pos, offset, size) 的 size 单位是
 *       【元素】（offset 是字节）
 *   S8. 直出：fixpipe L0C→GM 直达，NZ2DN（COLUMN_MAJOR，引擎 LoadL0c2GmNormal 先例）：
 *       src = 16*MS*dn + 16*CS*n + 16*SS*(d/16) + d%16（C0 单位）、
 *       dst = DM*dn + n + DS*d（元素单位）；dnNum=coutLength、mSize=hkwk、nSize=cinLength
 *       （均真实值）、MS=1、CS=alignedCout、SS=alignedCout*hkwk、DM=cinTotal*dhwK、DS=dhwK、
 *       dk 列偏移 = dk*hkwK。y 输出 = 生产布局 y[co][ci][dhwK] 行主序（=dw 折叠）
 *   S9. 事件链（winograd WinoMMAD 纪律，手动形态）：成对 id（前向就绪/释放背压）不复用、
 *       释放向 Init 预置位、前向同迭代 set-then-wait、End 消费残留（Set/Wait 计数配平）。
 *       MTE2→MTE1（驻留装载一次性）、MTE1→M + M→MTE1（L0A/L0B 半区前向+背压，id=bool
 *       0/1）、M→FIX + FIX→M（L0C 半区前向+背压）、MTE1→MTE2（跨块 B1/A1 装载背压，
 *       TQue Alloc/Free 手动等价：Init 预置位/装载前 Wait/End Set/入口 Drain 消费）。★事件链每段恒定（含空段），守卫只落
 *       在数据操作上；★dk 全 pad 空段跳过 fixpipe 本体（y 保持 host 清零值，golden=0）
 */

#ifndef CONV_BP_BL1_FULLLOAD_COMPUTE_H
#define CONV_BP_BL1_FULLLOAD_COMPUTE_H

#include <cstdint>
#include "basic_api/kernel_basic_intf.h"
#include "op_kernel/math_util.h"
#include "utils/std/algorithm.h"
#include "../util/conv_bp_util.h"
#include "conv_bp_bl1_fullload_config.h"

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#include "basic_api/kernel_struct_fixpipe.h"
#endif

namespace BpFullLoad {

using namespace AscendC;
using BpUtils::CoutCinRange;


template <typename SrcT>
class BL1FullLoadCompute {
public:
    // GM 张量入口（fmap = 正向输入 NCDHW [batch][cin][din][hin][win]，dy = 反向梯度 NCDHW
    // [batch][cout][dout][hout][wout]）
    __aicore__ inline void Init(const GlobalTensor<SrcT>& fmap, const GlobalTensor<SrcT>& dy)
    {
        fmap_ = fmap;
        dy_ = dy;
        loadedCinIdx_ = 0xFFFFFFFF;
        loadedCinLength_ = 0;
        // 跨块/跨半区背压预置位（框架 InitBuffer:193 同款 + winograd 释放向预置位先例）：
        //   MTE1_MTE2(0)：跨块 B1 覆写背压四件套（End Set + 下块 Wait + Drain 消费）
        //   FIX_M(0)：首块 Mmad 前的 fixpipe 等待
        //   M_MTE1(0-3)：L0A/L0B 四半区释放链首用免等（第十二轮修改1：Wait 前置装载前、
        //     Mmad 后只 Set——末次 Set 无消费者，Drain 补消费）
        //   MTE1_MTE2(1/2)：A1 ping/pong 半区释放链首用免等（第十二轮续改动1：MTE1→MTE2
        //     方向——Mmad 未完但 L1→L0 搬完即可覆写，不等 Mmad 执行完）
        SetFlag<HardEvent::MTE1_MTE2>(0);
        SetFlag<HardEvent::FIX_M>(0);
        SetFlag<HardEvent::M_MTE1>(0);
        SetFlag<HardEvent::M_MTE1>(1);
        SetFlag<HardEvent::M_MTE1>(2);
        SetFlag<HardEvent::M_MTE1>(3);
        SetFlag<HardEvent::MTE1_MTE2>(1);
        SetFlag<HardEvent::MTE1_MTE2>(2);
    }

    // kernel 入口在全部块迭代结束后调用：消费跨块背压的残留 Set（TQue Reset 残留
    // freeBufEvt 消费先例）——漏掉则残留 flag 会污染同核后续 kernel 的首块装载
    __aicore__ inline void Drain()
    {
        // 消费残留 Set（Set/Wait 全程配平铁律，防跨 kernel 污染同核后续算子）：
        //   MTE1_MTE2(0)：末块 MTE1 排空信号 | FIX_M(0)：末块 fixpipe 释放
        //   M_MTE1(0-3)：各 L0A/L0B 半区末次释放（kernel 内无后续装载消费）
        //   MTE1_MTE2(1/2)：A1 各半区末次释放（同上；从未使用的半区消费 Init 预置位）
        WaitFlag<HardEvent::MTE1_MTE2>(0);
        WaitFlag<HardEvent::FIX_M>(0);
        WaitFlag<HardEvent::M_MTE1>(0);
        WaitFlag<HardEvent::M_MTE1>(1);
        WaitFlag<HardEvent::M_MTE1>(2);
        WaitFlag<HardEvent::M_MTE1>(3);
        WaitFlag<HardEvent::MTE1_MTE2>(1);
        WaitFlag<HardEvent::MTE1_MTE2>(2);
    }

    // 单个基本块 [coutRange × cinRange] 的 K 全载计算（第十一轮结构 + 第十二轮装载改造）：
    // L0C 全 dhw 驻留单视图（n 轴 [dk][hkwk][cin16] 块序），dk 内移循环 batch×dout×howo16 窗
    // → dk 段内累加（cmatrixInitVal 段首清零），块尾 fixpipe NZ2DN 整块一次直出。
    // 第十二轮：dy 单 batch 搬运 + A1 L1 ping-pong + M_MTE1 释放向引擎模式（Wait 前置）。
    //
    // y 布局契约（生产布局）：
    //   y = [cout][cin][dhwK] ND 行主序（全局），dhwK = dk*hwK
    //   本块基址 yBase = coutIdx*cinTotal*dhwK + cinIdx*dhwK（由入口计算）
    //   元素 y[yBase + co*cinTotal*dhwK + ci*dhwK + dk*hwK + tap] = C(co, ci, tap) of 本 dk 段
    __aicore__ inline void IterateK(const BL1FullLoadConfig& config, const CoutCinRange& cRange,
                                    GlobalTensor<SrcT>& y, uint64_t yBase)
    {
        const ShapeAttribute& shape = config.shape;
        const uint32_t alignedCout = config.tiling.singleShapeAlignedCout;
        const uint32_t alignedCin = config.tiling.singleShapeFullLoadAlignedCin;
        const uint32_t hkwk = shape.hk * shape.wk;
        const uint32_t dhwk = shape.dk * hkwk;
        const uint32_t cin16G = (cRange.cinLength + 15) / 16;
        // 第十一轮：L0C 保留完整 dhw——n 轴块序 [dk][hkwk][cin16]，单视图无 ping-pong。
        // 段级 N（Mmad n 参数）= 16·cin16G·hkwk；块级全 N = 16·cin16G·dhwk
        const uint32_t nL0c = 16 * cin16G * hkwk;
        const uint32_t nFull = 16 * cin16G * dhwk;

        // ★howo 窗宽（k 轴，第十二轮续改动4）：从 tiling.kl0HoWo 传入，不再写死 16。
        // 约束：kl0HoWo 须为 16 的倍数（load3d mStartPt = 窗起点须 16 对齐）
        const uint32_t howoWin = config.tiling.kl0HoWo;
        ASCENDC_ASSERT(howoWin != 0 && howoWin % 16 == 0, {
            KERNEL_LOG(KERNEL_ERROR, "BL1 fullload kl0HoWo invalid: %u (need 16-multiple)", howoWin);
        });
        // ★容量门（本路径不支持超容量 shape，无 fallback）：
        //   L0C: alignedCout·nFull·sizeof ≤ TOTAL_L0C_SIZE（256KB，全 dhw 驻留单视图）
        //   L0B: howoWin·nL0c·sizeof ≤ TOTAL_L0B_SIZE/2（32KB 半区）——单 (win,dk) tile
        //        = Ceil(howoWin,8)·8·nL0c 元素（k-row 数 × 8N）≤ 半区；kl0HoWo=16 时
        //        即旧式 16·nL0c（现值），dk≥2 时蕴含于旧总量门
        ASCENDC_ASSERT(static_cast<uint64_t>(alignedCout) * nFull * sizeof(SrcT) <= TOTAL_L0C_SIZE, {
            KERNEL_LOG(KERNEL_ERROR, "BL1 fullload L0C overflow: %u*%u*%uB", alignedCout, nFull,
                       static_cast<uint32_t>(sizeof(SrcT)));
        });
        ASCENDC_ASSERT(static_cast<uint64_t>(howoWin) * nL0c * sizeof(SrcT) <= TOTAL_L0B_SIZE / 2, {
            KERNEL_LOG(KERNEL_ERROR, "BL1 fullload L0B half overflow: %u*%u*%uB", howoWin, nL0c,
                       static_cast<uint32_t>(sizeof(SrcT)));
        });

        // ★第十二轮修改2/3 L1 布局：B1@0（fmap k 轴全载驻留，含 batch 维——只有 fmap
        // 承诺全载；dy 从不承诺 batch 轴全载）+ al1Ping + al1Pong 双半区（各 = 单 batch 的
        // [co1g][paddedDhowo][co0]，32B 对齐）——下一 batch 的 A1 MTE2 装载与当前 batch
        // 计算重叠
        const uint64_t dhwin = static_cast<uint64_t>(shape.din) * shape.hin * shape.win;
        const uint64_t bl1Elems = static_cast<uint64_t>(shape.batch) * dhwin * alignedCin;
        const uint32_t dhowoTotal = shape.dout * shape.hout * shape.wout;
        // A1 howo 行距 16 对齐 padding（第九轮）：单命令转置 srcStride = paddedDhowo/128 元素
        const uint32_t paddedDhowo = (dhowoTotal + 15) / 16 * 16;
        const uint32_t al1HalfElems = paddedDhowo * alignedCout;
        const uint32_t bl1Bytes = static_cast<uint32_t>((bl1Elems * sizeof(SrcT) + 31) / 32 * 32);
        const uint32_t al1HalfBytes = (al1HalfElems * sizeof(SrcT) + 31) / 32 * 32;
        LocalTensor<SrcT> bl1(TPosition::A1, 0, static_cast<uint32_t>(bl1Elems));
        // ★L1 容量门（第十二轮补 A1×2 项）：B1 + 2·A1 半区 ≤ TOTAL_L1_SIZE（512KB）
        ASCENDC_ASSERT(static_cast<uint64_t>(bl1Bytes) + 2 * al1HalfBytes <= TOTAL_L1_SIZE, {
            KERNEL_LOG(KERNEL_ERROR, "BL1 fullload L1 overflow: %u+2*%uB", bl1Bytes, al1HalfBytes);
        });

        constexpr uint32_t l0aHalfElems = TOTAL_L0A_SIZE / 2 / sizeof(SrcT);
        constexpr uint32_t l0bHalfElems = TOTAL_L0B_SIZE / 2 / sizeof(SrcT);
        LocalTensor<SrcT> l0c(TPosition::CO1, 0, TOTAL_L0C_SIZE / sizeof(SrcT));

        // ★跨块 MTE2←MTE1 背压（第十轮四件套，第十二轮起仅针对 B1——A1 半区化后自带
        // MTE1_MTE2 逐半区背压链）：闸住 B1 覆写（MTE2 队首）
        WaitFlag<HardEvent::MTE1_MTE2>(0);
        LoadL1Fmap(bl1, config, cRange);
        SetFlag<HardEvent::MTE2_MTE1>(0);
        // B1 就绪（MTE1 一次性等待，窗口循环外 = MTE1 队首，先于一切 load2d/load3d）
        WaitFlag<HardEvent::MTE2_MTE1>(0);
        // FIX→M 跨块背压：等上一块 fixpipe 读完 L0C（首次靠 Init 预置位；放装载后使
        // MTE2 装载与上一块 fixpipe 并行）
        WaitFlag<HardEvent::FIX_M>(0);

        // ★dk 内移循环 + A/B/A1 三级独立 ping-pong（第十一轮方案 + 第十二轮修改1-3）：
        // A 侧与 dk 无关 → load2d 每窗一次（k3³ 时 -67%）；B 侧每 (win,dk) 独占一个 L0B
        // 半区；A1 侧每 batch 独占一个 L1 半区（dy 单 batch 搬运，dhowo 轴全载）。
        // ★全局 flag id 分配表（通道互异，id 不重叠）：
        //   MTE2_MTE1: 0 = B1 就绪（每块，即时对）| 1/2 = A1 ping/pong 就绪（每 batch，即时对）
        //   MTE1_MTE2: 0 = 跨块 B1 覆写背压（End Set + 下块装载前 Wait + Drain，四件套）
        //             | 1/2 = A1 ping/pong 半区背压（每 batch 尾 Set + 装载前 Wait，Init
        //               预置/Drain 消费；第十二轮续改动1：由 M_MTE2 改 MTE1 方向——
        //               L1→L0 搬完即可覆写，不等 Mmad 完成）
        //   MTE1_M:    0/1 = A(dy) L0A 半区装载就绪 | 2/3 = B(fmap) L0B 半区装载就绪（即时对）
        //   M_MTE1:    0/1 = L0A 半区释放 | 2/3 = L0B 半区释放（★修改1：Wait 前置到装载前、
        //              Mmad 后只 Set 不 Wait；Init 预置 4 id / Drain 消费 4 id）
        //   M_FIX/FIX_M = 0（fixpipe 前向即时对 / 跨块 L0C 释放，Init 预置/Drain 消费）
        // ★Set/Wait 配平表（逐 id，跨块累计）：
        //   M_MTE1 id k: Set = Init(1)+使用 n_k，Wait = n_k+Drain(1) → 恒配平
        //   MTE1_MTE2 id 1/2: Set = Init(1)+batch 使用 p_k，Wait = p_k+Drain(1) → 恒配平
        //   （半区未用则 n_k/p_k=0：Init 预置由 Drain 消费，仍配平）
        // 同步时序（①②就绪向即时对保留；③④释放向改引擎模式）：
        //   ① W<M_MTE1>(aId) → load2d → S/W<MTE1_M>(aId)
        //   ② W<M_MTE1>(bId+2) → load3d → S/W<MTE1_M>(bId+2)
        //   ③ Mmad 后 S<M_MTE1>(bId+2)（只 Set：释放信号挂 M 队列）
        //   ④ dk 循环后 S<M_MTE1>(aId)（只 Set：本窗全部 dk Mmad 消费完 A 半区）
        // 释放向 Wait 前置后，MTE1 装载（load2d/load3d）可与 M 流水重叠（修改1 前 ③④
        // 为即时 S/W，标量被 Mmad 完成阻塞、全串行化）
        // cmatrixInitVal：每 dk 段首 Mmad 清累加——段首 = (batch=0, 该 dk 首个有效 dout,
        // howoIdx=0)，首个有效 dout = max(0, dPad-dk)（round-8 shape 门保证存在）
        for (uint32_t batchIdx = 0; batchIdx < shape.batch; batchIdx++) {
            // ★修改3：A1 半区选择（a1Id∈{0,1} → flag id = a1Id+1∈{1,2}，id 0 留给 B1）
            const uint32_t a1Id = a1Pong_;
            a1Pong_ ^= 1;
            LocalTensor<SrcT> al1(TPosition::A1, bl1Bytes + a1Id * al1HalfBytes,
                                  al1HalfElems);
            // A1 半区背压（MTE1_MTE2，第十二轮续改动1：MTE1→MTE2 方向）：等 2 个 batch
            // 前同半区的 load2d（MTE1 流）全部排空——Mmad 未完但 L1→L0 搬完即可覆写
            // （A1 单 batch 布局，半区隔 batch 复用；Init 预置首两 batch 免等）
            WaitFlag<HardEvent::MTE1_MTE2>(a1Id + 1);
            // ★修改2：dy 每 batch 单独搬运（dnNum=1，dhowo 轴全载）——batch 轴不保证全载
            LoadA1Dy(al1, config, cRange, batchIdx);
            // A1 半区就绪（即时对，S 挂 MTE2 队列装载后）
            SetFlag<HardEvent::MTE2_MTE1>(a1Id + 1);
            WaitFlag<HardEvent::MTE2_MTE1>(a1Id + 1);
            for (uint32_t dout = 0; dout < shape.dout; dout++) {
                const uint32_t howoTotal = shape.hout * shape.wout;
                for (uint32_t howoIdx = 0; howoIdx < howoTotal; howoIdx += howoWin) {
                    const uint32_t howoLen = Std::min(howoWin, howoTotal - howoIdx);
                    // A pong 翻转点：每次 LoadL0Dy 调用翻一次（aId 即半区选择与 flag id）
                    const uint32_t aId = aPong_;
                    aPong_ ^= 1;
                    LocalTensor<SrcT> l0a(TPosition::A2, aId * (TOTAL_L0A_SIZE / 2),
                                         l0aHalfElems);
                    // ①修改1：释放向 Wait 前置（等 2 窗前同半区 Mmad 读完再装载）
                    WaitFlag<HardEvent::M_MTE1>(aId);
                    LoadL0Dy(al1, config, cRange, dout, howoIdx, howoLen, l0a);
                    // A 侧装载就绪（S 挂 MTE1 队列 load2d 后，即时配对）
                    SetFlag<HardEvent::MTE1_M>(aId);
                    WaitFlag<HardEvent::MTE1_M>(aId);
                    for (uint32_t dk = 0; dk < shape.dk; dk++) {
                        const int32_t dIn = static_cast<int32_t>(dout + dk) -
                                            static_cast<int32_t>(shape.dPad);
                        if (dIn < 0 || dIn >= static_cast<int32_t>(shape.din)) {
                            continue; // 该 dk 平面全 pad（段内部分跳过，round-8 门保证段非全空）
                        }
                        // B pong 翻转点：每次 LoadL0Fmap 调用翻一次（半区基址 = bId×半区，
                        // flag id = bId+2）
                        const uint32_t bId = bPong_;
                        bPong_ ^= 1;
                        LocalTensor<SrcT> l0b(TPosition::B2, bId * (TOTAL_L0B_SIZE / 2),
                                             l0bHalfElems);
                        // ②修改1：释放向 Wait 前置（等 2 次调用前同半区 Mmad 读完再装载）
                        WaitFlag<HardEvent::M_MTE1>(bId + 2);
                        LoadL0Fmap(bl1, config, cRange, batchIdx, static_cast<uint32_t>(dIn),
                                   howoIdx, howoLen, nL0c, l0b);
                        // B 侧装载就绪（S 挂 MTE1 队列 load3d 后，即时配对）
                        SetFlag<HardEvent::MTE1_M>(bId + 2);
                        WaitFlag<HardEvent::MTE1_M>(bId + 2);
                        MmadParams mmad;
                        mmad.m = alignedCout;
                        mmad.n = nL0c;
                        mmad.k = howoLen;
                        const uint32_t firstDout = shape.dPad > dk ? shape.dPad - dk : 0;
                        mmad.cmatrixInitVal =
                            (batchIdx == 0) && (howoIdx == 0) && (dout == firstDout);
                        // Mmad 目的带 dk 段偏移：L0C 段基址 = 16M·cin16G·dk·hkwk 元素（n 序不变）；
                        // B 指针 = 本 (win,dk) 半区基址（k-row0 即半区起点，无段内偏移）
                        Mmad(l0c[16 * alignedCout * cin16G * dk * hkwk], l0a[0], l0b[0], mmad);
                        // ③修改1：B 半区释放（只 Set 不 Wait——释放信号挂 M 队列 Mmad 后，
                        // 2 次调用后的装载前 Wait 消费；MTE1 与 M 由此流水重叠）
                        SetFlag<HardEvent::M_MTE1>(bId + 2);
                    }
                    // ④修改1：A 半区释放（只 Set 不 Wait——本窗全部 dk Mmad 消费完 A 半区，
                    // 2 窗后的装载前 Wait 消费）
                    SetFlag<HardEvent::M_MTE1>(aId);
                }
            }
            // ★第十二轮续改动1：A1 半区释放（MTE1_MTE2 信号挂 MTE1 队列——本 batch 全部
            // load2d/load3d 排空后 fire，此时 Mmad 可仍在执行（只读 L0A/L0B，不再读 L1）；
            // 2 个 batch 后的同半区装载前 Wait 消费）
            SetFlag<HardEvent::MTE1_MTE2>(a1Id + 1);
        }
        // M→FIX 前向（整块一次 fixpipe，即时配对）
        SetFlag<HardEvent::M_FIX>(0);
        WaitFlag<HardEvent::M_FIX>(0);
        DirectOutL0C(l0c, y, yBase, alignedCout, hkwk, dhwk, cRange, shape.cin);
        // FIX→M 释放（跨块：下一块 IterateK 开头的 Wait 消费，末块由 Drain 消费）
        SetFlag<HardEvent::FIX_M>(0);
        // MTE1 管排空信号（跨块 B1/A1 装载背压，第十轮）
        SetFlag<HardEvent::MTE1_MTE2>(0);
    }

private:
    // ---------- A 侧：dy GM → A1(单 batch 驻留 [co1][paddedDhowo][co0]) → L0A(load2d 转置读) ----------
    // S3（第十二轮修改2）：dy 每 batch 单独装载——只有 fmap(B1) 承诺 k 轴全载（含 batch），
    // dy 只承诺 dhowo 轴全载、batch 轴不保证（每 batch 一条 Dn2Nz{dnNum=1}，半区承载 batch 维）。
    // d 轴=co 跨 co1g 组落 C0/C1 槽（与 B1 的 dValue=cinLength 同机制，B1 实证无 camodel 分歧）。
    // A1 半区目的布局（单 batch）：
    //   addr(co, pt) = (co/C0)*(paddedDhowo*C0) + pt*C0 + (co%C0)
    // LoadL0Dy 源偏移删 batch 项（A1 内 batch 维消失），paddedDhowo 行距 16 对齐逻辑照旧
    __aicore__ inline void LoadA1Dy(const LocalTensor<SrcT>& al1, const BL1FullLoadConfig& config,
                                    const CoutCinRange& cRange, uint32_t batchIdx)
    {
        const ShapeAttribute& shape = config.shape;
        const uint32_t alignedCout = config.tiling.singleShapeAlignedCout;
        const uint32_t dhowo = shape.dout * shape.hout * shape.wout;
        const uint32_t paddedDhowo = (dhowo + 15) / 16 * 16; // A1 行距 16 对齐（见 IterateK 注释）
        Dn2NzParams dn2nz;
        dn2nz.dnNum = 1;                                           // ★单 batch 搬运（修改2）
        dn2nz.dValue = cRange.coutLength;                          // 真实 co 段长（B1 同构取真实段长）
        dn2nz.nValue = static_cast<uint16_t>(dhowo);               // dhowo 合轴全量（真实行数，pad 行不写）
        dn2nz.srcDnMatrixStride = dhowo;                           // dnNum=1 不生效，语义完整保留
        dn2nz.srcDValue = dhowo;                                   // co 行距
        dn2nz.dstNzC0Stride = static_cast<uint16_t>(paddedDhowo);  // A1 行距 = padded（srcStride 整除性）
        dn2nz.dstNzNStride = 1;
        dn2nz.dstNzMatrixStride = static_cast<uint32_t>(paddedDhowo) * alignedCout; // dnNum=1 不生效
        // 源基址：batch 段 + 块内 co 偏移（dy GM NCDHW 每 co 行 dhowo）
        const uint64_t offset = static_cast<uint64_t>(batchIdx) * shape.cout * dhowo +
                                static_cast<uint64_t>(cRange.coutIdx) * dhowo;
        DataCopy(al1, dy_[offset], dn2nz);
    }

    // S5'（第九轮专家裁决）：load2d 单命令大范围转置——一条命令搬整窗 [m=co 段][k=howo 窗]，
    // 替代第八轮前的逐 (m1,k1) 块循环（每窗 (M/8)·Ceil(K/8) 条 → 1 条）。
    // 参数式 = winograd ComputePoints fp32 转置先例（真机验证代码）：
    //   mStep = Ceil(窗宽,16)（k 方向 16 组数；kl0HoWo=16 现值 → 恒 1）
    //   kStep = M/8（m 方向 C0 组数；M=alignedCout 恒 16 对齐 → kStep 偶数，专家约束 ✓）
    //   srcStride = m1 组距/128 元素 = paddedDhowo/16（k1 组内 16 howo 连续、硬件按 1 单位推进）
    //   dstStride = kStep/2（winograd 原样）
    // 命令覆盖 [16·mStep k][M m]，按 S1 布局连续落 l0a[0]；尾窗 howoLen<16 时 k∈[howoLen,16)
    // 读 pad/garbage，Mmad k=howoLen 不消费 ✓；A1 行距 16 对齐 padding 保证 srcStride 整数
    __aicore__ inline void LoadL0Dy(const LocalTensor<SrcT>& al1, const BL1FullLoadConfig& config,
                                    const CoutCinRange& cRange, uint32_t dout,
                                    uint32_t howoIdx, uint32_t howoLen, const LocalTensor<SrcT>& l0a)
    {
        const ShapeAttribute& shape = config.shape;
        const uint32_t alignedCout = config.tiling.singleShapeAlignedCout;
        const uint32_t dhowo = shape.dout * shape.hout * shape.wout;
        const uint32_t paddedDhowo = (dhowo + 15) / 16 * 16;
        LoadData2DParamsV2 load2d;
        load2d.ifTranspose = 1;
        load2d.mStartPosition = 0;
        load2d.kStartPosition = 0;
        load2d.mStep = static_cast<uint16_t>((howoLen + 15) / 16);
        load2d.kStep = static_cast<uint16_t>(alignedCout / BpUtils::C0<SrcT>());
        load2d.srcStride = static_cast<int32_t>(paddedDhowo / 16);
        load2d.dstStride = static_cast<uint16_t>(load2d.kStep / 2);
        // 源偏移：(dout*hwout+howoIdx)*C0（co1g=0 段内 point 起点；A1 半区为单 batch
        // 布局，batch 项由半区基址承载，第十二轮修改2）
        const uint64_t srcOff =
            (static_cast<uint64_t>(dout) * shape.hout * shape.wout + howoIdx) * BpUtils::C0<SrcT>();
        LoadData(l0a[0], al1[srcOff], load2d);
    }

    // ---------- B 侧：B1(fmap 合轴驻留) → L0B（load3d 合轴 tile，第十轮层1'） ----------
    // S4'：每 tap 一条命令 + rt=2*cin16G（repeat 配对按 +16 槽连续铺块——probe_r10 块级
    // 落位验证），替代 g16×tap 双循环（命令数/窗 cin16G·hkwk → hkwk）。
    // 通道视图 g' = c1g*din + d（平面基址 g'*hwIn*C0 均匀），d 平面选择走 L1 源偏移
    // srcOff = (batch 段 + dIn)*hwIn*C0（c1g=0 基，rt 的 rs=din*hkwk 自动步进 c1g）。
    // ★块序变化（消费端适配）：块 B = cin16G*tap + g（tap 外层，配对连续落位所致）——
    // fixpipe 的 CS/SS 相应重映射（见 DirectOutL0C），y 生产布局不变。
    // 尾组安全：rt 固定 2*cin16G，超出真实 c1g 数的 repeat 读 B1 尾部/邻区 garbage
    // 落块上半，fixpipe nSize=真实 cinLength 不读（cinLength≤16*g+8 的上界保证）。
    // ★层2（单命令/窗）不可行证明：fixpipe 的 d%16+16·SS·(d/16) 结构强制 L0C n 轴按
    // ci 16 槽块组织且 d=ci/n=tap 角色锁定（dst=DM·dn+n+DS·d 与 y[co][ci][tap] 唯一
    // 匹配），而 load3d k 轴内禀 [c1][tap][c0] 连续落位 + repeat 单 rs 等差游走无法
    // 产出该块序（跨 tap 需非等差跳步）——三重约束下 hkwk 条/窗为下界。
    __aicore__ inline void LoadL0Fmap(const LocalTensor<SrcT>& bl1, const BL1FullLoadConfig& config,
                                      const CoutCinRange& cRange, uint32_t batchIdx, uint32_t dIn,
                                      uint32_t howoIdx, uint32_t howoLen, uint32_t nExt,
                                      const LocalTensor<SrcT>& l0b)
    {
        const ShapeAttribute& shape = config.shape;
        const uint32_t alignedCin = config.tiling.singleShapeFullLoadAlignedCin;
        const uint32_t hwIn = shape.hin * shape.win;
        const uint32_t hkwk = shape.hk * shape.wk;
        const uint32_t cin16G = (cRange.cinLength + 15) / 16;
        const uint64_t dhwin = static_cast<uint64_t>(shape.din) * hwIn;

        LoadData3DParamsV2<SrcT> load3d;
        load3d.l1H = shape.hin;
        load3d.l1W = shape.win;
        load3d.padList[0] = static_cast<uint8_t>(shape.wPad); // left
        load3d.padList[1] = static_cast<uint8_t>(shape.wPad); // right
        load3d.padList[2] = static_cast<uint8_t>(shape.hPad); // top（引擎 bL1PadUp）
        load3d.padList[3] = 255;                               // bottom：引擎 DEFAULT_PAD_DOWN 恒 255
        load3d.channelSize = static_cast<uint16_t>(shape.din * alignedCin); // 合轴通道视图 [c1g][d]
        load3d.kExtension = BpUtils::C0<SrcT>();                        // 单 tap 窗（kExt 多 tap 的 k 轴序
                                                              // [c1][tap][c0] 与 FZ 16 槽不兼容）
        load3d.kStartPt = 0;
        load3d.mStartPt = static_cast<uint16_t>(howoIdx);     // howo 窗起点
        load3d.mExtension = static_cast<uint16_t>(howoLen);   // 真实窗宽（引擎 baseUseK 先例）
        load3d.strideW = 1;
        load3d.strideH = 1;
        load3d.filterW = shape.wk;
        load3d.filterH = shape.hk;
        load3d.dilationFilterW = 1;
        load3d.dilationFilterH = 1;
        load3d.enTranspose = false;

        // rs = din*hkwk：repeat r 源窗口 +r*rs*C0 = (c1g r, 同 d, 同 tap)（跨通道自动进位）；
        // rt = 2*cin16G：配对 (2g, 2g+1) 落块 g（下 8 槽 c1g 偶组 + 上 8 槽 c1g 奇组）
        LoadDataRepeatParamWithStride rep;
        rep.repeatStride = static_cast<uint16_t>(shape.din * hkwk);
        rep.repeatTime = static_cast<uint8_t>(2 * cin16G);
        rep.repeatMode = 1;
        rep.dstStride = static_cast<uint16_t>(nExt / 16);           // ds=引擎 ShiftCeilM0(baseUseN,n0)
        SetLoadDataRepeatWithStride(rep);

        // 源偏移：batch 段 + dIn 平面基址（c1g=0 起，rs 自动步进全 c1g）
        const uint64_t srcOff = static_cast<uint64_t>(batchIdx) * dhwin * alignedCin +
                                static_cast<uint64_t>(dIn) * hwIn * BpUtils::C0<SrcT>();
        for (uint32_t tap = 0; tap < hkwk; tap++) {
            load3d.kStartPt = static_cast<uint16_t>(tap * BpUtils::C0<SrcT>());
            // ★第十一轮 B 侧 ping-pong：每个 (win,dk) 的 tile 独占一个 L0B 半区（基址由
            // 调用方 pong 位选择），段内 tap 块紧堆于半区起点（dk 段偏移消失）。
            // k-row1 硬件语义（probe_r10/r11 钉死）：mExt 跨 8 行时 k-row1 自动落
            // dst 基址+8N（N = 段宽 16*cin16G*hkwk）处（偏移量由 N 决定，与 tap/dstStride
            // 无关）——tile 实占 [半区基址, +16N) = k-row0+k-row1，故容量门 16·nL0c ≤ 半区。
            // （dk 段曾按整 tile 步长 2*cin16G*hkwk 块全视图堆叠——k-row1 与 N 向紧堆的
            // 下段 k-row0 恒重叠（8N==128*cin16G*hkwk），本轮半区化后自然消除。
            // L0C 侧 n 序不变：cin16G*(dk*hkwk+tap)+g，B/C 视图解耦合法）
            LoadDataWithStride(l0b[128 * cin16G * tap], bl1[srcOff], load3d);
        }
    }

    // ---------- fmap GM → B1 合轴驻留（S2：dnNum=batch 一把搬全载） ----------
    __aicore__ inline void LoadL1Fmap(const LocalTensor<SrcT>& bl1, const BL1FullLoadConfig& config,
                                      const CoutCinRange& cRange)
    {
        if (loadedCinIdx_ == cRange.cinIdx && loadedCinLength_ == cRange.cinLength) {
            return;
        }
        const ShapeAttribute& shape = config.shape;
        const uint32_t alignedCin = config.tiling.singleShapeFullLoadAlignedCin;
        const uint64_t dhwin = static_cast<uint64_t>(shape.din) * shape.hin * shape.win;

        Dn2NzParams dn2nz;
        dn2nz.dnNum = static_cast<uint16_t>(shape.batch);        // 一把搬全部 batch
        dn2nz.dValue = cRange.cinLength;                         // 真实段长（引擎 bL1cin1CopyLen）
        dn2nz.nValue = static_cast<uint16_t>(dhwin);             // din×hin×win 合轴
        dn2nz.srcDnMatrixStride = static_cast<uint64_t>(shape.cin) * dhwin; // batch 步距（NCDHW）
        dn2nz.srcDValue = dhwin;                                 // cin 行距
        dn2nz.dstNzC0Stride = static_cast<uint16_t>(dhwin);
        dn2nz.dstNzNStride = 1;
        dn2nz.dstNzMatrixStride = static_cast<uint32_t>(dhwin * alignedCin); // batch 段距
        // 源基址：块内 cin 偏移（fmap GM NCDHW 每 cin 行 dhwin）
        const uint64_t offset = static_cast<uint64_t>(cRange.cinIdx) * dhwin;
        DataCopy(bl1, fmap_[offset], dn2nz);

        loadedCinIdx_ = cRange.cinIdx;
        loadedCinLength_ = cRange.cinLength;
    }

    // ---------- L0C 直出（每 dk 段一次 NZ2DN，直写生产布局 y[co][ci][dk*hwK+tap]） ----------
    // S8：fixpipe L0C→GM 直达，COLUMN_MAJOR(NZ2DN)——引擎 LoadL0c2GmNormal 先例
    //（退化 curSingleCoreDk=1：每 dk 段独占 L0C，源固定段基址，无 srcDkStride）
    __aicore__ inline void DirectOutL0C(const LocalTensor<SrcT>& l0c, GlobalTensor<SrcT>& y, uint64_t yBase,
                                        uint32_t alignedCout, uint32_t hkwk, uint32_t dhwk,
                                        const CoutCinRange& cRange, uint32_t cinTotal)
    {
        FixpipeParamsArch3510<CO2Layout::COLUMN_MAJOR> fp;
        fp.params.dnNum = static_cast<uint16_t>(cRange.coutLength);   // DN 矩阵数 = co（真实值）
        fp.mSize = static_cast<uint16_t>(dhwk);                     // DN 列 = dhwk（整块一次，第十一轮）
        fp.nSize = static_cast<uint16_t>(cRange.cinLength);          // DN 行 = ci（真实值）
        fp.params.srcNzMatrixStride = 1;                             // C0 单位：co 步进 16 元素
        // ★第十轮块序适配：L0C 块 B = cin16G*tap + g（tap 外层，LoadL0Fmap 配对连续落位）。
        // 块序公式 B = (CS*tap + SS*g)/M → CS = M*cin16G（tap 步进 cin16G 块）、SS = M（g 步进 1 块）。
        // 旧序 B = g16*hkwk + tap 时为 CS=M、SS=M*hkwk。y 生产布局不变（DM/DS/nSize/mSize 不动）
        fp.params.srcNzC0Stride = static_cast<uint16_t>(alignedCout * ((cRange.cinLength + 15) / 16)); // tap 步进
        fp.srcStride = static_cast<uint16_t>(alignedCout);           // cin16 组步进（C0 单位）
        fp.dstStride = dhwk;                                         // 元素：DN 行长
        fp.params.dstDnMatrixStride = cinTotal * dhwk;               // 元素：相邻 co 的 DN 步进
        fp.quantPre = QuantMode_t::NoQuant;
        fp.unitFlag = 0;
        Fixpipe<SrcT, float, CFG_COLUMN_MAJOR>(y[yBase], l0c, fp);
    }

    // A/B/A1 独立 pong 位（翻转频率各不相同，必须三个独立位）：
    //   aPong_  每次 LoadL0Dy 翻转（(dout,win) 级，L0A 半区）
    //   bPong_  每次 LoadL0Fmap 翻转（(win,dk) 级，L0B 半区）
    //   a1Pong_ 每次 LoadA1Dy 翻转（batch 级，A1 L1 半区，第十二轮）
    uint32_t aPong_ = 0;
    uint32_t bPong_ = 0;
    uint32_t a1Pong_ = 0;
    GlobalTensor<SrcT> fmap_;
    GlobalTensor<SrcT> dy_;
    uint32_t loadedCinIdx_ = 0;
    uint32_t loadedCinLength_ = 0;
};

} // namespace BpFullLoad

#endif // CONV_BP_BL1_FULLLOAD_COMPUTE_H
