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
 *   S2. B1(fmap) 合轴驻留 [batch][c1g][d][h][w][c0]（C1 外 D 内，dhw 合轴连续；第十六轮
 *       起段距按本块 cinAlign16 = CeilAlign(cinLength,16)——尾块/主块一套逻辑，B1 分配
 *       上界另按 max(tiling alignedCin, cinAlign16) 口径给出）：
 *       addr(b,ci,d,hw) = b*(dhwin*cinAlign16) + (ci/C0)*(dhwin*C0) + d*(hwIn*C0) + hw*C0 + (ci%C0)
 *       LoadL1Fmap = 单条 Dn2Nz{dnNum=batch, dValue=cinLength, nValue=dhwin, …} 全载驻留，
 *       loadedCinIdx_ 缓存使 B1 仅在 cin 块切换时重装（命中含 cinLength → 布局逐块自洽）
 *   S3. A1(dy) 合轴驻留 [batch][co1g][dhowo][co0]（D 在 C1 内，dhowo 合轴连续）：
 *       addr(b,co,pt) = b*(dhowo*alignedCout) + (co/C0)*(dhowo*C0) + pt*C0 + (co%C0)
 *       LoadA1Dy = 每 batch 一条 Dn2Nz{dnNum=co1g 段数, dValue=C0}（dn 维堆叠承载 co1g 段距）
 *   S4. load3d(B1→L0B) 合轴 tile（第十一轮）：每 (窗,dk,tap) 一条命令（kExt=C0、
 *       kStartPt=tap*C0）+ rt=cinAlign16/C0（=2*cin16G，配对 (2g,2g+1) 落块 g、+16 槽
 *       连续铺块）、rs=din*hkwk、ds=nExt/16、dst=l0b[128*cin16G*tap]（半区基址 + tap
 *       段内偏移，第十二轮 B 侧 ping-pong）；块序 B = cin16G*tap + g（L0C 侧 n 序另带
 *       dk 段偏移）；★Fmatrix/padding 块首外提一次（第十五轮），命令内不重设
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
 *       TQue Alloc/Free 手动等价：Init 预置位/装载前 Wait/End Set/入口 Drain
 * 消费）。★事件链每段恒定（含空段），守卫只落 在数据操作上；★dk 全 pad 空段跳过 fixpipe 本体（y 保持 host
 * 清零值，golden=0）★A1 预载链（第十五轮）：预载 batch0 → 循环{装载本批 → 计算上一批}
 * → 循环外计算末批；装载前 Wait<MTE1_MTE2>(半区id) / 计算后 Set<MTE1_MTE2>(半区id)、
 *       首两轮 Init 预置；装载次数 = 计算次数 = batch，全链配平（详见 IterateK 注释）
 *   S10. ★LocalTensor 索引/偏移入参必须 u32（第十四轮板测实锤）：L1/L0 偏移恒 < 2^32
 *       （片上容量上界：L1 512KB/4B = 131072 元素），u64 传入 operator[] 会在内联优化下
 *       触发 S64 标量溢出（check_status overflow）；GM/GlobalTensor 侧偏移才用 u64
 */

#ifndef CONV_BP_BL1_FULLLOAD_COMPUTE_H
#define CONV_BP_BL1_FULLLOAD_COMPUTE_H

#include "basic_api/kernel_basic_intf.h"
#include "op_kernel/math_util.h"
#include "utils/std/algorithm.h"

#include "../util/conv_bp_common_util.h"
#include "conv_bp_bl1_fullload_config.h"

namespace BpFullLoad {

using namespace AscendC;
using BpUtils::C0;
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
        SetFlag<HardEvent::MTE1_MTE2>(MTE2FlagFmap());
        SetFlag<HardEvent::FIX_M>(0);
        SetFlag<HardEvent::M_MTE1>(MTE1Flag<BpUtils::InputTensor::DY>(false));
        SetFlag<HardEvent::M_MTE1>(MTE1Flag<BpUtils::InputTensor::DY>(true));
        SetFlag<HardEvent::M_MTE1>(MTE1Flag<BpUtils::InputTensor::FMAP>(false));
        SetFlag<HardEvent::M_MTE1>(MTE1Flag<BpUtils::InputTensor::FMAP>(true));
        SetFlag<HardEvent::MTE1_MTE2>(MTE2FlagDy(false));
        SetFlag<HardEvent::MTE1_MTE2>(MTE2FlagDy(true));
    }

    // kernel 入口在全部块迭代结束后调用：消费跨块背压的残留 Set（TQue Reset 残留
    // freeBufEvt 消费先例）——漏掉则残留 flag 会污染同核后续 kernel 的首块装载
    __aicore__ inline void Drain()
    {
        // 消费残留 Set（Set/Wait 全程配平铁律，防跨 kernel 污染同核后续算子）：
        //   MTE1_MTE2(0)：末块 MTE1 排空信号 | FIX_M(0)：末块 fixpipe 释放
        //   M_MTE1(0-3)：各 L0A/L0B 半区末次释放（kernel 内无后续装载消费）
        //   MTE1_MTE2(1/2)：A1 各半区末次释放（同上；从未使用的半区消费 Init 预置位）
        WaitFlag<HardEvent::MTE1_MTE2>(MTE2FlagFmap());
        WaitFlag<HardEvent::FIX_M>(0);
        WaitFlag<HardEvent::M_MTE1>(MTE1Flag<BpUtils::InputTensor::DY>(false));
        WaitFlag<HardEvent::M_MTE1>(MTE1Flag<BpUtils::InputTensor::DY>(true));
        WaitFlag<HardEvent::M_MTE1>(MTE1Flag<BpUtils::InputTensor::FMAP>(false));
        WaitFlag<HardEvent::M_MTE1>(MTE1Flag<BpUtils::InputTensor::FMAP>(true));
        WaitFlag<HardEvent::MTE1_MTE2>(MTE2FlagDy(false));
        WaitFlag<HardEvent::MTE1_MTE2>(MTE2FlagDy(true));
    }

    // 单个基本块 [coutRange × cinRange] 的 K 全载计算（第十五轮 A1 预载结构）：
    // L0C 全 dhw 驻留单视图（n 轴 [dk][hkwk][cin16] 块序），dk 内移循环 batch×dout×howo16 窗
    // → dk 段内累加（cmatrixInitVal 段首清零），块尾 fixpipe NZ2DN 整块一次直出。
    //
    // y 布局契约（生产布局）：
    //   y = [cout][cin][dhwK] ND 行主序（全局），dhwK = dk*hwK
    //   本块基址 yBase = coutIdx*cinTotal*dhwK + cinIdx*dhwK（由入口计算）
    //   元素 y[yBase + co*cinTotal*dhwK + ci*dhwK + dk*hwK + tap] = C(co, ci, tap) of 本 dk 段
    //
    // ★A1 预载结构（第十五轮，解决 fp32 load3d 指令数高堵 issueque）：
    //   预载 batch0 的 dy→A1 → for batchIdx=1..batch-1 { 先发 batchIdx 的 A1 DataCopy
    //   （MTE2 抢在上一批 load3d 洪流前入队）→ 计算上一 batch（IterateKL0，batchIdx-1）}
    //   → 循环外计算最后 batch（batch-1）。装载次数 = 计算次数 = batch。
    //
    // ★事件链时序（A1 半区 ping-pong，flag id 见下表；半区物理布局 B1@0 + 2×A1 半区）：
    //   装载前 Wait<MTE1_MTE2>(半区id)（等 2 个 batch 前同半区 load2d 排空，首两轮靠
    //   Init 预置）→ LoadA1Dy → Set<MTE2_MTE1>(半区id)（MTE2 就绪即时对）
    //   → 计算前 Wait<MTE2_MTE1>(半区id) → IterateKL0（本批全部 load2d/load3d/Mmad）
    //   → 计算后 Set<MTE1_MTE2>(半区id)（本批 MTE1 流排空即 fire，Mmad 可仍在执行）
    //   Set/Wait 配平（MTE1_MTE2 id1/2 逐 id，跨 kernel 累计）：
    //   Set = Init(1) + batch 计算次数 p_k，Wait = batch 装载次数 p_k + Drain(1) → 恒配平
    //   （装载与计算同 batch 数且半区序一致——装载半区 2 个 batch 后即计算同半区）
    //   其余链同旧版：L0A/L0B 半区 M_MTE1 背压（Wait 前置装载前 + Mmad 后只 Set）、
    //   FIX/M_FIX 即时对 + FIX_M 跨块、B1 跨块四件套（入口 Wait/End Set/Drain 消费）
    __aicore__ inline void IterateK(const BL1FullLoadConfig& config, const CoutCinRange& cRange, GlobalTensor<SrcT>& y,
                                    uint64_t yBase)
    {
        const ShapeAttribute& shape = config.shape;
        const FullLoadTiling& tiling = config.tiling;
        const uint32_t alignedCout = tiling.singleShapeAligned16Cout;
        const uint32_t alignedCin = tiling.singleShapeFullLoadAligned16Cin;
        const uint32_t hkwk = shape.hk * shape.wk;
        const uint32_t dhwk = shape.dk * hkwk;

        // ★通道量统一（第十六轮专家裁决）：B1 布局与 L0/L0C 视图统一按本块
        // cinAlign16 = CeilAlign(cinLength,16)（尾块/主块一套逻辑），消除第十五轮的
        // alignedCin/cinAlign16 双轨。tiling 的 alignedCin 仅用于 B1 分配上界（见下）。
        //   cin16G = Ceil(cinLength/16)（L0B/L0C 的 16 槽块数）；cinAlign16 = 16·cin16G（= nL0c/hkwk）
        //   cout16Align = Ceil(coutLength/16)（块按 alignedCout 切、尾块长 ∈ (16k-16,16k]，恒 = alignedCout）
        const uint32_t cin16G = (cRange.cinLength + 15) / 16;
        const uint32_t cinAlign16 = 16 * cin16G;
        const uint32_t cout16Align = (cRange.coutLength + 15) / 16 * 16;

        // ★howo 窗宽（k 轴，第十二轮续改动4）：从 tiling.kl0HoWo 传入，不再写死 16。
        // 约束：kl0HoWo 须为 16 的倍数（load3d mStartPt = 窗起点须 16 对齐）
        const uint32_t howoWin = tiling.kl0HoWo;
        ASCENDC_ASSERT(howoWin != 0 && howoWin % 16 == 0,
                       { KERNEL_LOG(KERNEL_ERROR, "BL1 fullload kl0HoWo invalid: %u (need 16-multiple)", howoWin); });
        // ★容量门（本路径不支持超容量 shape，无 fallback）：
        //   L0C: alignedCout·(16·cin16G·dhwk)·sizeof ≤ TOTAL_L0C_SIZE（256KB，全 dhw 驻留单视图）
        //   L0B: howoWin·(16·cin16G·hkwk)·sizeof ≤ TOTAL_L0B_SIZE/2（32KB 半区）——单 (win,dk)
        //        tile = Ceil(howoWin,8)·8·16·cin16G·hkwk 元素（k-row 数 × 8N）≤ 半区；
        //        kl0HoWo=16 时即旧式 16·nL0c，dk≥2 时蕴含于旧总量门
        ASCENDC_ASSERT(static_cast<uint64_t>(alignedCout) * (cinAlign16 * dhwk) * sizeof(SrcT) <= TOTAL_L0C_SIZE, {
            KERNEL_LOG(KERNEL_ERROR, "BL1 fullload L0C overflow: %u*%u*%uB", alignedCout, cinAlign16 * dhwk,
                       static_cast<uint32_t>(sizeof(SrcT)));
        });
        ASCENDC_ASSERT(static_cast<uint64_t>(howoWin) * (cinAlign16 * hkwk) * sizeof(SrcT) <= TOTAL_L0B_SIZE / 2, {
            KERNEL_LOG(KERNEL_ERROR, "BL1 fullload L0B half overflow: %u*%u*%uB", howoWin, cinAlign16 * hkwk,
                       static_cast<uint32_t>(sizeof(SrcT)));
        });

        // ★L1 布局：B1@0（fmap 合轴全载驻留，含 batch 维——只有 fmap 承诺全载；dy 从不
        // 承诺 batch 轴全载）+ A1 ping/pong 双半区（各 = 单 batch 的
        // [co1g][paddedDhowo][co0]，32B 对齐）——下一 batch 的 A1 MTE2 装载与当前 batch
        // 计算重叠。A1 半区元素上界 = alignedCout·dout·CeilAlign(howoTotal,16)
        // ≥ 实际占用 Ceil(coutLength/C0)·paddedDhowo·C0（paddedDhowo = Ceil(dhowo,16)
        // 为 A1 实际行距，见 LoadA1Dy dstNzC0Stride）
        const uint32_t dhwin = shape.din * shape.hin * shape.win;
        // ★B1 分配=上界 / 布局=本块实际（第十六轮裁决二分，与 A1"分配大、布局紧凑"同构）：
        //   布局：B1 batch 段距 = dhwin·cinAlign16（本块 16 对齐，LoadL1Fmap/LoadL0Fmap 同源），
        //         实际写入 = 各 batch 段内 Ceil(cinLength/C0) 个 c1g 组；
        //   分配：bl1Elems 按 max(tiling alignedCin, 本块 cinAlign16) 口径——tiling 口径
        //         为块宽上界（专家域 aCin 恒 16 对齐 ≥ cinAlign16 时即 tiling 口径原样）；
        //         历史 8 对齐档（alignedCin < cinAlign16，如 sim case1 族）若仍按 tiling 口径，
        //         batch≥2 时 B1 batch1 段会侵入 A1 半区且被后续 LoadA1Dy 覆写 → 数值污染，
        //         故取 max 保证分配 ≥ 布局（A1 基址 = bl1Bytes 安全后移，且同 tiling 下
        //         跨块恒定——兄弟 cout 块 cinAlign16 相同，cin 块切换时重装同口径）
        const uint32_t bl1CinSpan = alignedCin > cinAlign16 ? alignedCin : cinAlign16;
        const uint32_t bl1Elems = shape.batch * dhwin * bl1CinSpan;
        const uint32_t dhowoTotal = shape.dout * shape.hout * shape.wout;
        const uint32_t al1Elems = alignedCout * shape.dout *
                                  Ops::Base::CeilAlign<uint32_t>(shape.hout * shape.wout, BLOCK_CUBE);
        const uint32_t bl1Bytes = static_cast<uint32_t>((bl1Elems * sizeof(SrcT) + 31) / 32 * 32);
        const uint32_t al1HalfBytes = (al1Elems * sizeof(SrcT) + 31) / 32 * 32;
        LocalTensor<SrcT> bl1(TPosition::A1, 0, static_cast<uint32_t>(bl1Elems));
        // ★L1 容量门：B1 + 2·A1 半区 ≤ TOTAL_L1_SIZE（512KB）
        ASCENDC_ASSERT(static_cast<uint64_t>(bl1Bytes) + 2 * al1HalfBytes <= TOTAL_L1_SIZE,
                       { KERNEL_LOG(KERNEL_ERROR, "BL1 fullload L1 overflow: %u+2*%uB", bl1Bytes, al1HalfBytes); });

        LocalTensor<SrcT> l0c(TPosition::CO1, 0, TOTAL_L0C_SIZE / sizeof(SrcT));

        // ★跨块 MTE2←MTE1 背压四件套（入口 Wait）：闸住 B1 覆写（MTE2 队首）
        WaitFlag<HardEvent::MTE1_MTE2>(MTE2FlagFmap());
        LoadL1Fmap(bl1, config, cRange);
        SetFlag<HardEvent::MTE2_MTE1>(MTE2FlagFmap());
        // B1 就绪（MTE1 一次性等待，窗口循环外 = MTE1 队首，先于一切 load2d/load3d）
        WaitFlag<HardEvent::MTE2_MTE1>(MTE2FlagFmap());
        // FIX→M 跨块背压：等上一块 fixpipe 读完 L0C（首块靠 Init 预置位；放 B1 装载后使
        // MTE2 装载与上一块 fixpipe 并行）
        WaitFlag<HardEvent::FIX_M>(0);

        // ★load3d 状态外提（第十五轮）：Fmatrix/padding 参数仅依赖 shape（块内恒定），
        // 提到 B1 就绪后设置一次；LoadL0Fmap 的 LoadDataWithStride 以 L3D_NO_RESET
        // （{false,false}）免每命令重设（旧形态随 IsResetLoad3dConfig 默认 {true,true}
        // 每条 load3d 命令各设一次 → hkwk 条/(win,dk) × 全部 (win,dk,batch) → 每块 1 次）
        const uint8_t fmatrixPadList[4] = {static_cast<uint8_t>(shape.wPad), static_cast<uint8_t>(shape.wPad),
                                           static_cast<uint8_t>(shape.hPad), static_cast<uint8_t>(shape.hPad)};
        SetFmatrix(static_cast<uint16_t>(shape.hin), static_cast<uint16_t>(shape.win), fmatrixPadList,
                   FmatrixMode::FMATRIX_LEFT);
        SetLoadDataPaddingValue(static_cast<SrcT>(0));

        // ★A1 预载 batch0（预载侧 al1）：半区 id = a1Pong_ 翻转前值
        const uint32_t preloadId = a1Pong_;
        a1Pong_ = !a1Pong_;
        LocalTensor<SrcT> al1(TPosition::A1, bl1Bytes + preloadId * al1HalfBytes, al1Elems);
        // A1 半区背压（MTE1_MTE2，第十二轮续改动1：MTE1→MTE2 方向）：等 2 个 batch 前
        // 同半区的 load2d（MTE1 流）全部排空——Mmad 未完但 L1→L0 搬完即可覆写
        // （A1 单 batch 布局，半区隔 batch 复用；Init 预置首两 batch 免等）
        WaitFlag<HardEvent::MTE1_MTE2>(MTE2FlagDy(preloadId));
        LoadA1Dy(al1, config, cRange, 0);
        // A1 半区就绪（即时对，S 挂 MTE2 队列装载后）
        SetFlag<HardEvent::MTE2_MTE1>(MTE2FlagDy(preloadId));

        // 计算侧半区 id（循环外声明，循环内不 shadow）：首批计算用预载半区
        uint32_t a1ComputeId = preloadId;

        // ★A1 预载主循环：先发本批（batchIdx）的 A1 MTE2 装载（抢在上一批 load3d 洪流前
        // 入 MTE2 队列），再计算上一批（batchIdx-1，计算侧 al1Compute 半区 a1ComputeId）
        for (uint32_t batchIdx = 1; batchIdx < shape.batch; batchIdx++) {
            const uint32_t loadId = a1Pong_;
            a1Pong_ = !a1Pong_;
            LocalTensor<SrcT> al1(TPosition::A1, bl1Bytes + loadId * al1HalfBytes, al1Elems);
            WaitFlag<HardEvent::MTE1_MTE2>(MTE2FlagDy(loadId));
            LoadA1Dy(al1, config, cRange, batchIdx);
            SetFlag<HardEvent::MTE2_MTE1>(MTE2FlagDy(loadId));

            LocalTensor<SrcT> al1Compute(TPosition::A1, bl1Bytes + a1ComputeId * al1HalfBytes, al1Elems);
            WaitFlag<HardEvent::MTE2_MTE1>(MTE2FlagDy(a1ComputeId));
            IterateKL0(shape, batchIdx - 1, cinAlign16, cout16Align, howoWin, al1Compute, bl1, l0c);
            SetFlag<HardEvent::MTE1_MTE2>(MTE2FlagDy(a1ComputeId));

            a1ComputeId = loadId; // 本轮预载半区 → 下一轮计算半区
        }

        // ★循环外计算最后一批（batch-1）
        {
            LocalTensor<SrcT> al1Compute(TPosition::A1, bl1Bytes + a1ComputeId * al1HalfBytes, al1Elems);
            WaitFlag<HardEvent::MTE2_MTE1>(MTE2FlagDy(a1ComputeId));
            IterateKL0(shape, shape.batch - 1, cinAlign16, cout16Align, howoWin, al1Compute, bl1, l0c);
            SetFlag<HardEvent::MTE1_MTE2>(MTE2FlagDy(a1ComputeId));
        }

        // M→FIX 前向（整块一次 fixpipe，即时配对）
        SetFlag<HardEvent::M_FIX>(0);
        WaitFlag<HardEvent::M_FIX>(0);
        DirectOutL0C(l0c, y, yBase, alignedCout, hkwk, dhwk, cRange, shape.cin);
        // FIX→M 释放（跨块：下一块 IterateK 开头的 Wait 消费，末块由 Drain 消费）
        SetFlag<HardEvent::FIX_M>(0);
        // MTE1 管排空信号（跨块 B1 装载背压，下一块 IterateK 入口 Wait 消费，末块由 Drain 消费）
        SetFlag<HardEvent::MTE1_MTE2>(MTE2FlagFmap());
    }

private:
    // load3d 状态免重设配置（第十五轮 load3d 状态外提）：Fmatrix/padding 在 IterateK 块首
    // 设置一次后，LoadDataWithStride 以本配置跳过每命令的重设（默认 {true,true} 为每命令
    // 重设——旧形态即为此，指令数高是 fp32 堵 issueque 的成分之一）
    static constexpr IsResetLoad3dConfig L3D_NO_RESET = {false, false};

    template <BpUtils::InputTensor t>
    __aicore__ inline uint8_t MTE1Flag(bool pingPong) const
    {
        // fmap 01, dy23
        if constexpr (t == BpUtils::InputTensor::FMAP) {
            return pingPong;
        } else if constexpr (t == BpUtils::InputTensor::DY) {
            return pingPong + 2;
        }
    }

    __aicore__ inline uint8_t MTE2FlagDy(bool pingPong) const
    {
        // dy 1/2
        return pingPong + 1;
    }

    __aicore__ inline uint8_t MTE2FlagFmap() const { return 0; }

    // 单 batch 的 L0 搬运与 Mmad 计算循环（dout×howoWin×dk 内层，签名扁平化——
    // A1 计算侧半区 al1 / B1 驻留 bl1 / L0C 单视图 l0c 由调用方构造传入）：
    //   L0A 半区：a0Pong_ 窗尾翻转（flag/基址用翻转前值）
    //   L0B 半区：b0Pong_ 装载后翻转（flag/基址用翻转前值）
    //   flag id：A 侧（dy）= MTE1Flag<DY> = a0Pong_+2，B 侧（fmap）= MTE1Flag<FMAP> = b0Pong_
    //   （Init/Drain 对 M_MTE1 4 id 对称预置/消费，id 分配与旧版 A0/B2 互换不破坏配平）
    // 通道量统一（第十六轮裁决）：cinAlign16 = 16·cin16G 单轨贯穿 B1 布局
    //   （channelSize/srcOff batch 段距）与 L0 视图（mmad.n/L0B dst 基址/rt/ds）——
    //   尾块/主块一套逻辑，双轨参数已消除
    __aicore__ inline void IterateKL0(const ShapeAttribute& shape, uint32_t batchIdx, uint32_t cinAlign16,
                                      uint32_t coutAlign16, uint32_t kl0HoWoAlign16, const LocalTensor<SrcT>& al1,
                                      const LocalTensor<SrcT>& bl1, const LocalTensor<SrcT>& l0c)
    {
        constexpr uint32_t l0aHalfElems = TOTAL_L0A_SIZE / 2 / sizeof(SrcT);
        constexpr uint32_t l0bHalfElems = TOTAL_L0B_SIZE / 2 / sizeof(SrcT);

        const uint32_t howoTotal = shape.hout * shape.wout;

        for (uint32_t doutIdx = 0; doutIdx < shape.dout; doutIdx++) {
            for (uint32_t howoIdx = 0; howoIdx < howoTotal; howoIdx += kl0HoWoAlign16) {
                const uint32_t howoLen = Std::min(kl0HoWoAlign16, howoTotal - howoIdx);

                LocalTensor<SrcT> l0a(TPosition::A2, a0Pong_ * (TOTAL_L0A_SIZE / 2), l0aHalfElems);

                const uint8_t flagA = MTE1Flag<BpUtils::InputTensor::DY>(a0Pong_);
                WaitFlag<HardEvent::M_MTE1>(flagA);

                LoadL0Dy(shape, al1, l0a, coutAlign16, doutIdx, howoIdx, howoLen);

                SetFlag<HardEvent::MTE1_M>(flagA);
                WaitFlag<HardEvent::MTE1_M>(flagA);

                for (uint32_t dk = 0; dk < shape.dk; dk++) {
                    // 反算当前dk对应的fmap的dIn索引
                    const int32_t dIn = static_cast<int32_t>(doutIdx + dk) - static_cast<int32_t>(shape.dPad);
                    if (dIn < 0 || dIn >= static_cast<int32_t>(shape.din)) {
                        continue; // 该 dk 平面全 pad（段内部分跳过）
                    }

                    const uint8_t flagB = MTE1Flag<BpUtils::InputTensor::FMAP>(b0Pong_);
                    LocalTensor<SrcT> l0b(TPosition::B2, b0Pong_ * (TOTAL_L0B_SIZE / 2), l0bHalfElems);

                    WaitFlag<HardEvent::M_MTE1>(flagB);

                    LoadL0Fmap(shape, bl1, l0b, batchIdx, cinAlign16, static_cast<uint32_t>(dIn), howoIdx, howoLen);
                    b0Pong_ = !b0Pong_;

                    SetFlag<HardEvent::MTE1_M>(flagB);
                    WaitFlag<HardEvent::MTE1_M>(flagB);

                    MmadParams mmad;
                    mmad.m = coutAlign16;
                    mmad.n = cinAlign16 * shape.hk * shape.wk;
                    mmad.k = howoLen;
                    const uint32_t firstDout = shape.dPad > dk ? shape.dPad - dk : 0;
                    mmad.cmatrixInitVal = (batchIdx == 0) && (howoIdx == 0) && (doutIdx == firstDout);
                    // Mmad 目的带 dk 段偏移：L0C 段基址 = cinAlign16·coutAlign16·hkwk·dk 元素
                    // （= 16M·cin16G·dk·hkwk，n 序不变）；B 指针 = 本 (win,dk) 半区基址
                    // （k-row0 即半区起点，无段内偏移）
                    // l0c上按照[dk,hwk,cin1,cout1,cout0,cin0]排布，非常规FZ格式
                    Mmad(l0c[cinAlign16 * coutAlign16 * shape.hk * shape.wk * dk], l0a, l0b, mmad);
                    SetFlag<HardEvent::M_MTE1>(flagB);
                }
                SetFlag<HardEvent::M_MTE1>(flagA);
                a0Pong_ = !a0Pong_;
            }
        }
    }

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
        const uint32_t alignedCout = config.tiling.singleShapeAligned16Cout;
        const uint32_t dhowo = shape.dout * shape.hout * shape.wout;
        const uint32_t paddedDhowo = (dhowo + 15) / 16 * 16; // A1 行距 16 对齐（见 IterateK 注释）
        Dn2NzParams dn2nz;
        dn2nz.dnNum = 1;                   // ★单 batch 搬运（修改2）
        dn2nz.dValue = cRange.coutLength;  // 真实 co 段长（B1 同构取真实段长）
        dn2nz.nValue = dhowo;              // dhowo 合轴全量（真实行数，pad 行不写）
        dn2nz.srcDnMatrixStride = dhowo;   // dnNum=1 不生效，语义完整保留
        dn2nz.srcDValue = dhowo;           // co 行距
        dn2nz.dstNzC0Stride = paddedDhowo; // A1 行距 = padded（srcStride 整除性）
        dn2nz.dstNzNStride = 1;
        dn2nz.dstNzMatrixStride = paddedDhowo * alignedCout; // dnNum=1 不生效
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
    __aicore__ inline void LoadL0Dy(const ShapeAttribute& shape, const LocalTensor<SrcT>& al1,
                                    const LocalTensor<SrcT>& l0a, uint32_t coutAlign16, uint32_t doutIdx,
                                    uint32_t howoIdx, uint32_t howoLen)
    {
        const uint32_t dhowo = shape.dout * shape.hout * shape.wout;
        LoadData2DParamsV2 load2d;
        load2d.ifTranspose = 1;
        load2d.mStartPosition = 0;
        load2d.kStartPosition = 0;
        // 尾窗无需在 L1 上把 howo 补到 16 对齐（TODO 关闭，第九轮论证）：A1 行距
        // paddedDhowo=Ceil(dhowo,16) 保证 srcStride=CeilDiv(dhowo,16) 恒整数；尾窗
        // howoLen<16 时 k∈[howoLen,16) 读 pad/garbage 行，Mmad k=howoLen 不消费
        load2d.mStep = Ops::Base::CeilDiv<uint32_t>(howoLen, BLOCK_CUBE);
        load2d.kStep = static_cast<uint16_t>(coutAlign16 / C0<SrcT>());
        load2d.srcStride = Ops::Base::CeilDiv<uint32_t>(dhowo, BLOCK_CUBE);
        load2d.dstStride = static_cast<uint16_t>(load2d.kStep / 2);
        const uint32_t srcOff = (doutIdx * shape.hout * shape.wout + howoIdx) * BpUtils::C0<SrcT>();
        LoadData(l0a, al1[srcOff], load2d);
    }

    // ---------- B 侧：B1(fmap 合轴驻留) → L0B（load3d 合轴 tile，第十轮层1'） ----------
    // S4'：每 tap 一条命令 + rt=cinAlign16/C0（= 2*cin16G，repeat 配对按 +16 槽连续铺块
    // ——probe_r10 块级落位验证），替代 g16×tap 双循环（命令数/窗 cin16G·hkwk → hkwk）。
    // 通道视图 g' = c1g*din + d（平面基址 g'*hwIn*C0 均匀），d 平面选择走 L1 源偏移
    // srcOff = (batch 段 + dIn)*hwIn*C0（c1g=0 基，rt 的 rs=din*hkwk 自动步进 c1g）。
    // ★块序变化（消费端适配）：块 B = cin16G*tap + g（tap 外层，配对连续落位所致）——
    // fixpipe 的 CS/SS 相应重映射（见 DirectOutL0C），y 生产布局不变。
    // ★层2（单命令/窗）不可行证明：fixpipe 的 d%16+16·SS·(d/16) 结构强制 L0C n 轴按
    // ci 16 槽块组织且 d=ci/n=tap 角色锁定（dst=DM·dn+n+DS·d 与 y[co][ci][tap] 唯一
    // 匹配），而 load3d k 轴内禀 [c1][tap][c0] 连续落位 + repeat 单 rs 等差游走无法
    // 产出该块序（跨 tap 需非等差跳步）——三重约束下 hkwk 条/窗为下界。
    // ★通道量统一（第十六轮专家裁决）：channelSize 与 srcOff 的 batch 段步距均按本块
    // cinAlign16（B1 布局 c1g 总宽 = 16·cin16G，与 LoadL1Fmap dstNzMatrixStride 同源），
    // 与 rt/ds/dst 基址同一量纲——尾块/主块一套逻辑，双轨参数消除。语义收益：视图
    // din×cinAlign16 的 c1g 组数 = cinAlign16/C0 恰等于 rt，repeat 走满全视图——
    // 不再有"超出视图的 repeat"（旧 alignedCin 口径下尾块 rt < 视图组数，读 B1 尾部/
    // 邻区 garbage 落块上半）；cinLength 非 C0 倍数时尾组上 8 槽 garbage 依旧由
    // fixpipe nSize=真实 cinLength 截断（cinLength≤16*g+8 上界保证）
    // ★load3d 状态外提（第十五轮）：Fmatrix/padding 已在 IterateK 块首设置一次，本函数
    // 不再设置；LoadDataWithStride 模板参 L3D_NO_RESET（{false,false}）免每命令重设
    __aicore__ inline void LoadL0Fmap(const ShapeAttribute& shape, const LocalTensor<SrcT>& bl1,
                                      const LocalTensor<SrcT>& l0b, uint32_t batchIdx, uint32_t cinAlign16,
                                      uint32_t dIn, uint32_t howoIdx, uint32_t howoLen)
    {
        const uint32_t hwIn = shape.hin * shape.win;
        const uint32_t hkwk = shape.hk * shape.wk;
        const uint32_t dhwin = shape.din * hwIn;

        LoadData3DParamsV2<SrcT> load3d;
        load3d.l1H = shape.hin;
        load3d.l1W = shape.win;
        load3d.padList[0] = shape.wPad; // left
        load3d.padList[1] = shape.wPad; // right
        load3d.padList[2] = shape.hPad; // top（引擎 bL1PadUp）
        load3d.padList[3] = shape.hPad; // bottom（专家版；历史版为 255/DEFAULT_PAD_DOWN，
                                        // 回归差异时以此处为二分点）
        load3d.channelSize = shape.din * cinAlign16; // 合轴通道视图 [c1g][d]（本块 16 对齐宽）
        load3d.kExtension = C0<SrcT>();              // 单 tap 窗（kExt 多 tap 的 k 轴序
                                                     // [c1][tap][c0] 与 FZ 16 槽不兼容）
        load3d.kStartPt = 0;
        load3d.mStartPt = howoIdx;   // howo 窗起点
        load3d.mExtension = howoLen; // 真实窗宽（引擎 baseUseK 先例）
        load3d.strideW = 1;
        load3d.strideH = 1;
        load3d.filterW = shape.wk;
        load3d.filterH = shape.hk;
        load3d.dilationFilterW = 1;
        load3d.dilationFilterH = 1;
        load3d.enTranspose = false;

        // rs = din*hkwk：repeat r 源窗口 +r*rs*C0 = (c1g r, 同 d, 同 tap)（跨通道自动进位）；
        // rt = cinAlign16/C0 = 2*cin16G：配对 (2g, 2g+1) 落块 g（下 8 槽 c1g 偶组 + 上 8 槽
        // c1g 奇组），且 rt = 视图 c1g 组数（repeat 走满全视图，见函数头注释）。
        // （第十五轮笔误修正：曾误写 cinAlign16/sizeof(SrcT) = 4·cin16G，多一倍越界半区）
        LoadDataRepeatParamWithStride rep;
        rep.repeatStride = shape.din * hkwk;
        rep.repeatTime = static_cast<uint8_t>(cinAlign16 / C0<SrcT>());
        rep.repeatMode = 1;
        rep.dstStride = static_cast<uint16_t>(hkwk * (cinAlign16 / BLOCK_CUBE)); // ds=引擎 ShiftCeilM0(baseUseN,n0)
        SetLoadDataRepeatWithStride(rep);

        // 源偏移：batch 段（段距 = dhwin·cinAlign16，与 LoadL1Fmap dstNzMatrixStride 同源）
        // + dIn 平面基址（c1g=0 起，rs 自动步进全 c1g）
        // ★u32 域（第十四轮，板测 check_status overflow 根因修正）：bl1[srcOff] 是
        // LocalTensor 索引，operator[] 入参是 u32——u64 srcOff 传入后函数内联 + 编译器
        // 优化产生 u64/u32 混合乘（寄存器现场 (72<<32)|144 拼接特征），乘积 ~1e23 溢出
        // S64。B1 容量上界 131072 元素（L1 门）→ srcOff ≤ 131072，u32 恒够
        const uint32_t srcOff = batchIdx * dhwin * cinAlign16 + dIn * hwIn * C0<SrcT>();
        const LocalTensor<SrcT> bl1T = bl1[srcOff];
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
            LoadDataWithStride<SrcT, L3D_NO_RESET>(l0b[C0<SrcT>() * cinAlign16 * tap], bl1T, load3d);
        }
    }

    // ---------- fmap GM → B1 合轴驻留（S2：dnNum=batch 一把搬全载） ----------
    // ★B1 布局按本块 cinAlign16（第十六轮专家裁决）：batch 段距 = dhwin·cinAlign16
    // （尾块/主块一套逻辑），与 LoadL0Fmap 的 srcOff batch 段步距/channelSize 同源。
    // B1 分配上界（bl1Elems）由 IterateK 按 max(tiling alignedCin, cinAlign16) 口径给出
    // ——分配=上界、布局=本块实际（见 IterateK 注释）。
    // loadedCinIdx_/loadedCinLength_ 缓存语义不受布局改动影响：命中条件含 cinLength
    // 相等 → cinAlign16 相同 → 兄弟 cout 块（cin-major 块序下同 cin 块）复用 B1 时
    // 布局参数与本块完全一致，读写自洽 ✓
    __aicore__ inline void LoadL1Fmap(const LocalTensor<SrcT>& bl1, const BL1FullLoadConfig& config,
                                      const CoutCinRange& cRange)
    {
        if (loadedCinIdx_ == cRange.cinIdx && loadedCinLength_ == cRange.cinLength) {
            return;
        }
        const ShapeAttribute& shape = config.shape;
        const uint64_t dhwin = static_cast<uint64_t>(shape.din) * shape.hin * shape.win;
        // 本块 16 对齐 cin 段宽（B1 布局量；与 IterateK/LoadL0Fmap 的 cinAlign16 同式）
        const uint32_t cinAlign16 = (cRange.cinLength + 15) / 16 * 16;

        Dn2NzParams dn2nz;
        dn2nz.dnNum = static_cast<uint16_t>(shape.batch);                   // 一把搬全部 batch
        dn2nz.dValue = cRange.cinLength;                                    // 真实段长（引擎 bL1cin1CopyLen）
        dn2nz.nValue = static_cast<uint16_t>(dhwin);                        // din×hin×win 合轴
        dn2nz.srcDnMatrixStride = static_cast<uint64_t>(shape.cin) * dhwin; // batch 步距（NCDHW）
        dn2nz.srcDValue = dhwin;                                            // cin 行距
        dn2nz.dstNzC0Stride = static_cast<uint16_t>(dhwin);                 // 组内行距（与 cin 段宽无关）
        dn2nz.dstNzNStride = 1;
        dn2nz.dstNzMatrixStride = static_cast<uint32_t>(dhwin * cinAlign16); // batch 段距（本块 16 对齐）
        // 源基址：块内 cin 偏移（fmap GM NCDHW 每 cin 行 dhwin）
        const uint64_t offset = static_cast<uint64_t>(cRange.cinIdx) * dhwin;
        DataCopy(bl1, fmap_[offset], dn2nz);

        loadedCinIdx_ = cRange.cinIdx;
        loadedCinLength_ = cRange.cinLength;
    }

    // ---------- L0C 直出（每 dk 段一次 NZ2DN，直写生产布局 y[co][ci][dk*hwK+tap]） ----------
    // S8：fixpipe L0C→GM 直达，COLUMN_MAJOR(NZ2DN)——引擎 LoadL0c2GmNormal 先例
    // （退化 curSingleCoreDk=1：每 dk 段独占 L0C，源固定段基址，无 srcDkStride）
    __aicore__ inline void DirectOutL0C(const LocalTensor<SrcT>& l0c, GlobalTensor<SrcT>& y, uint64_t yBase,
                                        uint32_t alignedCout, uint32_t hkwk, uint32_t dhwk, const CoutCinRange& cRange,
                                        uint32_t cinTotal)
    {
        FixpipeParamsArch3510<CO2Layout::COLUMN_MAJOR> fp;
        fp.params.dnNum = static_cast<uint16_t>(cRange.coutLength); // DN 矩阵数 = co（真实值）
        fp.mSize = static_cast<uint16_t>(dhwk);                     // DN 列 = dhwk（整块一次，第十一轮）
        fp.nSize = static_cast<uint16_t>(cRange.cinLength);         // DN 行 = ci（真实值）
        fp.params.srcNzMatrixStride = 1;                            // C0 单位：co 步进 16 元素
        // ★第十轮块序适配：L0C 块 B = cin16G*tap + g（tap 外层，LoadL0Fmap 配对连续落位）。
        // 块序公式 B = (CS*tap + SS*g)/M → CS = M*cin16G（tap 步进 cin16G 块）、SS = M（g 步进 1 块）。
        // 旧序 B = g16*hkwk + tap 时为 CS=M、SS=M*hkwk。y 生产布局不变（DM/DS/nSize/mSize 不动）
        fp.params.srcNzC0Stride = static_cast<uint16_t>(alignedCout * ((cRange.cinLength + 15) / 16)); // tap 步进
        fp.srcStride = static_cast<uint16_t>(alignedCout); // cin16 组步进（C0 单位）
        fp.dstStride = dhwk;                               // 元素：DN 行长
        fp.params.dstDnMatrixStride = cinTotal * dhwk;     // 元素：相邻 co 的 DN 步进
        fp.quantPre = QuantMode_t::NoQuant;
        fp.unitFlag = 0;
        Fixpipe<SrcT, float, CFG_COLUMN_MAJOR>(y[yBase], l0c, fp);
    }

    bool a0Pong_ = 0;
    bool b0Pong_ = 0;
    bool a1Pong_ = 0;
    GlobalTensor<SrcT> fmap_;
    GlobalTensor<SrcT> dy_;
    uint32_t loadedCinIdx_ = 0;
    uint32_t loadedCinLength_ = 0;
};

} // namespace BpFullLoad

#endif // CONV_BP_BL1_FULLLOAD_COMPUTE_H
