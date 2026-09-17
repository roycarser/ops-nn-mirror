/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file conv_bp_dload_compute.h
 * \brief DLoad（d 轴 MTE2 载入）模板 L0 搬运与 Mmad 计算循环
 *
 * ★布局与协议语义：
 *   S1. fp32 L0A/L0B/L0C 线性布局（M=mmad.m, N=mmad.n, K=mmad.k）：
 *       addrA(m,k) = (k/8)*(8M) + 8m + (k%8)        // L0A: [k/8 块][m(步长8)][k%8]
 *       addrB(k,n) = (k/8)*(8N) + 8n + (k%8)        // L0B: [k/8 块][n(步长8)][k%8]
 *       addrC(m,n) = (n/16)*(16M) + 16m + (n%16)    // L0C: [n/16 块][m(步长16)][n%16]
 *   S2. B1(fmap) 单 batch 合轴驻留 [c1g][d][h][w][c0]（C1 外 D 内，dhw 合轴连续）：
 *       每 batch 一条 Dn2Nz{dnNum=1, dValue=cinLength, nValue=dhwin}。L1 两 bank 布局：
 *       ping/pong 各占 L1 上下 256KB，bank 内 B1 段在前 A1 段紧随——ping/pong 的
 *       MTE1/MTE2 落不同 bank，双流预载并行搬运零 bank 冲突。
 *       半区内 addr(ci,d,hw) = (ci/C0)*(dhwin*C0) + d*(hwIn*C0) + hw*C0 + (ci%C0)
 *       （段宽按本块 cinAlign16 = CeilAlign(cinLength,16)，尾块/主块一套逻辑）
 *   S3. A1(dy) 合轴驻留 [co1g][dhowo][co0]（D 在 C1 内，dhowo 合轴连续；单 batch 半区）：
 *       addr(co,pt) = (co/C0)*(paddedDhowo*C0) + pt*C0 + (co%C0)
 *       LoadA1Dy = 每 batch 一条 Dn2Nz{dnNum=1, dValue=coutLength, nValue=dhowo}（半区
 *       承载 batch 维）——B1/A1 双流完全同构
 *   S4. load3d(B1→L0B) 合轴 tile：每 (窗,dk,tap) 一条命令（kExt=C0、kStartPt=tap*C0）
 *       + rt=cinAlign16/C0（=2*cin16G，配对 (2g,2g+1) 落块 g、+16 槽连续铺块）、
 *       rs=din*hkwk、ds=nExt/16、dst=l0b[128*cin16G*tap]（半区基址 + tap 段内偏移）；
 *       块序 B = cin16G*tap + g（L0C 侧 n 序另带 dk 段偏移）；Fmatrix/padding 块首
 *       外提设置一次，命令内不重设
 *   S5. load2d(A1→L0A) 单命令大范围转置（winograd 同款参数式）：mStep=Ceil(窗宽,16)、
 *       kStep=M/8（偶数，M 恒 16 对齐）、srcStride=A1 行距/16（m1 组距，单位 128 元素；A1
 *       howo 行距 16 对齐 padding 保证整除）、dstStride=kStep/2；一条命令覆盖整窗 [M][16]
 *   S6. Mmad（dk 内移）：A(m=co,k=howo) × B(k=howo,n=ci 展开) → C(m=co,n)；
 *       m=alignedCout、n=段级 nExt=16*cin16G*hkwk、k=howoLen；dk 段 L0C/L0B n 区不相交
 *       （段基址 16M*cin16G*dk*hkwk 元素），dk 间免背压；cmatrixInitVal=每 dk 段首 Mmad
 *       true（段首=批0/该dk首有效dout/窗0）；garbage n0 槽由 fixpipe nSize 只读真实行
 *       截断
 *   S7. Mmad 目的必须是 TPosition::CO1；LocalTensor(pos, offset, size) 的 size 单位是
 *       【元素】（offset 是字节）
 *   S8. 直出：fixpipe L0C→GM 直达，NZ2DN（COLUMN_MAJOR，引擎 LoadL0c2GmNormal 先例）：
 *       src = 16*MS*dn + 16*CS*n + 16*SS*(d/16) + d%16（C0 单位）、
 *       dst = DM*dn + n + DS*d（元素单位）；dnNum=coutLength、mSize=hkwk、nSize=cinLength
 *       （均真实值）、MS=1、CS=alignedCout、SS=alignedCout*hkwk、DM=cinTotal*dhwK、DS=dhwK、
 *       dk 列偏移 = dk*hkwK。y 输出 = 生产布局 y[co][ci][dhwK] 行主序（=dw 折叠）
 *   S9. 事件链（winograd WinoMMAD 纪律，手动形态）：成对 id（前向就绪/释放背压）不复用、
 *       释放向 Init 预置位、前向同迭代 set-then-wait、End 消费残留（Set/Wait 计数配平）。
 *       MTE1→M + M→MTE1（L0A 双 buf（id 6/7）+ L0B buf 池（id 0..bufCnt-1 ≤ 4，bufCnt
 *       = min(L0B/tile,4) 自适应）前向+背压）、M→FIX + FIX→M（L0C 前向+背压）、
 *       MTE2→MTE1 + MTE1→MTE2（L1 双半区装载就绪+释放背压；B1+A1 双流同 k 段同半区
 *       节奏——归一单链 l1pong，各用 2 id；每通道 id 池上限 8 = QUE_MAX_EVENT，
 *       MTE2_MTE1/MTE1_MTE2 各 2 id + M_MTE1/MTE1_M 的 L0B 池 ≤4 + L0A 6/7，各通道
 *       均 ≤ 8 ✓）。事件链每段恒定（含空段），守卫只落在数据操作上；dk 全 pad 空段
 *       跳过 fixpipe 本体（y 保持 host 清零值，golden=0）
 * ★双流预载链：预载 batch0 的 fmap+dy
 * → 循环{预载本批 fmap+dy（先 fmap 后 dy，fmap 的 nValue=dhwin 更大）→ 计算上一批}
 * → 循环外计算末批；每流：装载前 Wait<MTE1_MTE2>(半区id)（等 2 batch 前同半区
 * load3d/load2d 排空，首两轮 Init 预置）/ 计算后 Set<MTE1_MTE2>(半区id)；装载次数
 * = 计算次数 = batch，全链配平。半区链天然覆盖跨块 MTE2←MTE1 背压（半区粒度：
 * 上一块对同半区的末次 Set 被本块装载前 Wait 消费，pong 位跨块持续保证半区错开）
 *   S10. ★LocalTensor 索引/偏移入参必须 u32：L1/L0 偏移恒 < 2^32
 *       （片上容量上界：L1 512KB/4B = 131072 元素），u64 传入 operator[] 会在内联优化下
 *       触发 S64 标量溢出（check_status overflow）；GM/GlobalTensor 侧偏移才用 u64
 */

#ifndef CONV_BP_DLOAD_COMPUTE_H
#define CONV_BP_DLOAD_COMPUTE_H

#include "basic_api/kernel_basic_intf.h"
#include "op_kernel/math_util.h"
#include "utils/std/algorithm.h"

#include "../util/conv_bp_common_util.h"
#include "conv_bp_dload_config.h"

namespace BpDLoad {

using namespace AscendC;
using BpUtils::C0;
using BpUtils::CoutCinRange;

template <typename SrcT>
class DLoadCompute {
public:
    // GM 张量入口（fmap = 正向输入 NCDHW [batch][cin][din][hin][win]，dy = 反向梯度 NCDHW
    // [batch][cout][dout][hout][wout]）+ config（bBufCnt_ 在此一次计算，tiling 上界口径
    // 跨块恒定——不逐块按本块 cinAlign16 计算）
    __aicore__ inline void Init(const GlobalTensor<SrcT>& fmap, const GlobalTensor<SrcT>& dy, const DLoadConfig& config)
    {
        fmap_ = fmap;
        dy_ = dy;
        // ★L0B buf 池参数（tiling 上界口径，跨块恒定）：
        //   tileBytes = Ceil(kl0HoWo,8)·8·nL0c上界·sizeof，nL0c上界 = CeilAlign(aCin,16)·hkwk
        //   （全块最大 cinAlign16：所有块 cinLength ≤ aCin → cinAlign16 ≤ CeilAlign(aCin,16)，
        //    8 对齐档 aCin=8 时上界 16——与 B1 半区统一口径同式）；
        //   bBufCnt_ = min(TOTAL_L0B_SIZE/tileBytes, 4)（上限 4：L0A 固定 6/7，总 6 id ≤
        //   QUE_MAX_EVENT=8）；buf 物理边界跨块恒定是池背压链的物理基础（同 id buf 跨块
        //   同区间防护、异 id 不相交并行安全）
        const ShapeAttribute& shape = config.shape;
        const DLoadTiling& tiling = config.tiling;
        const uint32_t hkwk = shape.hk * shape.wk;
        const uint32_t dhwk = shape.dk * hkwk;
        const uint32_t l0bTileRows = (tiling.kl0HoWo + 7) / 8 * 8;
        const uint32_t nL0cUpper = (tiling.singleShapeAligned16Cin + 15) / 16 * 16 * hkwk;
        bBufTileBytes_ = (l0bTileRows * nL0cUpper * sizeof(SrcT) + 31) / 32 * 32;
        uint32_t bufCnt = TOTAL_L0B_SIZE / bBufTileBytes_;
        bBufCnt_ = static_cast<uint8_t>(bufCnt > 4 ? 4 : bufCnt);
        ASCENDC_ASSERT(bBufCnt_ >= 2, {
            KERNEL_LOG(KERNEL_ERROR, "dload L0B tile overflow: tile=%uB (need ≤ %uB for 2-buf)", bBufTileBytes_,
                       static_cast<uint32_t>(TOTAL_L0B_SIZE / 2));
        });
        // ★L0C 双 buf ping-pong 门判定（Init 一次，tiling 上界口径跨块恒定——
        //   dhwk/cin16G/cout16G/cout16/cin16 排布空间占用）：
        //   开门 ⟺ 单视图占用 ≤ TOTAL_L0C_SIZE/2（256KB）
        //   ——单视图 L0C 占用 = alignedCout(aCout 上界) × alignedCin(aCin 上界) × dhwk，
        //   即每块 [cout16][cin16][dk][tap] 全 dhw 块序整块视图的字节数。上界口径同时
        //   保证：①门判定跨块恒定（不破坏 l0cPong_ 交替序）②半区基址几何跨块一致
        //   （若按本块口径逐块定界，尾块 cinAlign16 < alignedCin 时块间 Mmad/fixpipe
        //   存在数据竞争）；开门时本块单视图门（IterateK 内本块口径）蕴含成立
        //   （半区 ≤ 上界 ≤ L0C/2）。典型场景 cout32/cin32/k3³：半区 108KB、双 buf
        //   216KB ≤ 256KB 开门 ✓。同 kernel shape 恒定 → 门恒定 → 两路径不混用；
        //   半区从 L0C 中间切（与 L1 bank 切法同构）——ping @0 / pong @L0C/2 各 128KB，
        //   门判定即单视图占用 ≤ TOTAL_L0C_SIZE/2
        const uint32_t l0cHalfElems = tiling.singleShapeAligned16Cout * tiling.singleShapeAligned16Cin * dhwk;
        l0cPingpong_ = static_cast<uint64_t>(l0cHalfElems) * sizeof(SrcT) <= TOTAL_L0C_SIZE / 2;

        // 跨半区背压预置位（框架 InitBuffer 同款 + 释放向预置位先例）：
        //   MTE1_MTE2(0/1)：L1 双半区（B1+A1 归一链——fmap/dy 同 k 段同半区节奏装载，
        //     单游标 l1pong_ 单 id 对）
        //   FIX_M(0)：首块 Mmad 前的 fixpipe 等待
        //   M_MTE1(0..bBufCnt_-1)：L0B buf 池释放链首用免等（按实际 bBufCnt_ 循环
        //     预置，账目逐 id 精确）
        //   M_MTE1(6/7)：L0A(dy) 双 buf 释放链首用免等（固定 6/7，与 L0B 池无重叠）
        //   M_MTE1 释放链语义：Wait 前置装载前、Mmad 后只 Set——末次 Set 无消费者，
        //     End 补消费
        SetFlag<HardEvent::MTE1_MTE2>(0);
        SetFlag<HardEvent::MTE1_MTE2>(1);
        // FIX_M 双 id（开门 = 半区 id 0/1 首两块免等；关门 = id0，id1 预置由 End 直接
        // 消费配平）
        SetFlag<HardEvent::FIX_M>(0);
        SetFlag<HardEvent::FIX_M>(1);
        // hf32：按 tiling 标志开 HF32 模式（Mmad 使用前设置；End 收尾关闭——残留开启
        // 会污染同核后续 kernel）
        SetHF32Mode(tiling.hf32Flag);
        for (uint8_t i = 0; i < bBufCnt_; ++i) {
            SetFlag<HardEvent::M_MTE1>(MTE1FlagFmap(i));
        }
        SetFlag<HardEvent::M_MTE1>(MTE1FlagDy(false));
        SetFlag<HardEvent::M_MTE1>(MTE1FlagDy(true));
    }

    // kernel 入口在全部块迭代结束后调用：消费跨半区背压的残留 Set（TQue Reset 残留
    // freeBufEvt 消费先例）——漏掉则残留 flag 会污染同核后续 kernel 的首块装载
    __aicore__ inline void End()
    {
        // 消费残留 Set（Set/Wait 全程配平铁律，防跨 kernel 污染同核后续算子）：
        //   L1 归一链 2 id + L0B 池按 bBufCnt_ 循环 + L0A 6/7——与 Init 预置逐项对称
        WaitFlag<HardEvent::MTE1_MTE2>(0);
        WaitFlag<HardEvent::MTE1_MTE2>(1);
        WaitFlag<HardEvent::FIX_M>(0);
        WaitFlag<HardEvent::FIX_M>(1);
        // hf32：收尾关闭
        SetHF32Mode(false);
        for (uint8_t i = 0; i < bBufCnt_; ++i) {
            WaitFlag<HardEvent::M_MTE1>(MTE1FlagFmap(i));
        }
        WaitFlag<HardEvent::M_MTE1>(MTE1FlagDy(false));
        WaitFlag<HardEvent::M_MTE1>(MTE1FlagDy(true));
    }

    // 单个基本块 [coutRange × cinRange] 的 K 全载计算：
    // L0C 全 dhw 驻留单视图（n 轴 [dk][hkwk][cin16] 块序），dk 内移循环 batch×dout×howo16 窗
    // → dk 段内累加（cmatrixInitVal 段首清零），块尾 fixpipe NZ2DN 整块一次直出。
    //
    // y 布局契约（生产布局）：
    //   y = [cout][cin][dhwK] ND 行主序（全局），dhwK = dk*hwK
    //   本块基址 yBase = coutIdx*cinTotal*dhwK + cinIdx*dhwK（由入口计算）
    //   元素 y[yBase + co*cinTotal*dhwK + ci*dhwK + dk*hwK + tap] = C(co, ci, tap) of 本 dk 段
    //
    // ★B1+A1 双流预载（归一单 l1pong 链）：
    //   预载 batch0 的 fmap→B1 + dy→A1 → for batchIdx=1..batch-1 { 先发 batchIdx 的
    //   fmap+dy DataCopy（MTE2 抢在上一批 load3d 洪流前入队，先 fmap 后 dy——fmap 的
    //   nValue=dhwin 更大）→ 计算上一 batch } → 循环外计算最后 batch。
    //   每流装载次数 = 计算次数 = batch。
    //
    // ★L1 布局（四半区）：B1Ping@0 | B1Pong | A1Ping | A1Pong（32B 对齐逐段）——
    //   B1 半区 = 单 batch fmap [c1g][d][h][w][c0]（段宽本块 cinAlign16），A1 半区 =
    //   单 batch dy [co1g][paddedDhowo][co0]。泛化收益：L1 门不含 batch 因子
    //   （2×单batch B1 + 2×A1 ≤ 512KB，cin 大 batch 大但单 batch 小的场景可跑）。
    //
    // ★事件链时序（L1 归一链——半区 id p∈{0,1}（l1pong），fmap/dy 双流同段同 id）：
    //   装载前 Wait<MTE1_MTE2>(p)（等 2 个 k 段前同半区全部 load3d+load2d 排空，首两轮靠
    //   Init 预置）→ LoadL1Fmap + LoadA1Dy（fmap 先 dy 后）→ Set<MTE2_MTE1>(p)（挂 MTE2
    //   队列两条装载之后——双流都完成才 fire）
    //   → 计算前 Wait<MTE2_MTE1>(p)（双流均就绪）→ IterateKL0（本批全部 load3d/load2d/Mmad）
    //   → 计算后 Set<MTE1_MTE2>(p)（本批 MTE1 流排空即 fire，Mmad 可仍在执行）
    //   Set/Wait 配平（MTE1_MTE2 逐 id，跨 kernel 累计）：
    //   Set = Init(1) + 该半区计算次数 n_p，Wait = 该半区装载次数 n_p + End(1) → 恒配平
    //   归一前提（游标恒等性）：fmap/dy 双流成对无条件装载（无单流路径），共用游标
    //   l1pong 恒有效 ✓；batch=1 空段（循环不进、末批用预载半区）双流同构 ✓。
    //   释放向粗粒度合并（dy 侧多等 fmap 的 load3d）保守无害
    //   跨块背压由半区链天然覆盖（半区粒度）：上一块对同半区的末次 Set 被本块装载前
    //   Wait 消费（pong 位跨块持续保证相邻块半区错开）
    //   其余链：L0B buf 池 + L0A 双 buf 的 M_MTE1 背压（Wait 前置装载前 + Mmad 后
    //   只 Set）、FIX/M_FIX 即时对 + FIX_M 跨块
    __aicore__ inline void IterateK(const DLoadConfig& config, const CoutCinRange& cRange, GlobalTensor<SrcT>& y,
                                    uint64_t yBase)
    {
        const ShapeAttribute& shape = config.shape;
        const DLoadTiling& tiling = config.tiling;
        const uint32_t alignedCout = tiling.singleShapeAligned16Cout;
        const uint32_t alignedCin = tiling.singleShapeAligned16Cin;
        const uint32_t hkwk = shape.hk * shape.wk;
        const uint32_t dhwk = shape.dk * hkwk;

        // ★通道量统一：B1 布局与 L0/L0C 视图统一按本块
        // cinAlign16 = CeilAlign(cinLength,16)（尾块/主块一套逻辑）。
        //   cin16G = Ceil(cinLength/16)（L0B/L0C 的 16 槽块数）；cinAlign16 = 16·cin16G（= nL0c/hkwk）
        //   cout16Align = Ceil(coutLength/16)（块按 alignedCout 切、尾块长 ∈ (16k-16,16k]，恒 = alignedCout）
        const uint32_t cin16G = (cRange.cinLength + 15) / 16;
        const uint32_t cinAlign16 = 16 * cin16G;
        const uint32_t cout16Align = (cRange.coutLength + 15) / 16 * 16;

        // ★howo 窗宽（k 轴）：从 tiling.kl0HoWo 传入。
        // 约束：kl0HoWo 须为 16 的倍数（load3d mStartPt = 窗起点须 16 对齐）
        const uint32_t howoWin = tiling.kl0HoWo;
        ASCENDC_ASSERT(howoWin != 0 && howoWin % 16 == 0,
                       { KERNEL_LOG(KERNEL_ERROR, "dload kl0HoWo invalid: %u (need 16-multiple)", howoWin); });
        // ★容量门（本路径不支持超容量 shape，无 fallback）：
        //   L0C: alignedCout·(16·cin16G·dhwk)·sizeof ≤ TOTAL_L0C_SIZE（256KB，全 dhw 驻留单视图）
        //   L0B: bBufCnt_ ≥ 2 门已在 Init（tiling 上界口径一次计算——本块 tile 实占
        //        ≤ 上界 tileBytes ≤ bBufTileBytes_·bBufCnt_ ≤ L0B 恒成立）
        ASCENDC_ASSERT(static_cast<uint64_t>(alignedCout) * (cinAlign16 * dhwk) * sizeof(SrcT) <= TOTAL_L0C_SIZE, {
            KERNEL_LOG(KERNEL_ERROR, "dload L0C overflow: %u*%u*%uB", alignedCout, cinAlign16 * dhwk,
                       static_cast<uint32_t>(sizeof(SrcT)));
        });

        // ★L1 两 bank 布局：
        //   ping bank @0 / pong bank @TOTAL_L1_SIZE/2 各占 L1 上下 256KB——bank 内 B1 段
        //   在前（基址 + 0）、A1 段紧随（基址 + bl1HalfBytes）。ping 与 pong 的 MTE1/MTE2
        //   访问落在不同 bank：双流预载下本批 MTE1 写 pong 与上批 MTE2 读 ping 并行时
        //   bank 冲突消除。
        //   B1 段 = 单 batch [c1g][d][h][w][c0]、A1 段 = 单 batch [co1g][paddedDhowo][co0]
        //   （元素上界 alignedCout·dout·CeilAlign(howoTotal,16) ≥ 实际占用，paddedDhowo
        //   = Ceil(dhowo,16) 为 A1 实际行距，见 LoadA1Dy dstNzC0Stride）——batch 维由
        //   bank 基址（l1pong）承载。
        // ★B1 段大小 = 跨块统一口径 dhwin·max(tiling alignedCin, 本块 cinAlign16)——
        //   该 max 在同核块序列内恒定（aCin≥16 档：所有块 cinAlign16 ≤ aCin → 恒 aCin；
        //   8 对齐档：所有块 cinLength ≤ 8 → cinAlign16 恒 16 → 恒 16），段物理边界跨块
        //   稳定是半区背压链的物理基础：同 pong bank 跨块同区间（跨块 Set/Wait 续链防护）、
        //   异 bank 不相交（块 N+1 预载与块 N 残留 load3d 并行安全）——若按本块
        //   cinAlign16 逐块定界（如尾块 16 < 主块 32），块间边界漂移致物理重叠
        //   且背压链不设防（上一块未用目标半区时 Wait 消费 Init 预置直通）→ 数值污染。
        //   B1 段内实际写入 = Ceil(cinLength/C0)·dhwin·C0
        //   ≤ dhwin·cinAlign16 ≤ 段大小（布局紧凑、分配统一）
        const uint32_t dhwin = shape.din * shape.hin * shape.win;
        const uint32_t bl1CinSpan = alignedCin > cinAlign16 ? alignedCin : cinAlign16;
        const uint32_t bl1HalfElems = dhwin * bl1CinSpan;
        const uint32_t dhowoTotal = shape.dout * shape.hout * shape.wout;
        const uint32_t al1Elems = alignedCout * shape.dout *
                                  Ops::Base::CeilAlign<uint32_t>(shape.hout * shape.wout, BLOCK_CUBE);
        // 32B 对齐保留：bl1HalfBytes 向上取整 32（A1 段起点 = bank 基址 + bl1HalfBytes
        // 天然对齐；bl1HalfBytes/al1HalfBytes 语义 = bank 内段偏移/段容量）
        const uint32_t bl1HalfBytes = static_cast<uint32_t>((bl1HalfElems * sizeof(SrcT) + 31) / 32 * 32);
        const uint32_t al1HalfBytes = (al1Elems * sizeof(SrcT) + 31) / 32 * 32;
        // ★L1 容量门（bank 口径）：单 bank 装下 B1 段 + A1 段 ≤ TOTAL_L1_SIZE/2；
        // 不含 batch 因子（大 batch 场景可算）
        ASCENDC_ASSERT(static_cast<uint64_t>(bl1HalfBytes) + al1HalfBytes <= TOTAL_L1_SIZE / 2, {
            KERNEL_LOG(KERNEL_ERROR, "dload L1 bank overflow: %u+%uB > %uB", bl1HalfBytes, al1HalfBytes,
                       static_cast<uint32_t>(TOTAL_L1_SIZE / 2));
        });

        LocalTensor<SrcT> l0c(TPosition::CO1, 0, TOTAL_L0C_SIZE / sizeof(SrcT)); // 关门默认全区视图
        // ★L0C 半区分派（门判定在 Init，l0cPingpong_ 跨块恒定）——开门：本块 Mmad 写
        //   半区 p、块尾 fixpipe 读 p，与下块 Mmad 写 !p 并行（块级 ping-pong，FIX_M
        //   背压深度 2 等"上上块同半区"）；关门：单视图全区（FIX_M id 0 深度 1 等上一块）。
        //   半区从 L0C 中间切（与 L1 bank 同构，128KB 对齐无忧）：基址 = p × TOTAL_L0C_SIZE/2；
        //   每半区仍为全 nFull（[dk][tap][cin16] 全 dhw 块序）整块视图——不按 dk 段拆、
        //   不改 L0C 布局与 fixpipe 参数，仅基址半区化
        const uint8_t l0cId = l0cPingpong_ ? l0cPong_ : 0; // 本块半区（开门翻转前值；关门恒 0）
        if (l0cPingpong_) {
            l0cPong_ = !l0cPong_;
            // 半区基址 = l0cId × TOTAL_L0C_SIZE/2；size 取 L0C 剩余
            // 容量（≥ 实际占用——B2 位置"size 精确=用量"仿真器容量校验陷阱的同款规避）
            l0c = LocalTensor<SrcT>(TPosition::CO1, l0cId * (TOTAL_L0C_SIZE / 2),
                                    (TOTAL_L0C_SIZE - l0cId * (TOTAL_L0C_SIZE / 2)) / sizeof(SrcT));
        }

        // ★load3d 状态外提：Fmatrix/padding 参数仅依赖 shape（块内恒定），块首设置一次；
        // LoadL0Fmap 的 LoadDataWithStride 以 L3D_NO_RESET（{false,false}）免每命令重设
        const uint8_t fmatrixPadList[4] = {static_cast<uint8_t>(shape.wPad), static_cast<uint8_t>(shape.wPad),
                                           static_cast<uint8_t>(shape.hPad), static_cast<uint8_t>(shape.hPad)};
        SetFmatrix(static_cast<uint16_t>(shape.hin), static_cast<uint16_t>(shape.win), fmatrixPadList,
                   FmatrixMode::FMATRIX_LEFT);
        SetLoadDataPaddingValue(static_cast<SrcT>(0));

        // ★batch 循环装载/计算合并为单循环一份代码（游标延迟形态）——AscendC 全内联下
        //   装载/计算文本多份会放大目标码体积与编译时间，合并后各只保留一份：
        //     for (i = 0; i <= batch; ++i) { if (i < batch) 装载 i（唯一一份）;
        //                                  if (i > 0) 计算 i-1（唯一一份） }
        //   执行序：i=0 仅装载（= 预载 b0）；i=1..batch-1 装载 i → 计算 i-1（装载先发射
        //   = 预载语义保持，硬件重叠靠事件链不变）；i=batch 仅计算（= 末批）。
        //   装载游标 l1LoadPong = l1pong_ 翻转前值；计算游标 = 上一轮装载后的
        //   pong（首批计算 = 预载半区）。空段（batch=1）行为等价：i=0 装载、i=1 计算。
        //   L1 归一链配平账目（装载次数 = 计算次数 = batch，半区序一致）
        uint32_t l1ComputePong = l1pong_; // 首批计算半区 = i=0 装载半区（装载翻转前同值）
        uint32_t l1LoadedPong = l1pong_;  // 本迭代装载半区（须取翻转前值——i=batch 无装载
                                          // 不更新；若误取翻转后值，奇数次装载后错位一格，
                                          // Wait<MTE2_MTE1> 消费不到本批 Set）

        WaitFlag<HardEvent::FIX_M>(l0cId);
        for (uint32_t loadIdx = 0; loadIdx <= shape.batch; loadIdx++) {
            if (loadIdx < shape.batch) {
                // ---- 装载段（唯一一份）：fmap 先 dy 后，双流同半区归一链（l1pong）----
                const uint32_t l1LoadPong = l1pong_;
                l1pong_ = !l1pong_;
                l1LoadedPong = l1LoadPong; // 记录本迭代装载半区（i=batch 无装载不更新）
                {
                    LocalTensor<SrcT> bl1;
                    LocalTensor<SrcT> al1;
                    GetL1Buf(l1LoadPong != 0, bl1HalfBytes, bl1HalfElems, al1Elems, bl1, al1);
                    // L1 半区背压（归一链）：等 2 个 k 段前同半区的全部 load3d+load2d
                    // （MTE1 流）排空（跨块续链：上一块对同半区的末次 Set——首用靠
                    // Init 预置）
                    WaitFlag<HardEvent::MTE1_MTE2>(l1LoadPong);
                    LoadL1Fmap(bl1, config, cRange, loadIdx);
                    LoadA1Dy(al1, config, cRange, loadIdx);
                    // 双流装载就绪（归一单 Set：挂 MTE2 队列两条装载之后，两条都完成才 fire）
                    SetFlag<HardEvent::MTE2_MTE1>(l1LoadPong);
                }
            }
            if (loadIdx > 0) {
                // ---- 计算段（唯一一份）：B1/A1 计算侧半区（= 上一轮装载半区）----
                LocalTensor<SrcT> bl1Compute;
                LocalTensor<SrcT> al1Compute;
                GetL1Buf(l1ComputePong != 0, bl1HalfBytes, bl1HalfElems, al1Elems, bl1Compute, al1Compute);
                // 归一单 Wait：双流装载均就绪
                WaitFlag<HardEvent::MTE2_MTE1>(l1ComputePong);
                IterateKL0(shape, loadIdx - 1, cinAlign16, cout16Align, howoWin, al1Compute, bl1Compute, l0c);
                // 归一单 Set：本 k 段全部 load3d（读 B1）+ load2d（读 A1）排空后 fire——
                // 两 L1 区域同被保护（粗粒度但安全：dy 侧多等 fmap 的 load3d，保守无害）
                SetFlag<HardEvent::MTE1_MTE2>(l1ComputePong);
                // 本轮装载半区 → 下一轮计算半区（取本迭代装载的半区 l1LoadedPong；
                // i=batch 无装载不更新，游标不再使用）
                l1ComputePong = l1LoadedPong;
            }
        }

        // M→FIX 前向（整块一次 fixpipe，即时配对；开门用本块半区 id，关门 id 0）
        SetFlag<HardEvent::M_FIX>(l0cId);
        WaitFlag<HardEvent::M_FIX>(l0cId);
        DirectOutL0C(l0c, y, yBase, alignedCout, hkwk, dhwk, cRange, shape.cin);
        // FIX→M 释放（跨块：开门 = 下下块同半区 Wait 消费（深度 2）/关门 = 下一块消费
        // （深度 1），末块由 End 消费）
        SetFlag<HardEvent::FIX_M>(l0cId);
    }

private:
    // load3d 状态免重设配置：Fmatrix/padding 在 IterateK 块首设置一次后，
    // LoadDataWithStride 以本配置跳过每命令的重设（默认 {true,true} 为每命令重设，
    // 指令数高，fp32 下易堵 issueque）
    static constexpr IsResetLoad3dConfig L3D_NO_RESET = {false, false};

    __aicore__ inline uint8_t MTE1FlagDy(bool pingPong) const
    {
        // L0A(dy) 双 buf 固定 6/7（让出低 id 段给 L0B buf 池）
        return pingPong + 6;
    }

    __aicore__ inline uint8_t MTE1FlagFmap(uint32_t bufIdx) const
    {
        // L0B(fmap) buf 池 0..bBufCnt_-1（bufCnt ≤ 4，与 L0A 的 6/7 无重叠）
        return static_cast<uint8_t>(bufIdx);
    }

    // ★L1 两 bank 取段：
    // PingPong 按 L1/2 为界——ping bank @0 / pong bank @TOTAL_L1_SIZE/2（各 256KB），
    // bank 内 B1 段在前（+0）、A1 段紧随（+bl1HalfBytes）——ping 与 pong 的 MTE1/MTE2
    // 访问落在不同 bank，双流预载下本批 MTE1 写 pong 与上批 MTE2 读 ping 并行零 bank
    // 冲突。bl1HalfBytes 语义 = bank 内 B1→A1 段偏移（32B 对齐——A1 段起点天然对齐）；
    // pong 基址 = L1SIZE/2 单点保证（本 helper 是 L1 布局唯一构造入口，四处调用点共用）
    __aicore__ inline void GetL1Buf(bool pong, uint32_t bl1HalfBytes, uint32_t bl1HalfElems, uint32_t al1Elems,
                                    LocalTensor<SrcT>& bl1, LocalTensor<SrcT>& al1) const
    {
        const uint32_t bankBytes = static_cast<uint32_t>(pong) * (TOTAL_L1_SIZE / 2);
        bl1 = LocalTensor<SrcT>(TPosition::A1, bankBytes, bl1HalfElems);
        al1 = LocalTensor<SrcT>(TPosition::A1, bankBytes + bl1HalfBytes, al1Elems);
    }

    // 单 batch 的 L0 搬运与 Mmad 计算循环（dout×howoWin×dk 内层，签名扁平化——
    // A1 计算侧半区 al1 / B1 计算侧半区 bl1（均单 batch，半区基址承载 batch 维）/
    // L0C 单视图 l0c 由调用方构造传入）：
    //   L0A 双 buf：a0Pong_ 窗尾翻转（flag/基址用翻转前值）——A 侧 load2d 已单命令化
    //     非瓶颈，维持双 buf
    //   L0B buf 池：bBuf_ 装载后轮转 (bBuf_+1)%bufCnt——池深度 bufCnt =
    //     min(L0B/tileBytes, 4)，装载第 i+bufCnt 个 tile 前 Wait<M_MTE1>(bufIdx) 等第 i
    //     个 Mmad 完成（池深度语义）；tile tensor = (B2, bufIdx·tileBytes, tileElems)
    //     紧凑排布
    //   flag id：B 侧（fmap）= MTE1FlagFmap(bufIdx) = bufIdx ∈ [0, bBufCnt_)，
    //           A 侧（dy）= MTE1Flag<DY> = a0Pong_+6（固定 6/7，与池无重叠）
    // 通道量统一：cinAlign16 = 16·cin16G 单轨贯穿 B1 布局
    //   （channelSize/srcOff）与 L0 视图（mmad.n/L0B dst 基址/rt/ds）——尾块/主块一套逻辑
    __aicore__ inline void IterateKL0(const ShapeAttribute& shape, uint32_t batchIdx, uint32_t cinAlign16,
                                      uint32_t coutAlign16, uint32_t kl0HoWoAlign16, const LocalTensor<SrcT>& al1,
                                      const LocalTensor<SrcT>& bl1, const LocalTensor<SrcT>& l0c)
    {
        constexpr uint32_t l0aHalfElems = TOTAL_L0A_SIZE / 2 / sizeof(SrcT);
        // L0B tile 用 Init 的上界口径成员：bBufTileBytes_/bBufCnt_ 跨块恒定——buf 物理
        // 边界跨块稳定是池背压链的物理基础；本块 tile 实占
        // Ceil(kl0HoWo,8)·8·cinAlign16·hkwk ≤ 上界 tileBytes 恒成立

        const uint32_t howoTotal = shape.hout * shape.wout;

        for (uint32_t doutIdx = 0; doutIdx < shape.dout; doutIdx++) {
            for (uint32_t howoIdx = 0; howoIdx < howoTotal; howoIdx += kl0HoWoAlign16) {
                const uint32_t howoLen = Std::min(kl0HoWoAlign16, howoTotal - howoIdx);

                LocalTensor<SrcT> l0a(TPosition::A2, a0Pong_ * (TOTAL_L0A_SIZE / 2), l0aHalfElems);

                const uint8_t flagA = MTE1FlagDy(a0Pong_);
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

                    // buf 池轮转（翻转前值 = 本次 tile 的 buf 索引/基址/flag id）
                    const uint32_t bBuf = bBuf_;
                    const uint8_t flagB = MTE1FlagFmap(bBuf);
                    // ★tensor size 取 L0B 剩余容量（≥ tile 实占）——size 精确=tile 用量
                    // 形态会触发仿真器容量校验边界行为挂死；基址 bufIdx·tileBytes 保持紧凑
                    LocalTensor<SrcT> l0b(TPosition::B2, bBuf * bBufTileBytes_,
                                          (TOTAL_L0B_SIZE - bBuf * bBufTileBytes_) / sizeof(SrcT));

                    // 池深度背压：等第 i+bufCnt 次（回到本 buf）前、第 i 次 Mmad 完成
                    WaitFlag<HardEvent::M_MTE1>(flagB);

                    LoadL0Fmap(shape, bl1, l0b, cinAlign16, static_cast<uint32_t>(dIn), howoIdx, howoLen);
                    bBuf_ = (bBuf_ + 1) % bBufCnt_;

                    SetFlag<HardEvent::MTE1_M>(flagB);
                    WaitFlag<HardEvent::MTE1_M>(flagB);

                    MmadParams mmad;
                    mmad.m = coutAlign16;
                    mmad.n = cinAlign16 * shape.hk * shape.wk;
                    mmad.k = howoLen;
                    const uint32_t firstDout = shape.dPad > dk ? shape.dPad - dk : 0;
                    mmad.cmatrixInitVal = (batchIdx == 0) && (howoIdx == 0) && (doutIdx == firstDout);
                    // Mmad 目的带 dk 段偏移：L0C 段基址 = cinAlign16·coutAlign16·hkwk·dk 元素
                    // （= 16M·cin16G·dk·hkwk，n 序不变）；B 指针 = 本 (win,dk) tile buf 基址
                    // （k-row0 即 buf 起点，无段内偏移）
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
    // dy 每 batch 单独装载——只有 fmap(B1) 承诺 k 轴全载（含 batch），
    // dy 只承诺 dhowo 轴全载、batch 轴不保证（每 batch 一条 Dn2Nz{dnNum=1}，半区承载 batch 维）。
    // d 轴=co 跨 co1g 组落 C0/C1 槽（与 B1 的 dValue=cinLength 同机制）。
    // A1 半区目的布局（单 batch）：
    //   addr(co, pt) = (co/C0)*(paddedDhowo*C0) + pt*C0 + (co%C0)
    // LoadL0Dy 源偏移不含 batch 项（A1 内 batch 维消失），paddedDhowo 行距 16 对齐
    __aicore__ inline void LoadA1Dy(const LocalTensor<SrcT>& al1, const DLoadConfig& config, const CoutCinRange& cRange,
                                    uint32_t batchIdx)
    {
        const ShapeAttribute& shape = config.shape;
        const uint32_t alignedCout = config.tiling.singleShapeAligned16Cout;
        const uint32_t dhowo = shape.dout * shape.hout * shape.wout;
        const uint32_t paddedDhowo = (dhowo + 15) / 16 * 16; // A1 行距 16 对齐（见 IterateK 注释）
        Dn2NzParams dn2nz;
        dn2nz.dnNum = 1;                   // ★单 batch 搬运（半区承载 batch 维）
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

    // load2d 单命令大范围转置——一条命令搬整窗 [m=co 段][k=howo 窗]（逐 (m1,k1) 块循环
    // 需每窗 (M/8)·Ceil(K/8) 条 → 1 条）。参数式沿用 winograd fp32 转置：
    //   mStep = Ceil(窗宽,16)（k 方向 16 组数；kl0HoWo=16 现值 → 恒 1）
    //   kStep = M/8（m 方向 C0 组数；M=alignedCout 恒 16 对齐 → kStep 偶数，硬件约束 ✓）
    //   srcStride = m1 组距/128 元素 = paddedDhowo/16（k1 组内 16 howo 连续、硬件按 1 单位推进）
    //   dstStride = kStep/2
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
        // 尾窗无需在 L1 上把 howo 补到 16 对齐：A1 行距
        // paddedDhowo=Ceil(dhowo,16) 保证 srcStride=CeilDiv(dhowo,16) 恒整数；尾窗
        // howoLen<16 时 k∈[howoLen,16) 读 pad/garbage 行，Mmad k=howoLen 不消费
        load2d.mStep = Ops::Base::CeilDiv<uint32_t>(howoLen, BLOCK_CUBE);
        load2d.kStep = static_cast<uint16_t>(coutAlign16 / C0<SrcT>());
        load2d.srcStride = Ops::Base::CeilDiv<uint32_t>(dhowo, BLOCK_CUBE);
        load2d.dstStride = static_cast<uint16_t>(load2d.kStep / 2);
        const uint32_t srcOff = (doutIdx * shape.hout * shape.wout + howoIdx) * BpUtils::C0<SrcT>();
        LoadData(l0a, al1[srcOff], load2d);
    }

    // ---------- B 侧：B1(fmap 合轴驻留) → L0B（load3d 合轴 tile） ----------
    // 每 tap 一条命令（命令数/窗 = hkwk）+ rt=cinAlign16/C0（= 2*cin16G，repeat 配对按
    // +16 槽连续铺块），dst = l0b[128*cin16G*tap]（半区基址 + tap 段内偏移）。
    // 通道视图 g' = c1g*din + d（平面基址 g'*hwIn*C0 均匀），d 平面选择走 L1 源偏移
    // srcOff = (batch 段 + dIn)*hwIn*C0（c1g=0 基，rt 的 rs=din*hkwk 自动步进 c1g）。
    // ★块序（消费端适配）：块 B = cin16G*tap + g（tap 外层，配对连续落位所致）——
    // fixpipe 的 CS/SS 相应重映射（见 DirectOutL0C），y 生产布局不变。
    // ★单命令/窗不可行：fixpipe 的 d%16+16·SS·(d/16) 结构强制 L0C n 轴按
    // ci 16 槽块组织且 d=ci/n=tap 角色锁定（dst=DM·dn+n+DS·d 与 y[co][ci][tap] 唯一
    // 匹配），而 load3d k 轴内禀 [c1][tap][c0] 连续落位 + repeat 单 rs 等差游走无法
    // 产出该块序（跨 tap 需非等差跳步）——三重约束下 hkwk 条/窗为下界。
    // ★通道量统一：channelSize 与 srcOff 的 batch 段步距均按本块
    // cinAlign16（B1 布局 c1g 总宽 = 16·cin16G，与 LoadL1Fmap dstNzMatrixStride 同源），
    // 与 rt/ds/dst 基址同一量纲——尾块/主块一套逻辑，双轨参数消除。语义收益：视图
    // din×cinAlign16 的 c1g 组数 = cinAlign16/C0 恰等于 rt，repeat 走满全视图——
    // 不存在超出视图的 repeat；cinLength 非 C0 倍数时尾组上 8 槽 garbage 依旧由
    // fixpipe nSize=真实 cinLength 截断（cinLength≤16*g+8 上界保证）
    // ★load3d 状态外提：Fmatrix/padding 已在 IterateK 块首设置一次，本函数
    // 不再设置；LoadDataWithStride 模板参 L3D_NO_RESET（{false,false}）免每命令重设
    __aicore__ inline void LoadL0Fmap(const ShapeAttribute& shape, const LocalTensor<SrcT>& bl1,
                                      const LocalTensor<SrcT>& l0b, uint32_t cinAlign16, uint32_t dIn, uint32_t howoIdx,
                                      uint32_t howoLen)
    {
        const uint32_t hwIn = shape.hin * shape.win;
        const uint32_t hkwk = shape.hk * shape.wk;

        LoadData3DParamsV2<SrcT> load3d;
        load3d.l1H = shape.hin;
        load3d.l1W = shape.win;
        load3d.padList[0] = shape.wPad;              // left
        load3d.padList[1] = shape.wPad;              // right
        load3d.padList[2] = shape.hPad;              // top（引擎 bL1PadUp）
        load3d.padList[3] = shape.hPad;              // bottom
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
        LoadDataRepeatParamWithStride rep;
        rep.repeatStride = shape.din * hkwk;
        rep.repeatTime = static_cast<uint8_t>(cinAlign16 / C0<SrcT>());
        rep.repeatMode = 1;
        rep.dstStride = static_cast<uint16_t>(hkwk * (cinAlign16 / BLOCK_CUBE)); // ds=引擎 ShiftCeilM0(baseUseN,n0)
        SetLoadDataRepeatWithStride(rep);

        // 源偏移：dIn 平面基址（c1g=0 起，rs 自动步进全 c1g；batch 维由半区基址承载，
        // bl1 即本 batch 半区视图）
        // ★u32 域：bl1[srcOff] 是 LocalTensor 索引，operator[] 入参是 u32——u64 传入后
        // 函数内联 + 编译器优化产生 u64/u32 混合乘，乘积溢出 S64（板测 check_status
        // overflow）。B1 半区容量上界 65536 元素（L1 门/2）→ srcOff ≤ 65536，u32 恒够
        const uint32_t srcOff = dIn * hwIn * C0<SrcT>();
        const LocalTensor<SrcT> bl1T = bl1[srcOff];
        for (uint32_t tap = 0; tap < hkwk; tap++) {
            load3d.kStartPt = static_cast<uint16_t>(tap * BpUtils::C0<SrcT>());
            // ★每个 (win,dk) 的 tile 独占一个 L0B 半区（基址由调用方 pong 位选择），
            // 段内 tap 块紧堆于半区起点（dk 段偏移消失）。
            // k-row1 硬件语义：mExt 跨 8 行时 k-row1 自动落
            // dst 基址+8N（N = 段宽 16*cin16G*hkwk）处（偏移量由 N 决定，与 tap/dstStride
            // 无关）——tile 实占 [半区基址, +16N) = k-row0+k-row1，故容量门 16·nL0c ≤ 半区。
            // L0C 侧 n 序不变：cin16G*(dk*hkwk+tap)+g，B/C 视图解耦合法
            LoadDataWithStride<SrcT, L3D_NO_RESET>(l0b[C0<SrcT>() * cinAlign16 * tap], bl1T, load3d);
        }
    }

    // ---------- fmap GM → B1 单 batch 半区（dnNum=1，与 LoadA1Dy 完全同构） ----------
    // 每 batch 一条 Dn2Nz{dnNum=1, dValue=cinLength, nValue=dhwin}，B1 双半区
    // ping-pong + 预载，batch 维由半区基址承载——L1 门不含 batch 因子
    // （大 batch 场景可算）。半区段宽按本块 cinAlign16（尾块/主块一套逻辑），
    // 与 LoadL0Fmap 的 channelSize 同源。
    __aicore__ inline void LoadL1Fmap(const LocalTensor<SrcT>& bl1, const DLoadConfig& config,
                                      const CoutCinRange& cRange, uint32_t batchIdx)
    {
        const ShapeAttribute& shape = config.shape;
        const uint64_t dhwin = static_cast<uint64_t>(shape.din) * shape.hin * shape.win;
        // 本块 16 对齐 cin 段宽（B1 布局量；与 IterateK/LoadL0Fmap 的 cinAlign16 同式）
        const uint32_t cinAlign16 = (cRange.cinLength + 15) / 16 * 16;

        Dn2NzParams dn2nz;
        dn2nz.dnNum = 1;                                    // ★单 batch 搬运（B1 半区承载 batch 维）
        dn2nz.dValue = cRange.cinLength;                    // 真实段长（引擎 bL1cin1CopyLen）
        dn2nz.nValue = static_cast<uint16_t>(dhwin);        // din×hin×win 合轴
        dn2nz.srcDnMatrixStride = dhwin;                    // dnNum=1 不生效，语义占位
        dn2nz.srcDValue = dhwin;                            // cin 行距
        dn2nz.dstNzC0Stride = static_cast<uint16_t>(dhwin); // 组内行距（与 cin 段宽无关）
        dn2nz.dstNzNStride = 1;
        dn2nz.dstNzMatrixStride = static_cast<uint32_t>(dhwin * cinAlign16); // 半区段距（dnNum=1 不生效）
        // 源基址：batch 段 + 块内 cin 偏移（fmap GM NCDHW 每 cin 行 dhwin）
        const uint64_t offset = static_cast<uint64_t>(batchIdx) * shape.cin * dhwin +
                                static_cast<uint64_t>(cRange.cinIdx) * dhwin;
        DataCopy(bl1, fmap_[offset], dn2nz);
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
        fp.mSize = static_cast<uint16_t>(dhwk);                     // DN 列 = dhwk（整块一次）
        fp.nSize = static_cast<uint16_t>(cRange.cinLength);         // DN 行 = ci（真实值）
        fp.params.srcNzMatrixStride = 1;                            // C0 单位：co 步进 16 元素
        // ★L0C 块序 B = cin16G*tap + g（tap 外层，LoadL0Fmap 配对连续落位）。
        // 块序公式 B = (CS*tap + SS*g)/M → CS = M*cin16G（tap 步进 cin16G 块）、SS = M（g 步进 1 块）。
        // y 生产布局不变（DM/DS/nSize/mSize 不动）
        fp.params.srcNzC0Stride = static_cast<uint16_t>(alignedCout * ((cRange.cinLength + 15) / 16)); // tap 步进
        fp.srcStride = static_cast<uint16_t>(alignedCout); // cin16 组步进（C0 单位）
        fp.dstStride = dhwk;                               // 元素：DN 行长
        fp.params.dstDnMatrixStride = cinTotal * dhwk;     // 元素：相邻 co 的 DN 步进
        fp.quantPre = QuantMode_t::NoQuant;
        fp.unitFlag = 0;
        Fixpipe<SrcT, float, CFG_COLUMN_MAJOR>(y[yBase], l0c, fp);
    }

    // L0/L1 半区/池游标（翻转频率各不相同，须独立）：
    //   a0Pong_  每次 LoadL0Dy 翻转（(dout,win) 级，L0A 双 buf）
    //   bBuf_    每次 LoadL0Fmap 轮转 (bBuf_+1)%bBufCnt_（(win,dk) 级，L0B buf 池；
    //            bBufCnt_ Init 一次计算 tiling 上界口径，跨块恒定——轮转游标跨块
    //            语义为纯模轮转）
    //   l1pong_  每 k 段翻转（batch 级，B1+A1 双 L1 半区归一游标——fmap/dy 成对装载
    //            无单流路径，共用半区节奏与 flag id 0/1）
    // ★l1pong_/bBuf_/a0Pong_ 跨块持续（无重置）：相邻块 buf 序天然错开，使上一块对同
    //   buf/半区的末次释放 Set 恰被本块装载前 Wait 消费（跨块背压，见 IterateK 注释）
    bool l1pong_ = 0;
    bool a0Pong_ = 0;
    uint8_t bBufCnt_ = 0; // Init 一次计算：min(TOTAL_L0B_SIZE/bBufTileBytes_, 4)
    uint8_t bBuf_ = 0;
    uint32_t bBufTileBytes_ = 0; // Init 一次计算：tiling 上界口径 tile 步长（跨块恒定）
    // L0C ping-pong 状态（Init 门判定一次）：
    //   l0cPingpong_ 开门标志（上界单视图占用 ≤ TOTAL_L0C_SIZE/2，跨块恒定 → 两路径
    //   不混用）；l0cPong_ 块级翻转（跨块持续保证相邻块半区错开）；半区从 L0C 中间切
    //   （ping@0/pong@L0C/2）
    bool l0cPingpong_ = 0; // 开门标志：上界单视图占用 ≤ TOTAL_L0C_SIZE/2（跨块恒定）
    bool l0cPong_ = 0;     // 块级翻转（开门路径使用；跨块持续保证相邻块半区错开）
    GlobalTensor<SrcT> fmap_;
    GlobalTensor<SrcT> dy_;
};

} // namespace BpDLoad

#endif // CONV_BP_DLOAD_COMPUTE_H
