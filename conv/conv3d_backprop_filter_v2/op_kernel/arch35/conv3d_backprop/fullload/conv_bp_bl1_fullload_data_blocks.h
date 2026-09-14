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
 * \file conv_bp_bl1_fullload_data_blocks.h
 * \brief BL1 全载（fmap 以 32 cin 切片整片驻留 L1）基本块走位迭代器
 */

#ifndef CONV_BP_BL1_FULLLOAD_DATA_BLOCKS_H
#define CONV_BP_BL1_FULLLOAD_DATA_BLOCKS_H

#include <cstdint>
#include <type_traits>
#include "../util/conv_bp_util.h" // 先含 kernel 链（mock 下定义 __aicore__），math_util 依赖
#include "op_kernel/math_util.h"
#include "utils/std/algorithm.h"

namespace BpFullLoad {

// BL1 全载基本块走位迭代器（fmap 驻留 L1，dedy 流式）
//
// 核心需求：每核迭代基本块时尽量维持自己的 cin 段不变，使 B1 驻留的 fmap 矩阵不被反复更新：
//   1. 基本块网格：coutCnt = CeilDiv(cout, singleShapeCout)，cinCnt = CeilDiv(cin, singleShapeCin)，
//      基本块总数 T = coutCnt × cinCnt；
//   2. 全局块序取 cin-major：t = cinBlockIdx × coutCnt + coutBlockIdx，同一 cin 组的块连续；
//   3. 每核认领连续段 [coreId·B, min(T, (coreId+1)·B))，B = ⌈T/coreNum⌉——均匀步长精确分割 [0, T)，
//      每块恰被一核处理（注意：与 fmap_resident_scheduler.h DecodeChunk 的比例式 ⌊T·c/N⌋ 不同式，
//      host 侧容量门/跨度推导须按本均匀步长式同步，勿照抄）；coreId/coreNum 取自
//      AicCoreId()/GetBlockNum()，不由上层传入；
//   4. 核内按 t 递增迭代 → cin 单调不降，仅在 cin 组边界切换 → B1 驻留切片只在组切换时装载。
// 越界纪律：GetLocalBlock 不校验迭代越界与空段，上层以 More() 判界后调用。
class BL1FullLoadBlockIterator {
public:
    // blockNum：块总数来源。0 = 取 GetBlockNum()（生产路径）；>0 = 调用方显式指定
    // （kernel 直调/UT 场景 <<<blockDim>>> 上下文可能未按 aclnn 语义填充该寄存器，显式传入更稳）
    inline __aicore__ BL1FullLoadBlockIterator(uint32_t cout, uint32_t cin, uint32_t singleShapeCout,
                                                uint32_t singleShapeCin, uint32_t blockNum = 0)
        : cout_(cout),
          cin_(cin),
          singleShapeCout_(singleShapeCout),
          singleShapeCin_(singleShapeCin),
          coutCnt_(Ops::Base::CeilDiv(cout, singleShapeCout)),
          cinCnt_(Ops::Base::CeilDiv(cin, singleShapeCin)),
          total_(coutCnt_ * cinCnt_),
          blocksPerCore_(Ops::Base::CeilDiv(
              total_, blockNum != 0 ? blockNum : static_cast<uint32_t>(AscendC::GetBlockNum()))),
          basicBlockIdx_(BpUtils::AicCoreId() * blocksPerCore_)
    {}

    inline __aicore__ bool More() const
    {
        return basicBlockIdx_ < total_ && basicBlockIdx_ < (BpUtils::AicCoreId() + 1) * blocksPerCore_;
    }

    // 获取当前aic计算的基本块范围；不校验迭代越界，由上层通过 More() 判断
    inline __aicore__ void GetLocalBlock(BpUtils::CoutCinRange& cRange) const
    {
        // 全局块序（cin-major）：t = cinBlockIdx × coutCnt + coutBlockIdx，同 cin 组的块连续
        const uint32_t cinBlockIdx = basicBlockIdx_ / coutCnt_;
        const uint32_t coutBlockIdx = basicBlockIdx_ - cinBlockIdx * coutCnt_;
        cRange.coutIdx = coutBlockIdx * singleShapeCout_;
        cRange.cinIdx = cinBlockIdx * singleShapeCin_;
        cRange.coutLength = AscendC::Std::min(singleShapeCout_, cout_ - cRange.coutIdx);
        cRange.cinLength = AscendC::Std::min(singleShapeCin_, cin_ - cRange.cinIdx);
    }

    inline __aicore__ void Next() { basicBlockIdx_++; }

private:
    const uint32_t cout_;
    const uint32_t cin_;
    const uint32_t singleShapeCout_;
    const uint32_t singleShapeCin_;
    const uint32_t coutCnt_;
    const uint32_t cinCnt_;
    const uint32_t total_;
    const uint32_t blocksPerCore_;
    uint32_t basicBlockIdx_;
};

} // namespace BpFullLoad

#endif // CONV_BP_BL1_FULLLOAD_DATA_BLOCKS_H
