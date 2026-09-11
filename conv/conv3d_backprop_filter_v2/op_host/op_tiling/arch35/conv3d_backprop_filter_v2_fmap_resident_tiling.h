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
 * \file conv3d_backprop_filter_v2_fmap_resident_tiling.h
 * \brief fmap_resident 场景 host tiling 模板：3×3×3/s1/d1/p1(dhw)/fp32+hf32/NCDHW/groups=1，
 *        fmap（B 矩阵）L1 全载驻留 + K 全载单块累加 + L0C 直出（design: fmap_resident_design.md）
 */
#ifndef CONV3D_BACKPROP_FILTER_V2_FMAP_RESIDENT_TILING_H
#define CONV3D_BACKPROP_FILTER_V2_FMAP_RESIDENT_TILING_H

#include "conv3d_backprop_filter_v2_basic_block_tiling_arch35.h"

namespace Ops {
namespace NN {
namespace Conv {
class Conv3dBpFilterV2FmapResidentTiling : public Conv3DDWV2BasicBlockTilingArch35 {
public:
    // 场景常量（与 kernel 侧 fmap_resident_common.h 同值——编译期两侧各自定义，改动须双侧同步，契约 C4）
    static constexpr uint32_t RES_CIN = 32;      // fmap 驻留切片 cin 宽（n0 对齐，无尾块前提）
    static constexpr uint32_t RES_M = 64;        // tile M 粒度（方案 0：32→64，轮数减半；kernel FR_RES_M 同值）
    static constexpr uint32_t BASE_K = 16;       // L0A/L0B K 步进（DoOpTiling baseK 与 CalcMLoad 容量门共用）
    static constexpr uint32_t L0A_HALF_K = 224;  // L0A 半区 K 行预算上界（32×224×4B=28KB ≤ 32KB）
    static constexpr uint64_t L1_AVAIL_BYTES = 512 * 1024 - 128; // 512KB − 32B 对齐余量×4

    explicit Conv3dBpFilterV2FmapResidentTiling(gert::TilingContext* context)
        : Conv3DDWV2BasicBlockTilingArch35(context)
    {
        Reset();
    }

    ~Conv3dBpFilterV2FmapResidentTiling() override = default;

protected:
    bool IsCapable() override;

    ge::graphStatus DoOpTiling() override;

    // MIX kernel ABI 必需：workspace 参数槽（入口 SetSysWorkspace/GetUserWorkspace 消费）。
    // 缺申报 → ttk/运行时 workspace 参数为空 → 启动参数少一槽 → kernel 的 workspace/tiling
    // 指针整体错位（tiling 指针槽读到 tiling 内容前 8 字节）→ 标量 GM 访问越界（AIC 264）
    ge::graphStatus GetWorkspaceSize() override;

    // 场景模板为 tiling 唯一权威（design §4.5：不继承 streamK 链）——base DoLibApiTiling 会以
    // blockTiling_（本场景未初始化）覆写 DoOpTiling 产物（streamkType/baseK/stepKa 等），必须拦截
    ge::graphStatus DoLibApiTiling() override { return ge::GRAPH_SUCCESS; }

    uint64_t GetTilingKey() const override;

private:
    bool CheckFmapResidentAttrs();
    bool CheckCapacity(); // 容量门：驻留 2-slice + A1 双缓冲 ≤ L1；不过则降级 streamK（IsCapable false）

    uint32_t CalcBatchExtent();
    uint32_t CalcMLoad(uint32_t batchExtent, uint32_t spanMax);
    uint32_t CalcSpanMax(); // chunk 公式 ⌊T·c/32⌋ 精确逐核算最大组跨度（与 kernel Scheduler 同式，契约 新5）
};
} // namespace Conv
} // namespace NN
} // namespace Ops
#endif // CONV3D_BACKPROP_FILTER_V2_FMAP_RESIDENT_TILING_H
