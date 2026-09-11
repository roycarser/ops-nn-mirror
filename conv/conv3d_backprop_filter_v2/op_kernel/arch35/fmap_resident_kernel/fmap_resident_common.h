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
 * \file fmap_resident_common.h
 * \brief SceneDesc + 场景引擎实例化（Cfg 三钩子拦截 + REGISTER_DW_IMPL 场景别名）。
 *        引擎与 impl 源码零改动：Init/Compute/IterateAll 经 DECLARE_IMPL Config 扩展点替换
 *        （conv_bp_util_arch35.h 官方扩展点）；UpdateMNIdx/GetTensorC/End 走引擎默认（本场景
 *        不消费 UpdateMNIdx 的 m/n 游标机——tile 循环由 IterateAll 驱动；GetTensorC 默认
 *        LoadL0c2Gm 的 dk 折叠分支与本场景 N=cin×hkwk×dk 组织天然一致）。
 */
#ifndef FMAP_RESIDENT_COMMON_H
#define FMAP_RESIDENT_COMMON_H

#include "../conv3d_backprop/conv_bp_func_arch35.h"
#include "../conv3d_backprop/conv_bp_util_arch35.h"
#include "../conv3d_backprop_filter_v2/conv3d_bp_filter_config.h"
#include "../conv3d_backprop_filter_v2/conv3d_bp_filter_impl.h"
#include "../conv3d_backprop_filter_v2/conv3d_bp_filter_intf.h"
#include "../conv3d_backprop_filter_v2/conv3d_backprop_filter_v2_tiling_data.h"

namespace ConvolutionBackprop {
namespace FmapResident {
// 三钩子实现体前向声明（fmap_resident_processor.h 定义；Cfg 成员模板体惰性实例化依赖）
template <class U>
__aicore__ inline void FrInitTque(U* self);
template <class U>
__aicore__ inline void FrCompute(U* self, ConvolutionBackpropFunc::Out2L1ScalarParams& out2L1Params);
template <class U>
__aicore__ inline void FrIterateAll(U* self, const GlobalTensor<typename U::DstT>& output, uint8_t enAtomic);
} // namespace FmapResident

// ---------------- SceneDesc（design §2.1） ----------------
enum class FrVecRole : uint8_t { None };
struct FrSceneDesc {
    static constexpr bool kNeedInitOutput = false; // 全覆写：T×[64, 32cin×dkhkwk] 精确覆盖 y（co 64 整除门）
    static constexpr FrVecRole kVecRole = FrVecRole::None; // 零跨核 flag；AIV 早退
    static constexpr bool kUseExistingEngine = true;       // Conv3DBackpropFilter 引擎 + 扩展点替换
};

// 场景常量（与 host 侧 Conv3dBpFilterV2FmapResidentTiling 同值——编译期两侧各自定义，改动须双侧同步，C4）
constexpr uint32_t FR_RES_CIN = 32;
constexpr uint32_t FR_RES_M = 64; // tile M 粒度（方案 0：32→64，轮数减半；host RES_M 同值）

// ---------------- 场景上下文（经 Cfg::ContextData 扩展，引擎 ctx 零改动） ----------------
// SrcT 模板化：场景被 W2 SEL 注册后所有 dtype 变体（fp32 场景 + fp16/bf16 守护实例化）都会
// 编译本 ContextData——张量类型须跟 Intf::SrcT（守护用例编译失败教训，2026-09-10）
template <typename SrcT>
struct FrBlockCtx {
    // chunk 走位状态（Scheduler 同式 ⌊T·c/usedCore⌋，契约 新5）
    uint64_t tileStart = 0;
    uint64_t tileEnd = 0;
    uint64_t mCnt = 0;
    uint64_t nCnt = 0;
    // 驻留租约状态（NEW ①；chunk 内组号单调递增 → 至多一次组切换，双槽位容量门保证）
    int32_t curGroup = -1;       // 已装载组号（-1 = 未装载）
    int32_t curGroupSlot = -1;   // 当前组所在槽位 ∈ {0,1}
    uint64_t groupSlice = 0;     // 单组驻留切片元素数 = batch×di×hwI×32
    uint64_t planeElems = 0;     // 单平面元素数 = hwI×32
    LocalTensor<SrcT> residentBuf; // 驻留区视图（AllocTensor 持久租约，切片寻址）
    GlobalTensor<SrcT> dedyBase;    // dedy 未平移基线（真根因#3：SetOutBackprop 会覆写
                                    // ctx.outBackPropGlobal_，逐 tile 平移必须在基线上做——否则累计漂移
                                    // → A 侧 GM 越界，2026-09-10 mock trace 实锤）
    TQue<TPosition::B1, 1> residentQue_; // 驻留区载体（设计偏差声明：原 raw LocalTensor[dx SmallKernel
                                         // 先例]在 CPU mock 下模型内存不合法——load3d padUp 窗口读
                                         // 注册区外 SIGSEGV；TQue 显式地址设备语义等价[引擎 a1Ping_
                                         // 同款]，AllocTensor 单次持久租约不 EnQue/不 Free）
};

template <class A, class B, class C, class D, const Conv3ddwConfig& CONV3DDW_CONFIG = CONV3DDW_CFG_DEFAULT>
struct Conv3dBpFmapResidentCfg : public Conv3DBpFilterCfg<A, B, C, D, CONV3DDW_CONFIG> {
    using Base = Conv3DBpFilterCfg<A, B, C, D, CONV3DDW_CONFIG>;
    __aicore__ inline Conv3dBpFmapResidentCfg() {}

    using ContextData = struct _ : public Base::ContextData {
        __aicore__ inline _() {}
        DEFINE_STUCT_FIELD(FrBlockCtx<typename C::Type>, frCtx); // 驻留区 TQue 载体（C=dedy ConvType，SrcT 跟随实例化 dtype）
    };

    // ---- 钩子 1：Init（替换引擎默认：CheckTiling/InitParams 复用 + 场景 InitTque 三区布局 + hf32） ----
    template <class U>
    struct Init {
        static __aicore__ inline void call(U* self, const AscendC::conv_bp_v2_kernel::TConv3DDwTiling* __restrict tiling)
        {
            self->ctx.tiling_ = tiling;
            ConvolutionBackpropFunc::CheckTiling<U>(self);
            ConvolutionBackpropFunc::InitParams<U>(self);
            FmapResident::FrInitTque<U>(self);
            if (self->ctx.tiling_->hf32Flag) {
                SetHF32Mode(true);
            }
        }
    };

    // ---- 钩子 2：Compute（替换 ComputeNormal：batchDout 循环 + 驻留平面选择 + 引擎 ComputeLoop） ----
    template <class U, bool sync>
    struct Compute {
        static __aicore__ inline void call(U* self, ConvolutionBackpropFunc::Out2L1ScalarParams& out2L1Params)
        {
            FmapResident::FrCompute<U>(self, out2L1Params);
        }
    };

    // ---- 钩子 3：IterateAll（替换 while-Iterate：tile-chunk 循环 + 驻留租约 + 逐 dk 段 Compute/直出） ----
    template <class U, bool sync>
    struct IterateAll {
        static __aicore__ inline void call(U* self, const GlobalTensor<typename U::DstT>& output, uint8_t enAtomic)
        {
            FmapResident::FrIterateAll<U>(self, output, enAtomic);
        }
    };
};

// 场景引擎别名（W3：引擎经 Config 扩展点拦截，Impl 复用通用 Conv3DBpFilterImpl——ContextData 扩展在 Cfg）
REGISTER_DW_IMPL(Conv3dBpFmapResidentEngine, Conv3dBpFmapResidentCfg, Conv3DBpFilterImpl, Conv3DBpFilterIntf);

} // namespace ConvolutionBackprop

#endif // FMAP_RESIDENT_COMMON_H
