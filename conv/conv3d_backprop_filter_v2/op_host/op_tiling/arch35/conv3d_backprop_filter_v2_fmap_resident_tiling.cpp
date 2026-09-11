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
 * \file conv3d_backprop_filter_v2_fmap_resident_tiling.cpp
 * \brief fmap_resident 场景 host tiling：七门准入 + 容量门 + 场景参数推导；priority=-1 直接 Register（P-3：
 *        REGISTER_TILING_TEMPLATE 宏族对负 priority token-paste 编译死锁，唯一可行路径为直接实例化）
 */
#include "conv3d_backprop_filter_v2_fmap_resident_tiling.h"
#include "op_host/tiling_templates_registry.h"
#include "conv/conv3d_backprop_filter_v2/op_kernel/arch35/conv3d_backprop_filter_v2/conv3d_backprop_filter_v2_tiling_key.h"
#include <string>

namespace Ops {
namespace NN {
namespace Conv {
namespace {
constexpr uint32_t KERNEL_SIZE_3 = 3;
constexpr uint32_t PAD_1 = 1;
constexpr uint32_t FP32_BYTES = 4;
} // namespace

bool Conv3dBpFilterV2FmapResidentTiling::CheckFmapResidentAttrs()
{
    // attrs 门：kd=kh=kw=3 ∧ stride/dilation dhw 全 1 ∧ pad 六值全 1 ∧ groups=1（dk=3 与 winograd 的 kd==1
    // 门构成第二重互斥；hf32 门为第一重完备互斥，见 HR-6 专项）
    if (runInfo_.kd != KERNEL_SIZE_3 || runInfo_.kh != KERNEL_SIZE_3 || runInfo_.kw != KERNEL_SIZE_3) {
        OP_LOGD(opName_, "fmap_resident only support 3*3*3 kernel");
        return false;
    }
    if (runInfo_.stride_d != 1 || runInfo_.stride_h != 1 || runInfo_.stride_w != 1 ||
        runInfo_.dilation_d != 1 || runInfo_.dilation_h != 1 || runInfo_.dilation_w != 1) {
        OP_LOGD(opName_, "fmap_resident only support stride/dilation 1");
        return false;
    }
    if (runInfo_.pad_f != PAD_1 || runInfo_.pad_b != PAD_1 || runInfo_.pad_u != PAD_1 ||
        runInfo_.pad_d != PAD_1 || runInfo_.pad_l != PAD_1 || runInfo_.pad_r != PAD_1) {
        OP_LOGD(opName_, "fmap_resident only support pad 1");
        return false;
    }
    if (runInfo_.groups != 1) {
        OP_LOGD(opName_, "fmap_resident only support groups 1");
        return false;
    }
    return true;
}

uint32_t Conv3dBpFilterV2FmapResidentTiling::CalcBatchExtent()
{
    // batchExtent = max{b ≤ batch : b×dout×ho×wo ≤ L0A_HALF_K}（dedy 载入块 K 行预算，契约 新6）
    uint64_t perBatchK = static_cast<uint64_t>(runInfo_.dout) * runInfo_.ho * runInfo_.wo;
    uint32_t extent = 0;
    while (extent < static_cast<uint32_t>(runInfo_.batch) &&
           static_cast<uint64_t>(extent + 1) * perBatchK <= L0A_HALF_K) {
        ++extent;
    }
    return extent == 0 ? 1 : extent; // perBatchK > 224 时仍取 1（kIter 尾块纪律兜底，L0 装载按 k 步进）
}

uint32_t Conv3dBpFilterV2FmapResidentTiling::CalcSpanMax()
{
    // chunk 走位公式 ⌊T·c/usedCore⌋ 精确逐核算最大组跨度（与 kernel Scheduler::DecodeChunk 同式，契约 新5；
    // mCnt=co/RES_M 方案 0 M 粒度 64）
    uint64_t mCnt = static_cast<uint64_t>(runInfo_.co) / RES_M;
    uint64_t nCnt = static_cast<uint64_t>(runInfo_.ci) / RES_CIN;
    uint64_t total = mCnt * nCnt;
    const uint32_t coreNum = platformInfo_.core_num < total ? platformInfo_.core_num : static_cast<uint32_t>(total);
    uint32_t spanMax = 1;
    for (uint32_t c = 0; c < coreNum; ++c) {
        uint64_t tileStart = total * c / coreNum;
        uint64_t tileEnd = total * (c + 1) / coreNum;
        if (tileStart >= tileEnd) {
            continue;
        }
        uint64_t gFirst = tileStart / mCnt;
        uint64_t gLast = (tileEnd - 1) / mCnt;
        uint32_t span = static_cast<uint32_t>(gLast - gFirst + 1);
        spanMax = span > spanMax ? span : spanMax;
    }
    return spanMax;
}

uint32_t Conv3dBpFilterV2FmapResidentTiling::CalcMLoad(uint32_t batchExtent, uint32_t spanMax)
{
    // 容量门预验 mLoad=RES_M=64（方案 0：tile M 粒度固定 64，与 kernel FR_RES_M 同值 C4）：
    // A1 槽位足迹 = al1Bound = RES_M×stepKa×baseK（stepKa=⌈ho×wo/baseK⌉，与 DoOpTiling 同式；
    // 与 kernel FrInitTque a1Bytes 同式两份，C4），预算 = (L1 − spanMax×驻留组切片)/2（A1 ping/pong 各半）。
    // 锚点复现：case1 [8,384,6,6,6] → 预算 40KB/槽，A1 槽 12KB 充足 → 64；case2 [8,768,3,3,3] → 64。
    // 预算不足 → 0 = 容量门失败，IsCapable 拒绝降级 streamK（PERF_PLAN 方案 0 改动点 1）
    (void)batchExtent; // A1 窗口逐 batchDout 流式（batch 维不进 A1 容量）
    uint64_t sliceBytes = static_cast<uint64_t>(runInfo_.batch) * RES_CIN * runInfo_.di * runInfo_.hi *
                          runInfo_.wi * FP32_BYTES;
    uint64_t kIter = (static_cast<uint64_t>(runInfo_.ho) * runInfo_.wo + BASE_K - 1) / BASE_K;
    uint64_t a1SlotBytes = static_cast<uint64_t>(RES_M) * kIter * BASE_K * FP32_BYTES;
    uint64_t budget = L1_AVAIL_BYTES > spanMax * sliceBytes ?
                      (L1_AVAIL_BYTES - spanMax * sliceBytes) / 2 : 0;
    return a1SlotBytes <= budget ? RES_M : 0; // 0 = 容量不足
}

bool Conv3dBpFilterV2FmapResidentTiling::CheckCapacity()
{
    uint32_t batchExtent = CalcBatchExtent();
    if (batchExtent == 0) {
        return false;
    }
    return CalcMLoad(batchExtent, CalcSpanMax()) != 0;
}

bool Conv3dBpFilterV2FmapResidentTiling::IsCapable()
{
    // 门 1：soc（winograd_tiling.cpp:158 同款；arch22 模板被同门拒绝，互不干扰）
    if (!IsSocVersion91095()) {
        return false;
    }
    // 门 2：格式（fmap/dedy/y 全 NCDHW）
    if (!CheckFormat()) {
        OP_LOGD(opName_, "fmap_resident only support NCDHW");
        return false;
    }
    // 门 3：dtype（a/b 全 fp32；c 恒 fp32）
    if (runInfo_.a_dtype != ge::DataType::DT_FLOAT || runInfo_.b_dtype != ge::DataType::DT_FLOAT) {
        OP_LOGD(opName_, "fmap_resident only support fp32");
        return false;
    }
    // 门 4：hf32（fmap_resident 域定义 fp32+hf32；与 winograd 的互斥由 3D 判定承担）
    if (runInfo_.hf32Flag != 1) {
        OP_LOGD(opName_, "fmap_resident requires enable_hf32");
        return false;
    }
    // 门 5-6：attrs + groups
    if (!CheckFmapResidentAttrs()) {
        return false;
    }
    // 门 7：通道整除（结构性全 tile：无尾块 → 全覆写 kNeedInitOutput=false 与 nSize 真实值共同前提）。
    // ci 32 整除（驻留切片宽）+ co 64 整除（方案 0：tile M 粒度 RES_M=64，mCnt=co/64 精确分块；
    // co 仅 32 整除的形状不再准入 → 升序遍历落 streamK 优雅降级）
    if (runInfo_.ci % RES_CIN != 0 || runInfo_.co % RES_M != 0) {
        OP_LOGD(opName_, "fmap_resident requires cin divisible by 32 and cout divisible by 64");
        return false;
    }
    // 门 8：容量门（不过 → return false → 升序遍历落 streamK(3) 优雅降级）
    if (!CheckCapacity()) {
        OP_LOGD(opName_, "fmap_resident capacity gate failed, fallback to streamK");
        return false;
    }
    return true;
}

ge::graphStatus Conv3dBpFilterV2FmapResidentTiling::DoOpTiling()
{
    // runInfo_/shape 链继承基类 GetShapeAttrsInfo；SetShapeTiling/SetAttrTiling 复用（shape/attr 全量）
    SetShapeTiling(tilingData_.dwTiling);
    SetAttrTiling(tilingData_.dwTiling);

    auto& dwt = tilingData_.dwTiling;
    const uint32_t batchExtent = CalcBatchExtent();
    const uint32_t spanMax = CalcSpanMax();
    const uint32_t mLoad = CalcMLoad(batchExtent, spanMax);
    if (mLoad == 0) {
        return ge::GRAPH_FAILED;
    }
    // 场景参数（design §4.5 字段清单；D-2：streamkType 结构体默认 1，此处显式置 0 禁遗漏，UT 断言兜底）；
    // M 侧 baseM/singleCoreM/singleCoreCout=RES_M=64（方案 0：Mmad [64,K,288]，L0A 4KB/L0C 2×73.7KB 均满足）
    dwt.m0 = 16;
    dwt.n0 = 16;
    dwt.k0 = 8; // fp32 C0=8 特参
    dwt.baseM = RES_M;
    dwt.baseN = RES_CIN * runInfo_.kh * runInfo_.kw; // 288 = 32 cin × 9 hkwk
    dwt.baseK = BASE_K;
    const uint64_t kIter = (static_cast<uint64_t>(runInfo_.ho) * runInfo_.wo + dwt.baseK - 1) / dwt.baseK;
    dwt.singleCoreM = RES_M;
    dwt.singleCoreN = dwt.baseN;
    dwt.singleCoreCin = RES_CIN;
    dwt.singleCoreCout = RES_M;
    dwt.singleCoreHo = runInfo_.ho;
    dwt.singleCoreK = static_cast<uint64_t>(runInfo_.ho) * runInfo_.wo;
    dwt.singleCoreBatch = static_cast<uint64_t>(batchExtent) * runInfo_.dout;
    dwt.singleCoreBatchDout = static_cast<uint64_t>(runInfo_.batch) * runInfo_.dout;
    dwt.singleCoreGroup = 1;
    dwt.usedCoreNum = platformInfo_.core_num;
    dwt.stepKa = static_cast<uint32_t>(kIter);
    dwt.stepKb = static_cast<uint32_t>(kIter);
    dwt.iterateOrder = 0;
    dwt.streamkType = 0; // HR-2 证据 1：显式置 0（默认 1，禁遗漏）
    dwt.splitWo = runInfo_.wo; // isSplitWo_ = false
    dwt.channelSize = 8;
    dwt.hf32Flag = runInfo_.hf32Flag;
    dwt.al0Pbuffer = 2;
    dwt.bl0Pbuffer = 2;
    dwt.cl0Pbuffer = 2;
    dwt.al1Pbuffer = 2; // A1 流式乒乓（design §4.2/§4.5；引擎 LoadToA1 按 kaStep 交替 ping/pong）
    dwt.bl1Pbuffer = 1; // B1 驻留单区（租约语义，非 pingpong）
    dwt.al1Bound = static_cast<uint64_t>(RES_M) * dwt.stepKa * dwt.baseK; // A1 窗口 [64, kal1]（方案 0 M=64）
    dwt.bl1Bound = static_cast<uint64_t>(runInfo_.batch) * runInfo_.di * runInfo_.hi * runInfo_.wi *
                   RES_CIN; // 驻留组切片（参考值）

    tilingData_.fmapResidentTiling.mLoad = mLoad;
    tilingData_.fmapResidentTiling.batchExtent = batchExtent;

    OP_LOGD(opName_,
            "fmap_resident tiling: mLoad=%u batchExtent=%u spanMax=%u stepKa=%u key=2 mCnt=%lu nCnt=%lu core=%u",
            mLoad, batchExtent, spanMax, dwt.stepKa,
            static_cast<uint64_t>(runInfo_.co) / RES_M, static_cast<uint64_t>(runInfo_.ci) / RES_CIN,
            platformInfo_.core_num);
    return ge::GRAPH_SUCCESS;
}

// MIX kernel ABI 必需：申报 workspace 参数槽（kernel 入口 SetSysWorkspace/GetUserWorkspace 消费；
// 本场景不使用 user workspace，仅满足参数槽存在性——缺失时 ttk/运行时启动参数少一槽，kernel 的
// workspace/tiling 指针整体错位，tiling 指针槽读到 tiling 内容前 8 字节 → 标量 GM 越界 AIC 264）
ge::graphStatus Conv3dBpFilterV2FmapResidentTiling::GetWorkspaceSize()
{
    constexpr uint64_t WORKSPACE = 16777216; // 16777216 : 16 * 1024 * 1024（对齐 streamK/winograd 申报口径）
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = WORKSPACE;
    return ge::GRAPH_SUCCESS;
}

uint64_t Conv3dBpFilterV2FmapResidentTiling::GetTilingKey() const
{
    // DECL 尾部 append：TPL_FMAP_RESIDENT=3 在 [TPL_STREAM_K, TPL_MN_STREAM_K, TPL_FMAP_RESIDENT] 中索引 2
    // → key=2（1.2 §1.2 定案；resident 位不复用——winograd workspace 驻留语义）
    const uint64_t tilingKey = GET_TPL_TILING_KEY(TPL_FMAP_RESIDENT, 0, 0, TPL_WINOGRAD_DISABLE, 0);
    OP_LOGD(context_->GetNodeName(), "fmap_resident tilingKey is: [%lu]", tilingKey);
    return tilingKey;
}

// W3：priority=-1 直接 Register 实例化（REGISTER_TILING_TEMPLATE 宏 token-paste 死锁，P-3；1.2 §2.3 探针双证
// 模板）。marker 手工补齐：L0 静态打包按 op_impl_register_template_ 前缀扫描符号，防链接裁剪。
[[maybe_unused]] std::string op_impl_register_template_Conv3dBpFilterV2FmapResidentTiling_neg1_register =
    std::string("op_impl_register_template_") + "Conv3DBackpropFilterV2";
[[maybe_unused]] static Ops::NN::Optiling::Register fmap_resident_tiling_register =
    Ops::NN::Optiling::Register("Conv3DBackpropFilterV2").tiling<Conv3dBpFilterV2FmapResidentTiling>(-1);
} // namespace Conv
} // namespace NN
} // namespace Ops
