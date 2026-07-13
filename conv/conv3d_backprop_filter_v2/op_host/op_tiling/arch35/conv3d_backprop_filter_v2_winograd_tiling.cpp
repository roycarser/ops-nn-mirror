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
 * \file conv3d_backprop_filter_v2_winograd_tiling.cpp
 * \brief
 */

#ifndef CONV3D_BACKPROP_FILTER_V2_WINOGRAD_TILING_CPP
#define CONV3D_BACKPROP_FILTER_V2_WINOGRAD_TILING_CPP

#include "conv/conv3d_backprop_filter_v2/op_kernel/arch35/conv3d_backprop_filter_v2/conv3d_backprop_filter_v2_tiling_key.h"
#include "op_host/tiling_templates_registry.h"
#include "error_util.h"
#include "op_host/util/math_util.h"
#include "conv3d_backprop_filter_v2_winograd_tiling.h"

namespace Ops {
namespace NN {
namespace Conv {
namespace {
bool CheckDtype(const Conv3dBpFilterV2RunInfo& runInfo, const char_t* opName)
{
    //float16/bfloat16浮点误差比较严重，禁用，如果有int量化到时可以开下
    if (runInfo.a_dtype != ge::DataType::DT_FLOAT ||
        runInfo.b_dtype != ge::DataType::DT_FLOAT ||
        runInfo.c_dtype != ge::DataType::DT_FLOAT) {
        OP_LOGD(opName, "Winograd tiling only support float");
        return false;
    }
    return true;
}

bool CheckAttrs(const Conv3dBpFilterV2RunInfo& runInfo, const char_t* opName)
{
    if (runInfo.di != 1 || runInfo.dout != 1 || runInfo.kd != 1 ||
        runInfo.dilation_d != 1 || runInfo.stride_d != 1 ||
        runInfo.pad_f != 0 || runInfo.pad_b != 0) {
        OP_LOGD(opName, "Winograd tiling is only supported for 2d");
        return false;
    }

    if (runInfo.dilation_h != 1 || runInfo.dilation_w != 1 ||
        runInfo.stride_h != 1 || runInfo.stride_w != 1 ||
        runInfo.groups != 1 ||
        runInfo.pad_u != runInfo.pad_d ||
        runInfo.pad_l != runInfo.pad_r) {
        OP_LOGD(opName, "Winograd tiling is not support current attrs");
        return false;
    }

    if (runInfo.kh != 3 || runInfo.kw != 3) {
        OP_LOGD(opName, "Winograd tiling only support 3*3 kernel");
        return false;
    }

    if (runInfo.pad_u > Conv3DBackpropFilterV2WinogradTiling::RECOMMEND_PAD_LIMIT ||
        runInfo.pad_l > Conv3DBackpropFilterV2WinogradTiling::RECOMMEND_PAD_LIMIT) {
        OP_LOGD(opName, "pad is too large for winograd");
        return false;
    }
    return true;
}

bool CheckShape(const Conv3dBpFilterV2RunInfo& runInfo, const char_t* opName)
{
    uint64_t tileH = Ops::Base::CeilDiv(runInfo.ho, 2);
    uint64_t tileW = Ops::Base::CeilDiv(runInfo.wo, 2);
    if (tileH * tileW * runInfo.batch > Conv3DBackpropFilterV2WinogradTiling::RECOMMEND_K_MAX_SIZE) {
        OP_LOGD(opName, "current reduce asix is too large for Winograd impl");
        return false;
    }

    if ((runInfo.co / 64) * (runInfo.ci / 64) < 16) {
        OP_LOGD(opName, "the cout/cin is too small for winograd");
        return false;
    }
    return true;
}
}

bool Conv3DBackpropFilterV2WinogradTiling::IsCapable()
{
    if (!IsSocVersion91095()) {
        return false;
    }

    if (!CheckFormat()) {
        OP_LOGD(opName_, "current format is not support by winograd tiling");
        return false;
    }

    if (!CheckDtype(runInfo_, opName_)) {
        return false;
    }

    if (!CheckAttrs(runInfo_, opName_)) {
        return false;
    }

    if (!CheckShape(runInfo_, opName_)) {
        return false;
    }

    return true;
}

constexpr size_t Y_INDEX = 2;
constexpr size_t FILTER_INDEX = 0;
constexpr size_t OUTPUT_BP_INDEX = 0;

bool Conv3DBackpropFilterV2WinogradTiling::CheckFormat()
{
    const auto fmapDesc = context_->GetInputDesc(OUTPUT_BP_INDEX);
    OP_TILING_CHECK(
        fmapDesc == nullptr, CUBE_INNER_ERR_REPORT("Conv3DBackpropFilterV2", "fmap_desc is null"),
        return false);
    auto fmapFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(fmapDesc->GetStorageFormat()));
    const auto dedyDesc = context_->GetInputDesc(Y_INDEX);
    OP_TILING_CHECK(
        dedyDesc == nullptr, CUBE_INNER_ERR_REPORT("Conv3DBackpropFilterV2", "dedyDesc is null"),
        return false);
    auto dedyFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(dedyDesc->GetStorageFormat()));
    const auto filterDesc = context_->GetOutputDesc(FILTER_INDEX);
    OP_TILING_CHECK(
        filterDesc == nullptr, CUBE_INNER_ERR_REPORT("Conv3DBackpropFilterV2", "filterDesc is null"),
        return false);
    auto filter_format = static_cast<ge::Format>(ge::GetPrimaryFormat(filterDesc->GetStorageFormat()));

    return fmapFormat == ge::FORMAT_NCDHW &&
           dedyFormat == ge::FORMAT_NCDHW &&
           filter_format == ge::FORMAT_NCDHW;
}


uint64_t Conv3DBackpropFilterV2WinogradTiling::GetTilingKey() const
{
    uint8_t tilingFlag = 1;
     if (singleShapeTile_ == B16H8W8_B32H4W8) {
        tilingFlag = 1;
    } else if (singleShapeTile_ == B16H4W16_B32H2W16) {
        tilingFlag = 2;
    }

    constexpr uint8_t ResidentFmap = 0;
    constexpr uint8_t ResidentDY = 1;
    //fmap和dy选较小的驻留
    bool residentFlag = runInfo_.ci > runInfo_.co ? ResidentDY : ResidentFmap;
    const uint64_t tilingKey = GET_TPL_TILING_KEY(1, 0, 0, tilingFlag, residentFlag);
    OP_LOGD(context_->GetNodeName(), "tilingKey is: [%lu] , use winograd tiling flag [%lu]", tilingKey, tilingFlag);
    return tilingKey;
}


Conv3DBackpropFilterV2WinogradTiling::SingleShapeTile SelectTemplate(uint32_t tileH, uint32_t tileW, bool isFp32)
{

    //B16H8W8_B32H4W8
    uint32_t singleShapeTileH1 = isFp32 ? 4 : 8;
    uint32_t singleShapeTileW1 = 8;

    //B16H4W16_B32H2W16
    uint32_t singleShapeTileH2 = isFp32 ? 2 : 4;
    uint32_t singleShapeTileW2 = 16;

    uint32_t clusters1 = Ops::Base::CeilDiv(tileH, singleShapeTileH1) *
                         Ops::Base::CeilDiv(tileW, singleShapeTileW1);

    uint32_t clusters2 = Ops::Base::CeilDiv(tileH, singleShapeTileH2) *
                         Ops::Base::CeilDiv(tileW, singleShapeTileW2);

    //谁的空转块更少选谁,计算块一样多时，当前选择H4W16,内轴更大，可能会有一些优势
    if (clusters1 < clusters2) {
        return Conv3DBackpropFilterV2WinogradTiling::B16H8W8_B32H4W8;
    }
    return Conv3DBackpropFilterV2WinogradTiling::B16H4W16_B32H2W16;
}


ge::graphStatus Conv3DBackpropFilterV2WinogradTiling::DoOpTiling()
{
    uint32_t tileH = Ops::Base::CeilDiv(runInfo_.ho, 2);
    uint32_t tileW = Ops::Base::CeilDiv(runInfo_.wo, 2);

    singleShapeTile_ = SelectTemplate(tileH, tileW, runInfo_.a_dtype_bytes == 4);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv3DBackpropFilterV2WinogradTiling::GetWorkspaceSize()
{
    constexpr uint64_t WORKSPACE = 16777216; // 16777216 : 16 * 1024 * 1024 libapiworkspace
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);

    uint32_t singleShapeTileH;
    uint32_t singleShapeTileW;
     if (singleShapeTile_ == B16H8W8_B32H4W8) {
        singleShapeTileH = runInfo_.a_dtype_bytes == 2 ? 8 : 4;
        singleShapeTileW = 8;
    } else if (singleShapeTile_ == B16H4W16_B32H2W16) {
        singleShapeTileH = runInfo_.a_dtype_bytes == 2 ? 4 : 2;
        singleShapeTileW = 16;
    } else {
        return ge::GRAPH_FAILED;
    }

    uint32_t tileH = Ops::Base::CeilDiv(runInfo_.ho, 2);
    uint32_t tileW = Ops::Base::CeilDiv(runInfo_.wo, 2);

    uint32_t k1 = Ops::Base::CeilDiv(tileH, singleShapeTileH) *
                  Ops::Base::CeilDiv(tileW, singleShapeTileW);

    uint32_t k0 = singleShapeTileH * singleShapeTileW * 16;
    uint32_t c0Byte = 32;
    uint32_t c1c0Fmap = Ops::Base::CeilAlign(static_cast<uint32_t>(runInfo_.ci * runInfo_.a_dtype_bytes), c0Byte);
    uint32_t c1c0Dy = Ops::Base::CeilAlign(static_cast<uint32_t>(runInfo_.co * runInfo_.b_dtype_bytes), c0Byte);

    //全局驻留的空间
    size_t userWorkSpaceSize = static_cast<size_t>(runInfo_.batch) * std::max(c1c0Fmap, c1c0Dy) * k0 * k1;

    //nc1hwc0转换的空间
    userWorkSpaceSize += static_cast<size_t>(runInfo_.batch) * c1c0Fmap * runInfo_.hi * runInfo_.wi;
    userWorkSpaceSize += static_cast<size_t>(runInfo_.batch) * c1c0Dy * runInfo_.ho * runInfo_.wo;

    //切k的空间
    userWorkSpaceSize += 64 * 64 * 3 * 3 * sizeof(float) * platformInfo_.core_num;

    workspaces[0] = WORKSPACE + userWorkSpaceSize;
    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("Conv3DBackpropFilterV2", Conv3DBackpropFilterV2WinogradTiling, 2);
}
}
}
#endif //CONV3D_BACKPROP_FILTER_V2_WINOGRAD_TILING_CPP