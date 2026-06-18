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

bool Conv3DBackpropFilterV2WinogradTiling::IsCapable()
{
    if (!IsSocVersion91095()) {
        return false;
    }

    if (dtypeByte_ != ge::GetSizeByDataType(ge::DT_BF16) && dtypeByte_ != ge::GetSizeByDataType(ge::DT_FLOAT16) &&
        dtypeByte_ != ge::GetSizeByDataType(ge::DT_FLOAT)) {
        OP_LOGD(opName_, "Winograd tiling is only supported for bf16/fp16/fp32/hf32 dataType.");
        return false;
        }

    if (!CheckFormat()) {
        OP_LOGD(opName_, "current format is not support by winograd tiling");
        return false;
    }

    if (runInfo_.a_dtype != runInfo_.b_dtype) {
        //TODO 适配通路里ctype为假的fp32
        OP_LOGD(opName_, "Winograd tiling is not support different dtype");
        return false;
    }

    if (runInfo_.di != 1 ||
        runInfo_.dout != 1 ||
        runInfo_.kd != 1 ||
        runInfo_.dilation_d != 1 ||
        runInfo_.stride_d != 1 ||
        runInfo_.pad_f != 0 ||
        runInfo_.pad_b != 0) {
        OP_LOGD(opName_, "Winograd tiling is only supported for 2d");
        return false;
    }

    if (runInfo_.dilation_h != 1 ||
        runInfo_.dilation_w != 1 ||
        runInfo_.stride_h != 1 ||
        runInfo_.stride_w != 1 ||
        runInfo_.groups != 1 ||
        runInfo_.pad_u != runInfo_.pad_d ||
        runInfo_.pad_l != runInfo_.pad_r) {
        OP_LOGD(opName_, "Winograd tiling is not support current attrs");
        return false;
    }

    if (runInfo_.kh != 3 ||runInfo_.kw != 3 ) {
        OP_LOGD(opName_, "Winograd tiling only support 3*3 kernel");
        return false;
    }

    if ((runInfo_.ho / 2) * (runInfo_.wo / 2) * runInfo_.batch > 65536) {
        //累加轴过大暂时不处理，winograd累加轴比常规实现少了4倍，应该能囊括绝大部分case
        //有需要可以适当放大
        OP_LOGD(opName_, "current reduce asix is too large for Winograd impl");
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
    uint32_t tilingFlag = 1;
    if (singleShapeTile_ == B16H2W32_B32H2W16) {
        tilingFlag = 1;
    } else if (singleShapeTile_ == B16H8W8_B32H4W8) {
        tilingFlag = 2;
    }else if (singleShapeTile_ == B16H4W16_B32H2W16) {
        tilingFlag = 3;
    }
    const uint64_t tilingKey = GET_TPL_TILING_KEY(1, 0, 0,tilingFlag);
    OP_LOGD(context_->GetNodeName(), "tilingKey is: [%lu] , use winograd tiling flag [%lu]", tilingKey, tilingFlag);
    return tilingKey;
}


Conv3DBackpropFilterV2WinogradTiling::SingleShapeTile SelectTemplate(uint32_t tileH, uint32_t tileW, bool isFp32)
{
    //B16H2W32_B32H2W16
    uint32_t singleShapeTileH0 = 2;
    uint32_t singleShapeTileW0 = isFp32 ? 16 : 32;

    //B16H8W8_B32H4W8
    uint32_t singleShapeTileH1 = isFp32 ? 4 : 8;
    uint32_t singleShapeTileW1 = 8;

    //B16H4W16_B32H2W16
    uint32_t singleShapeTileH2 = isFp32 ? 2 : 4;
    uint32_t singleShapeTileW2 = 16;

    uint32_t clusters0 = Ops::Base::CeilDiv(tileH, singleShapeTileH0) *
                        Ops::Base::CeilDiv(tileW, singleShapeTileW0);

    uint32_t clusters1 = Ops::Base::CeilDiv(tileH, singleShapeTileH1) *
                            Ops::Base::CeilDiv(tileW, singleShapeTileW1);

    uint32_t clusters2 = Ops::Base::CeilDiv(tileH, singleShapeTileH2) *
                                Ops::Base::CeilDiv(tileW, singleShapeTileW2);

    //谁的空转块更少选谁,计算块一样多时，当前选择H4W16,冗余数据量相比H2W32小，同时内轴更大，16个C0应该能用一个outstanding发出去
    if (clusters0 == clusters1 && clusters1 == clusters2) {
        return Conv3DBackpropFilterV2WinogradTiling::B16H4W16_B32H2W16;
    }
    if (clusters0 == std::min({clusters0, clusters1, clusters2})) {
        return Conv3DBackpropFilterV2WinogradTiling::B16H2W32_B32H2W16;
    }
    if (clusters1 == std::min({clusters0, clusters1, clusters2})) {
        return Conv3DBackpropFilterV2WinogradTiling::B16H8W8_B32H4W8;
    }
    if (clusters2 == std::min({clusters0, clusters1, clusters2})) {
        return Conv3DBackpropFilterV2WinogradTiling::B16H4W16_B32H2W16;
    }

    //should not reach here
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
    size_t userWorkSpaceSize = 0;

    uint32_t singleShapeTileH;
    uint32_t singleShapeTileW;
    if (singleShapeTile_ == B16H2W32_B32H2W16) {
        singleShapeTileH = 2;
        singleShapeTileW = runInfo_.a_dtype_bytes == 2 ? 32 : 16;
    } else if (singleShapeTile_ == B16H8W8_B32H4W8) {
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

    userWorkSpaceSize = std::max(c1c0Fmap, c1c0Dy) * k0 * k1 * runInfo_.batch;

    workspaces[0] = WORKSPACE + userWorkSpaceSize;
    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("Conv3DBackpropFilterV2", Conv3DBackpropFilterV2WinogradTiling, 2);
}
}
}
#endif //CONV3D_BACKPROP_FILTER_V2_WINOGRAD_TILING_CPP