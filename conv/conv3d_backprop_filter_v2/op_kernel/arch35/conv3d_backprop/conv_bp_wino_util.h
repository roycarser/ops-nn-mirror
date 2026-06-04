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
 * \file conv_bp_wino_util.h
 * \brief
 */

#ifndef CONV_BP_WINO_UTIL_H
#define CONV_BP_WINO_UTIL_H

#include "basic_api/kernel_basic_intf.h"
#include "utils/std/algorithm.h"
#include "op_kernel/math_util.h"

template <typename T>
static constexpr __aicore__ inline uint32_t C0()
{
    return AscendC::DEFAULT_C0_SIZE / sizeof(T);
}

template <typename T>
static constexpr __aicore__ inline uint32_t VL()
{
    return AscendC::VECTOR_REG_WIDTH / sizeof(T);
}

struct HWBox {
    uint32_t hIdx = 0;
    uint32_t wIdx = 0;
    uint32_t hLength = 0;
    uint32_t wLength = 0;
    uint32_t elements = 0;
};

struct HWPad {
    uint16_t hTop = 0;
    uint16_t hBottom = 0;
    uint16_t wLeft = 0;
    uint16_t wRight = 0;
};

template <uint32_t STRIDE, uint32_t WINDOW_SIZE>
class SlideWindows {
public:
    static __aicore__ inline void CalculateSrcBox(
        const HWBox& tile,
        uint32_t srcH,
        uint32_t srcW,
        uint32_t padH,
        uint32_t padW,
        HWBox& outSrc,
        HWPad& outPad)
    {
        //将tile转换成(src+pad)中坐标[start,end)
        uint32_t startH = tile.hIdx * STRIDE;
        uint32_t startW = tile.wIdx * STRIDE;
        uint32_t endH = (tile.hIdx + tile.hLength - 1) * STRIDE + WINDOW_SIZE;
        uint32_t endW = (tile.wIdx + tile.wLength - 1) * STRIDE + WINDOW_SIZE;

        //(src+pad)中实际非pad区域的坐标[start,end)
        uint32_t startValidH = padH;
        uint32_t startValidW = padW;
        uint32_t endValidH = padH + srcH;
        uint32_t endValidW = padW + srcW;

        //计算两个区域的相交矩形[start,end)
        uint32_t startSrcH = AscendC::Std::max(startValidH, startH);
        uint32_t startSrcW = AscendC::Std::max(startValidW, startW);
        uint32_t endSrcH = AscendC::Std::min(endH, endValidH);
        uint32_t endSrcW = AscendC::Std::min(endW, endValidW);

        //tile区域和非padding区域不相交,整个tile都是在padding区域内
        if (startSrcH >= endSrcH || startSrcW >= endSrcW) {
            outSrc.hIdx = 0;
            outSrc.wIdx = 0;
            outSrc.hLength = 0;
            outSrc.wLength = 0;
            outSrc.elements = 0;
            outPad.hTop = 0;
            outPad.hBottom = endH - startH;
            outPad.wLeft = 0;
            outPad.wRight = endW - startW;
        } else {
            outSrc.hIdx = startSrcH - padH;
            outSrc.wIdx = startSrcW - padW;
            outSrc.hLength = endSrcH - startSrcH;
            outSrc.wLength = endSrcW - startSrcW;
            outSrc.elements = outSrc.hLength * outSrc.wLength;
            outPad.hTop = startSrcH - startH;
            outPad.hBottom = endH - endSrcH;
            outPad.wLeft = startSrcW - startW;
            outPad.wRight = endW - endSrcW;
        }
    }

    static __aicore__ inline uint32_t SrcLength2Tiles(const uint32_t srcLength)
    {
        if constexpr (WINDOW_SIZE == STRIDE) {
            //滑窗大小和stride相同时可以简化下处理逻辑
            return Ops::Base::CeilDiv(srcLength, WINDOW_SIZE);
        }
        return Ops::Base::CeilDiv(
                   srcLength > WINDOW_SIZE ? srcLength - WINDOW_SIZE : 0,
                   STRIDE) + 1;
    }

    static constexpr __aicore__ inline uint32_t Tiles2SrcLength(const uint32_t tiles)
    {
        if constexpr (WINDOW_SIZE == STRIDE) {
            //滑窗大小和stride相同时可以简化下处理逻辑
            return tiles * WINDOW_SIZE;
        }
        //tile长度转换成fmap长度
        return tiles == 0 ? 0 : (tiles - 1) * STRIDE + WINDOW_SIZE;
    }

    static constexpr __aicore__ inline uint32_t Tiles2Elements(const uint32_t tiles)
    {
        //tile里的元素个数
        return WINDOW_SIZE * WINDOW_SIZE * tiles;
    }

    static __aicore__ inline uint32_t Tiles2Size(const uint32_t tiles)
    {
        //tile里单边元素个数
        return WINDOW_SIZE * tiles;
    }
};

static constexpr uint32_t F23_TRANSFORM_TILE_SIZE_4 = 4;
static constexpr uint32_t F23_TRANSFORM_TILE_ELEMENTS_16 = 16;

static constexpr __aicore__ inline uint32_t TileUnfoldElements(uint32_t tiles)
{
    return tiles * F23_TRANSFORM_TILE_ELEMENTS_16;
}

static constexpr __aicore__ inline uint32_t TileUnfoldSize(uint32_t tiles)
{
    return tiles * F23_TRANSFORM_TILE_SIZE_4;
}

namespace ConstexprMaths {
//用于编译期计算的特殊函数，很多库函数没有constexpr标记，没办法赋值给constexpr对象

template <typename T>
static constexpr __aicore__ inline T Max(const T src0, const T src1)
{
    return (src0 > src1) ? src0 : src1;
}

template <typename T>
static constexpr __aicore__ inline T Min(const T src0, const T src1)
{
    return (src0 < src1) ? src0 : src1;
}

static constexpr __aicore__ inline uint32_t AlignUp(const uint32_t a, const uint32_t b)
{
    return AscendC::AlignUp(a, b);
}

static constexpr __aicore__ inline uint32_t CeilDiv(const uint32_t a, const uint32_t b)
{
    return AscendC::ConstCeil(a, b);
}
}

class AivPartitioner {
public:
    static __aicore__ inline uint32_t CeilAvgLength(uint32_t length)
    {
        return Ops::Base::CeilDiv(length, AivNum());
    }

    template <typename T>
    static __aicore__ inline uint32_t Get2DAlignBufLength(uint32_t splitDim0, uint32_t innerDim1)
    {
        uint32_t length0 = CeilAvgLength(splitDim0);
        return Ops::Base::CeilAlign(length0 * innerDim1, C0<T>());
    }

    static __aicore__ inline void GetPartition(
        uint32_t length,
        uint32_t& outStartIdx, uint32_t& outLength)
    {
        uint32_t avgLength = CeilAvgLength(length);
        outStartIdx = avgLength * AivIdx();
        outLength = AscendC::Std::min(avgLength, length - outStartIdx);
    }

private:
    static __aicore__ inline uint32_t AivNum()
    {
        return AscendC::GetSubBlockNum();
    }

    static __aicore__ inline uint32_t AivIdx()
    {
        return AscendC::GetSubBlockIdx();
    }
};

namespace BlockConfig {
enum InputTensor {
    FMAP,
    DY,
};

template <
    uint16_t SingleShapeCoutVal,
    uint16_t SingleShapeCinVal,
    uint16_t SingleTransformC1Val,
    uint16_t SingleShapeTileHVal,
    uint16_t SingleShapeTileWVal,
    uint16_t SingleTransformBufCntVal,
    uint16_t SingleShapeResidentCValue,
    InputTensor ResidentTargetValue>
struct Tiling {
    static constexpr uint16_t SingleShapeCout = SingleShapeCoutVal;
    static constexpr uint16_t SingleShapeCin = SingleShapeCinVal;
    static constexpr uint16_t SingleTransformC1 = SingleTransformC1Val;
    static constexpr uint16_t SingleShapeTileH = SingleShapeTileHVal;
    static constexpr uint16_t SingleShapeTileW = SingleShapeTileWVal;
    static constexpr uint16_t SingleTransformBufCnt = SingleTransformBufCntVal;
    static constexpr uint16_t SingleShapeResidentC = SingleShapeResidentCValue;
    static constexpr InputTensor ResidentTarget = ResidentTargetValue;
};

template <typename TilingT>
static constexpr __aicore__ inline uint16_t SingleShapeCout()
{
    return TilingT::SingleShapeCout;
}

template <typename TilingT>
static constexpr __aicore__ inline uint16_t SingleShapeCin()
{
    return TilingT::SingleShapeCin;
}

template <typename TilingT, InputTensor TensorType>
static constexpr __aicore__ inline uint16_t SingleShapeC()
{
    if constexpr (TensorType == FMAP) {
        return SingleShapeCin<TilingT>();
    } else if constexpr (TensorType == DY) {
        return SingleShapeCout<TilingT>();
    }
}

template <typename TilingT>
static constexpr __aicore__ inline uint16_t SingleTransformC1()
{
    return TilingT::SingleTransformC1;
}

template <typename TilingT>
static constexpr __aicore__ inline uint16_t SingleShapeTileH()
{
    return TilingT::SingleShapeTileH;
}

template <typename TilingT>
static constexpr __aicore__ inline uint16_t SingleShapeTileW()
{
    return TilingT::SingleShapeTileW;
}

template <typename TilingT>
static constexpr __aicore__ inline uint16_t SingleShapeTileHW()
{
    return SingleShapeTileH<TilingT>() * SingleShapeTileW<TilingT>();
}

template <typename TilingT>
static constexpr __aicore__ inline uint16_t SingleTransformBufCnt()
{
    return TilingT::SingleTransformBufCnt;
}

template <typename TilingT>
static constexpr __aicore__ inline uint16_t SingleShapeResidentC()
{
    return TilingT::SingleShapeResidentC;
}

template <typename TilingT>
static constexpr __aicore__ inline InputTensor ResidentTarget()
{
    return TilingT::ResidentTarget;
}
}


#endif //CONV_BP_WINO_UTIL_H