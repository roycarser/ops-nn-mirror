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
 * \file conv_bp_winograd.h
 * \brief
 */


#ifndef CONV_BP_WINOGRAD_H
#define CONV_BP_WINOGRAD_H

#include "basic_api/kernel_basic_intf.h"
#include "utils/std/algorithm.h"
#include "op_kernel/math_util.h"

using namespace AscendC;

//winograd在f23下单个tile长宽为4,元素数为16,fmap上stride为2
static constexpr uint32_t F23_WINO_TILE_SIZE_4 = 4;
static constexpr uint32_t F23_WINO_TILE_ELEMENTS_16 = 16;
static constexpr uint32_t F23_WINO_TILE_FMAP_STRIDE_2 = 2;

template <typename T>
static constexpr __aicore__ inline uint32_t C0()
{
    return DEFAULT_C0_SIZE / sizeof(T);
}

template <typename T>
static constexpr __aicore__ inline uint32_t VL()
{
    return VECTOR_REG_WIDTH / sizeof(T);
}

template <typename T, typename U>
static __aicore__ inline U C1(U c)
{
    constexpr uint32_t c0 = C0<T>();
    if constexpr (c0 == 16) {
        return (c + 15) >> 4;
    } else if constexpr (c0 == 8) {
        return (c + 7) >> 3;
    } else {
        static_assert(!(c0 == 16 || c0 == 8), "unsupported c0 size");
        return 0;
    }
}

template <typename T, typename U>
static __aicore__ inline U AlignC0(U c)
{
    if constexpr (constexpr uint32_t c0 = C0<T>(); c0 == 16) {
        return (c + 15) & ~15;
    } else if constexpr (c0 == 8) {
        return (c + 7) & ~7;
    } else {
        static_assert(!(c0 == 16 || c0 == 8), "unsupported c0 size");
        return 0;
    }
}

//一个位宽能包含几个DataBlock(c0)
static constexpr uint32_t BLK_COUNT_IN_VL = VECTOR_REG_WIDTH / DEFAULT_C0_SIZE;
//fmap在搬运时hw轴按照16元素对齐
//这里主要是为了方便Transpose5HD在float32做转置
static constexpr uint32_t FMAP_HW_ALIGNED_16 = 16;


struct HWBox {
    uint32_t hIdx;
    uint32_t wIdx;
    uint32_t hLength;
    uint32_t wLength;
    uint32_t elements;
};

struct HWPad {
    uint16_t hTop;
    uint16_t hBottom;
    uint16_t wLeft;
    uint16_t wRight;

    static __aicore__ inline bool exists(const HWPad& expansion)
    {
        return expansion.hTop != 0
               || expansion.hBottom != 0
               || expansion.wLeft != 0
               || expansion.wRight != 0;
    }
};

struct TileBox {
    HWBox tile;
    HWBox fmap;
    HWPad pad;
};

class F23FmapTileCalculator {
public:
    static __aicore__ inline TileBox CalculateTileBox(
        const HWBox& tile,
        uint32_t hi,
        uint32_t wi,
        uint32_t padH,
        uint32_t padW)
    {
        // F23TileBox2FmapBox(box.tile, box.fmap, box.pad);
        //将tile转换成(fmap+pad)中坐标[start,end)
        uint32_t startH = tile.hIdx * F23_WINO_TILE_FMAP_STRIDE_2;
        uint32_t startW = tile.wIdx * F23_WINO_TILE_FMAP_STRIDE_2;
        uint32_t endH = (tile.hIdx + tile.hLength - 1) * F23_WINO_TILE_FMAP_STRIDE_2 + F23_WINO_TILE_SIZE_4;
        uint32_t endW = (tile.wIdx + tile.wLength - 1) * F23_WINO_TILE_FMAP_STRIDE_2 + F23_WINO_TILE_SIZE_4;

        //(fmap+pad)中实际非pad区域的坐标[start,end)
        uint32_t startValidH = padH;
        uint32_t startValidW = padW;
        uint32_t endValidH = padH + hi;
        uint32_t endValidW = padW + wi;

        //计算两个区域的相交矩形[start,end)
        uint32_t startFmapH = Std::max(startValidH, startH);
        uint32_t startFmapW = Std::max(startValidW, startW);
        uint32_t endFmapH = Std::min(endH, endValidH);
        uint32_t endFmapW = Std::min(endW, endValidW);

        //tile区域和非padding区域不相交,整个tile都是在padding区域内
        bool allTilesInPadding = startFmapH >= endFmapH || startFmapW >= endFmapW;

        TileBox box = {};
        box.tile = tile;
        if (allTilesInPadding) {
            box.fmap.hIdx = 0;
            box.fmap.wIdx = 0;
            box.fmap.hLength = 0;
            box.fmap.wLength = 0;
            box.fmap.elements = 0;
            box.pad.hTop = 0;
            box.pad.hBottom = endH - startH;
            box.pad.wLeft = 0;
            box.pad.wRight = endW - startW;
        } else {
            box.fmap.hIdx = startFmapH - padH;
            box.fmap.wIdx = startFmapW - padW;
            box.fmap.hLength = endFmapH - startFmapH;
            box.fmap.wLength = endFmapW - startFmapW;
            box.fmap.elements = box.fmap.hLength * box.fmap.wLength;
            box.pad.hTop = startFmapH - startH;
            box.pad.hBottom = endH - endFmapH;
            box.pad.wLeft = startFmapW - startW;
            box.pad.wRight = endW - endFmapW;
        }

        return box;
    }


    static __aicore__ inline uint32_t Tiles(uint32_t dim)
    {
        //F23下,Fmap正变换是一个4*4的滑窗在Fmap上按stride为2滑动
        //所以dim至少pad到4
        return Ops::Base::CeilDiv(
                   dim > F23_WINO_TILE_SIZE_4 ? dim - F23_WINO_TILE_SIZE_4 : 0,
                   F23_WINO_TILE_FMAP_STRIDE_2) + 1;
    }

    static __aicore__ inline uint32_t FmapLength(uint32_t tiles)
    {
        //tile长度转换成fmap长度
        return tiles == 0 ? 0 : (tiles - 1) * F23_WINO_TILE_FMAP_STRIDE_2 + F23_WINO_TILE_SIZE_4;
    }
};


template <typename T>
class WinoFmapTransformer {
public:
    __aicore__ inline WinoFmapTransformer(
        TPipe* pipe,
        __gm__ T* gm,
        const uint32_t cin,
        const uint32_t hi,
        const uint32_t wi,
        const uint16_t padH,
        const uint16_t padW,
        const uint16_t singleShapeCin,
        const uint16_t singleShapeTilesH,
        const uint16_t singleShapeTilesW,
        const uint16_t transposeBufCnt)
        : cin_(cin),
          hi_(hi),
          wi_(wi),
          padH_(padH),
          padW_(padW),
          tilesH_(F23FmapTileCalculator::Tiles(hi_ + 2 * padH_)),
          tilesW_(F23FmapTileCalculator::Tiles(wi_ + 2 * padW_)),
          singleShapeCin_(singleShapeCin),
          singleShapeTilesH_(singleShapeTilesH),
          singleShapeTilesW_(singleShapeTilesW),
          //额外提供的transposeBuf的数量,多的话或许能提高一些并行度减少同步，但是会占用更多空间
          transposeBufCnt_(transposeBufCnt),
          //额外申请[c0,align16(hw)]的空间给fmap做c1hwc0转置
          transposeBufLength_(
              CalculateTransposeBufLength(
                  singleShapeTilesH, singleShapeTilesW)),
          singleShapeBufLength_(
              CalculateSingleShapeBufLength(
                  singleShapeCin, singleShapeTilesH, singleShapeTilesW))
    {
        fmapGm_.SetGlobalBuffer(gm);

        //transposeBuf全局只需要一份就行
        uint32_t doubleBufferLength = singleShapeBufLength_ * 2 + transposeBufCnt * transposeBufLength_;
        //gm->vec->l1这条路这TQue里配置的同步事件不符合实际场景,所以自己用TBuf管理同步
        pipe->InitBuffer(vBuf_, doubleBufferLength * sizeof(T));
    }

    static inline uint32_t __aicore__ CalculateTransposeBufLength(uint32_t th, uint32_t tw)
    {
        uint32_t fmapH = F23FmapTileCalculator::FmapLength(th);
        uint32_t fmapW = F23FmapTileCalculator::FmapLength(tw);
        return Ops::Base::CeilAlign(fmapH * fmapW, FMAP_HW_ALIGNED_16) * C0<T>();
    }

    static inline uint32_t __aicore__ CalculateSingleShapeBufLength(
        uint32_t cin, uint32_t tw, uint32_t th)
    {
        //单次变换需要的buf为c1c0*tiles*16
        uint32_t c1c0 = AlignC0<T>(cin);
        uint32_t tiles = tw * th;
        uint32_t tileBuf = c1c0 * tiles * F23_WINO_TILE_ELEMENTS_16;
        return tileBuf;
    }

    struct TileKIter {
        uint32_t tileHIdx = 0;
        uint32_t tileWIdx = 0;
        bool end = false;
    };

    __aicore__ inline void Transform(
        uint32_t batchIdx,
        uint32_t cIdx,
        TileKIter& iter,
        GlobalTensor<T>& out)
    {
        //上层控制
        ascendc_assert(iter.end==false);

        uint32_t cLength = Std::min(static_cast<uint32_t>(singleShapeCin_), cin_ - cIdx);
        uint32_t tileHLength = Std::min(static_cast<uint32_t>(singleShapeTilesH_), tilesH_ - iter.tileHIdx);
        uint32_t tileWLength = Std::min(static_cast<uint32_t>(singleShapeTilesW_), tilesW_ - iter.tileWIdx);

        if ASCEND_IS_AIV {
            LocalTensor<T> mainBuf = GetPingPongBuffer();
            LocalTensor<T> transposeBuf = GetTransposeBuffer();

            const TileBox box = F23FmapTileCalculator::CalculateTileBox(
                {iter.tileHIdx, iter.tileWIdx,
                 tileHLength, tileWLength,
                 tileHLength * tileWLength},
                hi_, wi_, padH_, padW_);

            CopyIn(mainBuf, batchIdx, cIdx, cLength, box);
            // TODO setFlag/waitFlag
            PipeBarrier<PIPE_ALL>();
            Compute(mainBuf, transposeBuf, cLength, box);
            PipeBarrier<PIPE_ALL>();
            CopyOut(mainBuf, transposeBuf, box, cLength, out);
        }

        iter.tileWIdx += singleShapeTilesW_;
        if (iter.tileWIdx >= tilesW_) {
            iter.tileWIdx = 0;
            iter.tileHIdx += singleShapeTilesH_;
            iter.end = iter.tileHIdx >= tilesH_;
        }
    }

private:
    __aicore__ inline void CopyOut(
        LocalTensor<T>& buf,
        LocalTensor<T>& transposeBuf,
        const TileBox& box,
        uint16_t cLength,
        GlobalTensor<T>& out)
    {
        DataCopy(out, buf, buf.GetSize());
        // DataCopy(out, transposeBuf, transposeBuf.GetSize());
    }


    __aicore__ inline void CopyIn(
        LocalTensor<T>& buf,
        uint32_t batchIdx,
        uint32_t cIdx,
        uint32_t cLength,
        const TileBox& box)
    {
        if (const HWBox& fmap = box.fmap; fmap.elements != 0) {
            DataCopyExtParams params;
            params.blockCount = fmap.hLength;
            params.blockLen = fmap.wLength * sizeof(T);
            params.srcStride = wi_ * sizeof(T);
            params.dstStride = 0;

            uint32_t fmapHWAligned16 = Ops::Base::CeilAlign(fmap.elements, FMAP_HW_ALIGNED_16);

            LoopModeParams loop;
            loop.loop1Size = cLength;
            loop.loop1SrcStride = wi_ * hi_ * sizeof(T);
            loop.loop1DstStride = fmapHWAligned16 * sizeof(T);
            loop.loop2Size = 1;
            loop.loop1SrcStride = 0;
            loop.loop2DstStride = 0;

            SetLoopModePara(loop, DataCopyMVType::OUT_TO_UB);

            // 在大小为[c1,th4,tw4,c0]的buf里面从搬入大小为[c,aligned16(hw)]大小的fmap
            // 由于c1c0 >= c
            // 且根据数学推到[th4,tw4] >= align16(hw)
            // 所以当前buf一定能放得下所有数据

            DataCopyPad<T, PaddingMode::Compact>(
                buf,
                fmapGm_[(batchIdx * cin_ + cIdx) * hi_ * wi_ + fmap.hIdx * wi_ + fmap.wIdx],
                params,
                {false, 0, 0, 0});
            ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
        }
    }


    __aicore__ inline void Compute(
        LocalTensor<T>& mainBuf,
        LocalTensor<T>& transposeBuf,
        uint32_t cLength,
        const TileBox& box)
    {
        if (const HWBox& fmap = box.fmap; fmap.elements != 0) {
            uint32_t fmapHWAligned16 = Ops::Base::CeilAlign(fmap.elements, FMAP_HW_ALIGNED_16);
            uint32_t c1 = C1<T>(cLength);

            if (uint32_t tailC0 = c1 * C0<T>() - cLength; tailC0 != 0) {
                //cLength不是c0对齐的,给他补0补到c0对齐
                Duplicate(
                    mainBuf[cLength * fmapHWAligned16],
                    static_cast<T>(0),
                    tailC0 * fmapHWAligned16);
            }

            uint32_t tileBufWidthC0Blocks = box.tile.wLength * F23_WINO_TILE_SIZE_4;
            uint32_t tileBufWidth = tileBufWidthC0Blocks * C0<T>();
            uint32_t tileBufC1Stride = box.tile.elements * F23_WINO_TILE_ELEMENTS_16 * C0<T>();
            uint32_t fmapAligned16C1Stride = fmapHWAligned16 * C0<T>();

            UnfoldRowParamsV2 urp = {};
            urp.fmapWidth = fmap.wLength * C0<T>();
            //行展开后放在tileBuf的右侧,
            urp.wStoreOffset = tileBufWidth - (fmap.wLength + box.pad.wRight) * C0<T>();
            urp.wRepeatTimes = Ops::Base::CeilDiv(urp.fmapWidth, VL<T>());
            urp.tileHTailRepeatTimes = box.tile.hLength & 1;
            urp.tileHMainRepeatTimes = box.tile.hLength >> 1;

            uint32_t fmapLeftBoundOffset = (box.tile.wLength * 2 - 2) * C0<T>();

            UnfoldColParamsV2 ucp = {};
            ucp.hValidElements = box.tile.hLength * F23_WINO_TILE_SIZE_4 * C0<T>();
            ucp.hRepeatTimes = Ops::Base::CeilDiv(ucp.hValidElements, VL<T>());
            ucp.tileWTailRepeatTimes = box.tile.wLength & 1;
            ucp.tileWMainRepeatTimes = box.tile.wLength >> 1;
            ucp.fmapLeftBoundOffset = fmapLeftBoundOffset;

            //每次处理最多transposeBufCnt条c1
            uint32_t cnt = Ops::Base::CeilDiv(c1, static_cast<uint32_t>(transposeBufCnt_));
            for (uint32_t i = 0; i < cnt; i++) {
                //从末端开始处理，由于mainBuf的头部搬入了整块fmap
                //如果顺序处理,那可能在变换第一个c1的fmap时,污染了第二个c1的fmap
                //逆序处理就没这个问题,因为尾部不会连着其他fmap
                uint32_t c1Idx = (cnt - i - 1) * transposeBufCnt_;
                uint32_t c1Length = Std::min(c1 - c1Idx, static_cast<uint32_t>(transposeBufCnt_));

                TransposeCHW2C1HWC0(
                    mainBuf[c1Idx * fmapAligned16C1Stride],
                    transposeBuf[box.pad.hTop * box.fmap.wLength * C0<T>()],
                    c1Length,
                    fmapHWAligned16,
                    fmapAligned16C1Stride,
                    transposeBufLength_);
                //TODO 处理pad
                LocalTensor<T> tileBuf = mainBuf[c1Idx * tileBufC1Stride];
                //行变换
                UnfoldFromFmapBufVf(
                    reinterpret_cast<__ubuf__ T*>(tileBuf.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ T*>(transposeBuf.GetPhyAddr()),
                    c1Length,
                    tileBufC1Stride,
                    transposeBufLength_,
                    tileBufWidth,
                    tileBufWidthC0Blocks,
                    urp, ucp);
            }
        } else {
            //整个tile都由padding区域产生,不做计算直接置0,
            Duplicate(
                mainBuf,
                static_cast<T>(0),
                box.tile.elements * AlignC0<T>(cLength) * F23_WINO_TILE_ELEMENTS_16);
        }
    }


    __aicore__ static inline void TransposeCHW2C1HWC0(
        const LocalTensor<T>& srcBuf,
        const LocalTensor<T>& dstBuf,
        uint32_t c1Length,
        uint32_t fmapHWAligned16,
        uint32_t srcC1Stride,
        uint32_t dstC1Stride)
    {
        uint64_t srcList[16];
        uint64_t dstList[16];

        for (uint32_t c1 = 0; c1 < c1Length; c1++) {
            uint32_t srcOffset = c1 * srcC1Stride;
            uint32_t dstOffset = c1 * dstC1Stride;

            if constexpr (sizeof(T) == 2) {
#pragma unroll
                for (uint32_t i = 0; i < 16; i++) {
                    uint32_t s = srcOffset + i * fmapHWAligned16;
                    uint32_t d = dstOffset + i * 16;
                    srcList[i] = reinterpret_cast<uint64_t>(srcBuf[s].GetPhyAddr());
                    dstList[i] = reinterpret_cast<uint64_t>(dstBuf[d].GetPhyAddr());
                }
                TransDataTo5HDParams params;
                params.repeatTimes = fmapHWAligned16 / 16;
                params.srcRepStride = params.repeatTimes == 1 ? 0 : 1;
                params.dstRepStride = params.repeatTimes == 1 ? 0 : 16;
                TransDataTo5HD<T>(dstList, srcList, params);
            } else if constexpr (sizeof(T) == 4) {
#pragma unroll
                for (uint32_t i = 0; i < 8; i++) {
                    uint32_t s = srcOffset + i * fmapHWAligned16;
                    uint32_t d = dstOffset + i * 8;
                    srcList[i] = reinterpret_cast<uint64_t>(srcBuf[s].GetPhyAddr());
                    srcList[i + 8] = reinterpret_cast<uint64_t>(srcBuf[s + 8].GetPhyAddr());
                    dstList[i * 2] = reinterpret_cast<uint64_t>(dstBuf[d].GetPhyAddr());
                    dstList[i * 2 + 1] = reinterpret_cast<uint64_t>(dstBuf[d + 8 * 8].GetPhyAddr());
                }

                TransDataTo5HDParams params;
                params.repeatTimes = fmapHWAligned16 / 16;
                params.srcRepStride = params.repeatTimes == 1 ? 0 : 2;
                params.dstRepStride = params.repeatTimes == 1 ? 0 : 16;
                TransDataTo5HD<T>(dstList, srcList, params);
            }
        }
    }

    struct UnfoldColParamsV2 {
        uint32_t hValidElements;
        uint32_t fmapLeftBoundOffset;
        uint16_t hRepeatTimes;
        uint16_t tileWMainRepeatTimes;
        uint16_t tileWTailRepeatTimes;
    };

    struct UnfoldRowParamsV2 {
        uint32_t fmapWidth;
        uint32_t wStoreOffset;
        uint16_t wRepeatTimes;
        uint16_t tileHMainRepeatTimes;
        uint16_t tileHTailRepeatTimes;
    };

    static __simd_vf__ inline void UnfoldFromFmapBufVf(
        __ubuf__ T* __restrict__ tileBuf,
        __ubuf__ T* __restrict__ fmapBuf,
        const uint16_t c1Length,
        const uint32_t tileBufC1Stride,
        const uint32_t fmapBufC1Stride,
        const uint32_t tileBufWidth,
        const uint16_t tileBufWidthC0Blocks,
        const UnfoldRowParamsV2 urp,
        const UnfoldColParamsV2 ucp)
    {
        __ubuf__ T* tmp = tileBuf + urp.wStoreOffset;
        for (uint16_t c1 = 0; c1 < c1Length; c1++) {
            __ubuf__ T* t = tmp + c1 * tileBufC1Stride;
            __ubuf__ T* f = fmapBuf + c1 * fmapBufC1Stride;
            UnfoldRowsFromFmapBufVf(t, f, tileBufWidth, urp);
        }

        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_LOAD, MicroAPI::MemType::VEC_STORE>();

        for (uint16_t c1 = 0; c1 < c1Length; c1++) {
            __ubuf__ T* t = tileBuf + c1 * tileBufC1Stride;
            UnfoldColsVf(t, tileBufWidthC0Blocks, tileBufWidth, ucp);
        }
    }

    // 执行行变换
    // 原始数据:
    //
    //      w0 w1 w2 w3 w4 w5
    //     +--+--+--+--+--+--+
    //  h0 |xx|xx|xx|xx|xx|xx|
    //     +--+--+--+--+--+--+
    //  h1 |xx|xx|xx|xx|xx|xx|
    //     +--+--+--+--+--+--+
    //  h2 |xx|xx|xx|xx|xx|xx|
    //     +--+--+--+--+--+--+
    //  h3 |xx|xx|xx|xx|xx|xx|
    //     +--+--+--+--+--+--+
    //  h4 |xx|xx|xx|xx|xx|xx|
    //     +--+--+--+--+--+--+
    //  h5 |xx|xx|xx|xx|xx|xx|
    //     +--+--+--+--+--+--+
    //
    // 变换后内存:
    //
    //           w0         w1    w2-w4     w5     w6 w7
    //      +---------+---------+-------+---------+--+--+
    //  Th0 |h0w0-h2w0|h0w1-h2w1|.......|h0w5-h2w5|  |  |
    //      +---------+---------+-------+---------+--+--+
    //  Th1 |h1w0+h2w0|h1w1+h2w1|.......|h1w5+h2w5|  |  |
    //      +---------+---------+-------+---------+--+--+
    //  Th2 |h2w0-h1w0|h2w1-h1w1|.......|h2w5-h1w5|  |  |
    //      +---------+---------+-------+---------+--+--+
    //  Th3 |h1w0-h3w0|h1w1-h3w1|.......|h1w5-h3w5|  |  |
    //      +---------+---------+-------+---------+--+--+
    //  Th4 |h2w0-h4w0|h2w1-h4w1|.......|h2w5-h4w5|  |  |
    //      +---------+---------+-------+---------+--+--+
    //  Th5 |h4w0+h3w0|h4w1+h3w1|.......|h4w5+h3w5|  |  |
    //      +---------+---------+-------+---------+--+--+
    //  Th6 |h4w0-h3w0|h4w1-h3w1|.......|h4w5-h3w5|  |  |
    //      +---------+---------+-------+---------+--+--+
    //  Th7 |h3w0-h5w0|h3w1-h5w1|.......|h3w5-h5w5|  |  |
    //      +---------+---------+-------+---------+--+--+
    // 数据在H方向上展开后写入
    //
    static __simd_callee__ inline void UnfoldRowsFromFmapBufVf(
        __ubuf__ T* __restrict__ tileBuf,
        __ubuf__ T* __restrict__ fmapBuf,
        const uint32_t tileBufWidth,
        const UnfoldRowParamsV2& params)
    {
        const uint32_t fmapWidth = params.fmapWidth;
        const uint16_t tileHMainRepeatTimes = params.tileHMainRepeatTimes;
        const uint16_t tileHTailRepeatTimes = params.tileHTailRepeatTimes;
        const uint16_t wRepeatTimes = params.wRepeatTimes;

        uint32_t maskValue = fmapWidth;
        for (uint16_t i = 0; i < wRepeatTimes; i++) {
            MicroAPI::MaskReg mask = MicroAPI::UpdateMask<T>(maskValue);
            MicroAPI::RegTensor<T> s0;
            MicroAPI::RegTensor<T> s1;
            MicroAPI::RegTensor<T> s2;
            MicroAPI::RegTensor<T> s3;

            MicroAPI::RegTensor<T> d0;
            MicroAPI::RegTensor<T> d1;
            MicroAPI::RegTensor<T> d2;
            MicroAPI::RegTensor<T> d3;

            // 从最上方的tile开始滑窗
            // 先读取fmap首2行,每次循环往下读2行凑成4行执行变换
            // 但若一个滑窗在fmap的1-4行分别读入s0,s1,s2,s3
            // 在下一个滑窗s2,s3就变成1-2行,不考虑重新读取的话2-3行就只能读入s0,s1,滑窗1-4行就变成s2,s3,s0,s1
            // 如果将s0,s1的数据拷贝到s2,s3可能会产生多余的指令并且产生数据依赖降低性能
            // vf内scalar能力较弱不确定怎么样的代码编译器能正常优化
            // 所以这里按照最朴素的方式展开循环一个循环内处理2个连续滑窗,
            // 如果滑窗为奇数,则通过tileHTailRepeatTimes额外执行一次滑窗

            //循环fmapW
            const uint32_t wOffset = i * VL<T>();

            __ubuf__ T* src = fmapBuf + wOffset;
            MicroAPI::LoadAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(s0, src, fmapWidth);
            MicroAPI::LoadAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(s1, src, fmapWidth);

            __ubuf__ T* dst = tileBuf + wOffset;
            for (uint16_t th = 0; th < tileHMainRepeatTimes; th++) {
                MicroAPI::LoadAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(s2, src, fmapWidth);
                MicroAPI::LoadAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(s3, src, fmapWidth);

                TransformVf(s0, s1, s2, s3, d0, d1, d2, d3, mask);

                MicroAPI::StoreAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(dst, d0, tileBufWidth, mask);
                MicroAPI::StoreAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(dst, d1, tileBufWidth, mask);
                MicroAPI::StoreAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(dst, d2, tileBufWidth, mask);
                MicroAPI::StoreAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(dst, d3, tileBufWidth, mask);

                MicroAPI::LoadAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(s0, src, fmapWidth);
                MicroAPI::LoadAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(s1, src, fmapWidth);

                TransformVf(s2, s3, s0, s1, d0, d1, d2, d3, mask);

                MicroAPI::StoreAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(dst, d0, tileBufWidth, mask);
                MicroAPI::StoreAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(dst, d1, tileBufWidth, mask);
                MicroAPI::StoreAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(dst, d2, tileBufWidth, mask);
                MicroAPI::StoreAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(dst, d3, tileBufWidth, mask);
            }

            for (uint16_t th = 0; th < tileHTailRepeatTimes; th++) {
                MicroAPI::LoadAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(s2, src, fmapWidth);
                MicroAPI::LoadAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(s3, src, fmapWidth);

                TransformVf(s0, s1, s2, s3, d0, d1, d2, d3, mask);

                MicroAPI::StoreAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(dst, d0, tileBufWidth, mask);
                MicroAPI::StoreAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(dst, d1, tileBufWidth, mask);
                MicroAPI::StoreAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(dst, d2, tileBufWidth, mask);
                MicroAPI::StoreAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(dst, d3, tileBufWidth, mask);
            }
        }
    }

    // 在行变换的结果上执行变换
    // 原始布局S:
    //
    //      w0 w1 w2 w3 w4 w5 w6 w7
    //     +--+--+--+--+--+--+--+--+
    //  h0 |  |  |xx|xx|xx|xx|xx|xx|
    //     +--+--+--+--+--+--+--+--+
    //  h1 |  |  |xx|xx|xx|xx|xx|xx|
    //     +--+--+--+--+--+--+--+--+
    //  h2 |  |  |xx|xx|xx|xx|xx|xx|
    //     +--+--+--+--+--+--+--+--+
    //  h3 |  |  |xx|xx|xx|xx|xx|xx|
    //     +--+--+--+--+--+--+--+--+
    //  h4 |  |  |xx|xx|xx|xx|xx|xx|
    //     +--+--+--+--+--+--+--+--+
    //  h5 |  |  |xx|xx|xx|xx|xx|xx|
    //     +--+--+--+--+--+--+--+--+
    //  h6 |  |  |xx|xx|xx|xx|xx|xx|
    //     +--+--+--+--+--+--+--+--+
    //  h7 |  |  |xx|xx|xx|xx|xx|xx|
    //     +--+--+--+--+--+--+--+--+
    //
    // 变换后内存:
    //
    //           w0        w1     w2-w6     w7
    //      +---------+---------+-------+---------+
    //  Th0 |h0w2-h0w4|h0w3+h0w4|.......|h0w5-h0w7|
    //      +---------+---------+-------+---------+
    //  Th1 |h1w2-h1w4|h1w3+h1w4|.......|h1w5-h1w7|
    //      +---------+---------+-------+---------+
    //  Th2 |h2w2-h2w4|h2w3+h2w4|.......|h2w5-h2w7|
    //      +---------+---------+-------+---------+
    //  Th3 |h3w2-h3w4|h3w3+h3w4|.......|h3w5-h3w7|
    //      +---------+---------+-------+---------+
    //  Th4 |h4w2-h4w4|h4w3+h4w4|.......|h4w5-h4w7|
    //      +---------+---------+-------+---------+
    //  Th5 |h5w2-h5w4|h5w3+h5w4|.......|h5w5-h5w7|
    //      +---------+---------+-------+---------+
    //  Th6 |h6w2-h6w4|h6w3+h6w4|.......|h6w5-h6w7|
    //      +---------+---------+-------+---------+
    //  Th7 |h7w2-h7w4|h7w3+h7w4|.......|h7w5-h7w7|
    //      +---------+---------+-------+---------+
    //

    static __simd_callee__ inline void UnfoldColsVf(
        __ubuf__ T* buf,
        const uint32_t tileBufWidthC0Blocks,
        const uint32_t tileBufWidth,
        const UnfoldColParamsV2& params)
    {
        const uint32_t hValidElements = params.hValidElements;
        const uint16_t hRepeatTimes = params.hRepeatTimes;
        const uint16_t tileWMainRepeatTimes = params.tileWMainRepeatTimes;
        const uint16_t tileWTailRepeatTimes = params.tileWTailRepeatTimes;
        //fmap靠在整块buf的右侧,所以读取时需要加个左边的偏移
        const uint32_t fmapLeftBoundOffset = params.fmapLeftBoundOffset;

        uint32_t maskValue = hValidElements;
        for (uint16_t i = 0; i < hRepeatTimes; i++) {
            MicroAPI::RegTensor<T> s0;
            MicroAPI::RegTensor<T> s1;
            MicroAPI::RegTensor<T> s2;
            MicroAPI::RegTensor<T> s3;

            MicroAPI::RegTensor<T> d0;
            MicroAPI::RegTensor<T> d1;
            MicroAPI::RegTensor<T> d2;
            MicroAPI::RegTensor<T> d3;

            MicroAPI::MaskReg mask = MicroAPI::UpdateMask<T>(maskValue);

            const uint32_t hOffset = tileBufWidth * i * BLK_COUNT_IN_VL;

            __ubuf__ T* src = buf + hOffset + fmapLeftBoundOffset;

            MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                s0, src, tileBufWidthC0Blocks, 1, mask);
            MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                s1, src, tileBufWidthC0Blocks, 1, mask);

            __ubuf__ T* dst = buf + hOffset;

            for (uint16_t tw = 0; tw < tileWMainRepeatTimes; tw++) {
                MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                    MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    s2, src, tileBufWidthC0Blocks, 1, mask);
                MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                    MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    s3, src, tileBufWidthC0Blocks, 1, mask);

                TransformVf(s0, s1, s2, s3, d0, d1, d2, d3, mask);

                MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                    MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    dst, d0, tileBufWidthC0Blocks, 1, mask);
                MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                    MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    dst, d1, tileBufWidthC0Blocks, 1, mask);
                MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                    MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    dst, d2, tileBufWidthC0Blocks, 1, mask);
                MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                    MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    dst, d3, tileBufWidthC0Blocks, 1, mask);

                MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                    MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    s0, src, tileBufWidthC0Blocks, 1, mask);
                MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                    MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    s1, src, tileBufWidthC0Blocks, 1, mask);

                TransformVf(s2, s3, s0, s1, d0, d1, d2, d3, mask);
                MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                    MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    dst, d0, tileBufWidthC0Blocks, 1, mask);
                MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                    MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    dst, d1, tileBufWidthC0Blocks, 1, mask);
                MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                    MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    dst, d2, tileBufWidthC0Blocks, 1, mask);
                MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                    MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    dst, d3, tileBufWidthC0Blocks, 1, mask);
            }

            for (uint16_t th = 0; th < tileWTailRepeatTimes; th++) {
                MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                    MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    s2, src, tileBufWidthC0Blocks, 1, mask);
                MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                    MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    s3, src, tileBufWidthC0Blocks, 1, mask);

                TransformVf(s0, s1, s2, s3, d0, d1, d2, d3, mask);

                MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                    MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    dst, d0, tileBufWidthC0Blocks, 1, mask);
                MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                    MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    dst, d1, tileBufWidthC0Blocks, 1, mask);
                MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                    MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    dst, d2, tileBufWidthC0Blocks, 1, mask);
                MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                    MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    dst, d3, tileBufWidthC0Blocks, 1, mask);
            }
        }
    }


    //        wElementsInPadHBlock(fmapW+wLeft+wRight)*c0
    //       +---------------------------------+
    //   hTop|             Padding             |
    //       +--wLeft--+-------------+---------+
    //       |         |  NoPadding  |         |
    //       | Padding |fmapH x      | Padding |hElementsInPadWBlock(fmapH)*c0
    //       |         |    fmapW    |         |
    //       +---------+-------------+--wRight-+
    //       |             Padding             |hButton
    //       +---------------------------------+

    // template <bool hTop, bool hBottom>
    // static __simd_vf__ inline void PadH(
    //     __ubuf__ T* buf,
    //     const uint16_t repeatTimes,
    //     const uint32_t repeatStride,
    //     const uint32_t buttonOffset,
    //     const uint16_t hTopLength,
    //     const uint16_t hButtonLength,
    //     const uint16_t dataBlocksInW,
    //     const uint32_t dataBlocksSkippedInButton)
    // {
    //     MicroAPI::RegTensor<T> value;
    //     MicroAPI::Duplicate(value, static_cast<T>(0));
    //     constexpr uint8_t blockElements = GetDataBlockSizeInBytes() / sizeof(T);
    //
    //     uint32_t hTopPaddingElements = 0;
    //     uint16_t hTopRepeatTimes = 0;
    //     if constexpr (hTop) {
    //         hTopPaddingElements = hTopLength * dataBlocksInW * blockElements;
    //         hTopRepeatTimes = CeilDivision(hTopPaddingElements, VL<T>());
    //     }
    //
    //     uint32_t hButtonPaddingElements = 0;
    //     uint16_t hButtonRepeatTimes = 0;
    //     __ubuf__ T* buttonBuf = buf;
    //     if constexpr (hBottom) {
    //         hButtonPaddingElements = (hButtonLength * dataBlocksInW - dataBlocksSkippedInButton) * blockElements;
    //         hButtonRepeatTimes = CeilDivision(hButtonPaddingElements, VL<T>());
    //         //TODO
    //         buttonBuf = buf + buttonOffset+dataBlocksSkippedInButton*blockElements;
    //     }
    //
    //     for (uint16_t i0 = 0; i0 < repeatTimes; i0++) {
    //         if constexpr (hTop) {
    //             uint32_t maskValueHTop = hTopPaddingElements;
    //             for (uint16_t i = 0; i < hTopRepeatTimes; i++) {
    //                 MicroAPI::MaskReg mask = MicroAPI::UpdateMask<T>(maskValueHTop);
    //                 MicroAPI::StoreAlign(buf + i0 * repeatStride + i * VL<T>(), value, mask);
    //             }
    //         }
    //
    //         if constexpr (hBottom) {
    //             uint32_t maskValueHButton = hButtonPaddingElements;
    //             for (uint16_t i = 0; i < hButtonRepeatTimes; i++) {
    //                 MicroAPI::MaskReg mask = MicroAPI::UpdateMask<T>(maskValueHButton);
    //                 MicroAPI::StoreAlign(buttonBuf + i0 * repeatStride + i * VL<T>(), value, mask);
    //             }
    //         }
    //     }
    // }

    // struct PadParams {
    //     uint32_t tileBufC1Stride;
    //     uint32_t bufWidth;
    //     uint32_t bufWidthC0Blocks;
    //     uint32_t hElementsInPadWBlock;
    //     uint32_t wElementsInPadHBlock;
    //     uint32_t fmapH;
    //     uint32_t fmapW;
    //     uint16_t hRepeatTimesInPadWBlock;
    //     uint16_t wRepeatTimesInPadHBlock;
    //     HWPad pad;
    // };
    //
    // template <bool hTop, bool hBottom, bool wLeft, bool wRight>
    // static __simd_vf__ inline void PadFmap(
    //     __ubuf__ T* buf,
    //     const uint16_t c1Length,
    //     const PadParams params)
    // {
    //     const uint32_t bufWidth = params.bufWidth;
    //     const uint32_t bufWidthC0Blocks = params.bufWidthC0Blocks;
    //     const uint32_t hElementsInPadWBlock = params.hElementsInPadWBlock;
    //     const uint32_t wElementsInPadHBlock = params.wElementsInPadHBlock;
    //     const uint32_t fmapH = params.fmapH;
    //     const uint32_t fmapW = params.fmapW;
    //     const uint16_t hRepeatTimesInPadWBlock = params.hRepeatTimesInPadWBlock;
    //     const uint16_t wRepeatTimesInPadHBlock = params.wRepeatTimesInPadHBlock;
    //     const HWPad& pad = params.pad;
    //
    //     MicroAPI::RegTensor<T> value;
    //     MicroAPI::Duplicate(value, static_cast<T>(0));
    //
    //     for (uint16_t c1 = 0; c1 < c1Length; c1++) {
    //         if constexpr (hTop) {
    //             for (uint16_t i = 0; i < pad.hTop; i++) {
    //                 uint32_t maskValue = wElementsInPadHBlock;
    //                 uint32_t offset = i * bufWidth;
    //                 for (uint16_t w = 0; w < wRepeatTimesInPadHBlock; w++) {
    //                     MicroAPI::MaskReg mask = MicroAPI::UpdateMask<T>(maskValue);
    //                     MicroAPI::StoreAlign(buf + offset + w * VL<T>(), value, mask);
    //                 }
    //             }
    //         }
    //
    //         if constexpr (hBottom) {
    //             for (uint16_t i = 0; i < pad.hBottom; i++) {
    //                 uint32_t maskValue = wElementsInPadHBlock;
    //                 uint32_t offset = (i + pad.hTop + fmapH) * bufWidth;
    //                 for (uint16_t w = 0; w < wRepeatTimesInPadHBlock; w++) {
    //                     MicroAPI::MaskReg mask = MicroAPI::UpdateMask<T>(maskValue);
    //                     MicroAPI::StoreAlign(buf + offset + w * VL<T>(), value, mask);
    //                 }
    //             }
    //         }
    //
    //         if constexpr (wLeft) {
    //             for (uint16_t i = 0; i < pad.wLeft; i++) {
    //                 uint32_t maskValue = hElementsInPadWBlock;
    //                 uint32_t offset = pad.hTop * bufWidth + i * C0<T>();
    //                 for (uint16_t w = 0; w < hRepeatTimesInPadWBlock; w++) {
    //                     MicroAPI::MaskReg mask = MicroAPI::UpdateMask<T>(maskValue);
    //                     MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
    //                         buf + offset + w * BLK_COUNT_IN_VL * bufWidth, value,
    //                         bufWidthC0Blocks, mask);
    //                 }
    //             }
    //         }
    //
    //         if constexpr (wRight) {
    //             for (uint16_t i = 0; i < pad.wRight; i++) {
    //                 uint32_t maskValue = hElementsInPadWBlock;
    //                 uint32_t offset = pad.hTop * bufWidth + (i + pad.wLeft + fmapW) * C0<T>();
    //                 for (uint16_t w = 0; w < hRepeatTimesInPadWBlock; w++) {
    //                     MicroAPI::MaskReg mask = MicroAPI::UpdateMask<T>(maskValue);
    //                     MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
    //                         buf + offset + w * BLK_COUNT_IN_VL * bufWidth, value,
    //                         bufWidthC0Blocks, mask);
    //                 }
    //             }
    //         }
    //
    //         buf += params.tileBufC1Stride;
    //     }
    // }


    static __simd_callee__ inline void TransformVf(
        MicroAPI::RegTensor<T>& s0, MicroAPI::RegTensor<T>& s1, MicroAPI::RegTensor<T>& s2, MicroAPI::RegTensor<T>& s3,
        MicroAPI::RegTensor<T>& d0, MicroAPI::RegTensor<T>& d1, MicroAPI::RegTensor<T>& d2, MicroAPI::RegTensor<T>& d3,
        MicroAPI::MaskReg& mask)
    {
        MicroAPI::Sub(d0, s0, s2, mask);
        MicroAPI::Add(d1, s1, s2, mask);
        MicroAPI::Sub(d2, s2, s1, mask);
        MicroAPI::Sub(d3, s1, s3, mask);
    }

    __aicore__ inline LocalTensor<T> GetPingPongBuffer()
    {
        LocalTensor<T> mainBuf = vBuf_.GetWithOffset<T>(
            singleShapeBufLength_, static_cast<uint32_t>(pingFlag_) * singleShapeBufLength_ * sizeof(T));
        pingFlag_ = !pingFlag_;
        return mainBuf;
    }

    __aicore__ inline LocalTensor<T> GetTransposeBuffer()
    {
        return vBuf_.GetWithOffset<T>(
            transposeBufCnt_ * transposeBufLength_, singleShapeBufLength_ * 2 * sizeof(T));
    }

    TBuf<TPosition::VECIN> vBuf_;
    GlobalTensor<T> fmapGm_;
    const uint32_t cin_;
    const uint32_t hi_;
    const uint32_t wi_;
    const uint16_t padH_;
    const uint16_t padW_;
    const uint32_t tilesH_;
    const uint32_t tilesW_;
    const uint16_t singleShapeCin_;
    const uint16_t singleShapeTilesH_;
    const uint16_t singleShapeTilesW_;
    const uint16_t transposeBufCnt_;
    const uint32_t transposeBufLength_;
    const uint32_t singleShapeBufLength_;
    bool pingFlag_ = true;
};

#endif //CONV_BP_WINOGRAD_H