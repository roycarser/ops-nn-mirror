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
 * \file conv_bp_wino_transform.h
 * \brief
 */

#ifndef CONV_BP_WINO_TRANSFORM_H
#define CONV_BP_WINO_TRANSFORM_H

#include "conv_bp_wino_util.h"

//Transpose5HD做转置时按照16*16为最小单位的,所以搬运时hw轴要统一按照16元素对齐
static constexpr uint32_t HW_SRC_ALIGNED_16 = 16;
//tileBuf在满足tile空间的大小下需要pad 1列让宽变成奇数,防止行列变换时跨行读列时
//一整列都在少数bank产生bank冲突
static constexpr uint8_t TILE_BUF_BANK_CONFLICT_PADDING = 1;

struct CSlice {
    uint32_t idx;
    uint32_t length;
    uint16_t c1;

    template <typename T>
    __aicore__ inline uint32_t C1Idx() const
    {
        return idx / C0<T>();
    }
};

struct TileBox {
    HWBox tile;
    HWBox src;
    HWPad pad;
    CSlice c;
};

namespace WinoTransformDetail {
constexpr inline uint32_t __aicore__ CalColUnfoldBufWidth(uint32_t th)
{
    //要补一个pad到奇数
    return TileUnfoldSize(th) | TILE_BUF_BANK_CONFLICT_PADDING;
}

constexpr inline uint32_t __aicore__ Cal16TileHWBufWidth(uint32_t tileHW)
{
    //补一个pad到奇数
    return tileHW | TILE_BUF_BANK_CONFLICT_PADDING;
}


template <typename T>
constexpr inline __aicore__ uint32_t GetTransformBufSizeC0(uint32_t tileHW)
{
    return F23_TRANSFORM_TILE_ELEMENTS_16 * Cal16TileHWBufWidth(tileHW) * C0<T>();
}

template <typename TransformConfig>
constexpr __aicore__ static inline uint32_t GetInputBufSizeC0()
{
    constexpr uint32_t STRIDE = TransformConfig::STRIDE;
    constexpr uint32_t WINDOW_SIZE = TransformConfig::WINDOW_SIZE;
    using SlideWin = SlideWindows<STRIDE, WINDOW_SIZE>;
    using TilingConfigT = typename TransformConfig::TilingT;
    using T = typename TransformConfig::T;

    constexpr uint32_t srcHW =
        SlideWin::Tiles2SrcLength(BlockConfig::SingleShapeTileH<TilingConfigT>()) *
        SlideWin::Tiles2SrcLength(BlockConfig::SingleShapeTileW<TilingConfigT>());
    return srcHW * C0<T>();
}

template <typename TransformConfig>
constexpr inline __aicore__ uint32_t GetTransformBufSize()
{
    using TilingConfigT = typename TransformConfig::TilingT;
    using T = typename TransformConfig::T;
    constexpr uint32_t tileHW = BlockConfig::SingleShapeTileH<TilingConfigT>() *
                                BlockConfig::SingleShapeTileW<TilingConfigT>();
    return GetTransformBufSizeC0<T>(tileHW) * BlockConfig::SingleTransformC1<TilingConfigT>();
}

template <typename TransformConfig>
constexpr __aicore__ static inline uint32_t GetInputBufSize()
{
    using TilingConfigT = typename TransformConfig::TilingT;
    return GetInputBufSizeC0<TransformConfig>() * BlockConfig::SingleTransformC1<TilingConfigT>();
}

template <typename TransformConfig>
constexpr __aicore__ static inline uint32_t GetTmpBufLength()
{
    constexpr uint32_t STRIDE = TransformConfig::STRIDE;
    constexpr uint32_t WINDOW_SIZE = TransformConfig::WINDOW_SIZE;
    using TilingConfigT = typename TransformConfig::TilingT;
    using T = typename TransformConfig::T;

    constexpr uint32_t tileH = BlockConfig::SingleShapeTileH<TilingConfigT>();
    constexpr uint32_t tileW = BlockConfig::SingleShapeTileW<TilingConfigT>();
    constexpr uint32_t srcW = SlideWindows<STRIDE, WINDOW_SIZE>::Tiles2SrcLength(tileW);
    return srcW * CalColUnfoldBufWidth(tileH) * C0<T>();
}
}

namespace WinoTransformDetail {
template <typename T, typename Impl>
struct UnfoldIntf {
    using UnfoldColParamsT = typename Impl::UnfoldColParamsT;
    using UnfoldRowParamsT = typename Impl::UnfoldRowParamsT;

    static __aicore__ AscendC::Std::tuple<UnfoldColParamsT, UnfoldRowParamsT> InitUnfoldParams(const TileBox& box)
    {
        return Impl::InitUnfoldParams(box);
    }

    template <bool isTailTile>
    static __simd_callee__ inline void UnfoldColsVf(
        __ubuf__ T* unfoldColBuf,
        __ubuf__ T* srcBuf,
        const UnfoldColParamsT& params)
    {
        Impl::template UnfoldColsVf<isTailTile>(unfoldColBuf, srcBuf, params);
    }

    static __simd_callee__ inline void UnfoldRowsVf(
        __ubuf__ T* outBuf,
        __ubuf__ T* srcBuf,
        const UnfoldRowParamsT& params)
    {
        Impl::UnfoldRowsVf(outBuf, srcBuf, params);
    }
};
}

template <typename Type,
    uint32_t STRIDE_VAL,
    uint32_t WINDOWS_SIZE_VAL,
    typename UnfoldImplType,
    typename TilingType>
struct TransformConfig {
    using T = Type;
    using UnfoldImpl = UnfoldImplType;
    using TilingT = TilingType;

    static constexpr uint32_t STRIDE = STRIDE_VAL;
    static constexpr uint32_t WINDOW_SIZE = WINDOWS_SIZE_VAL;
};


template <typename Config>
class WinoTransformer {
public:
    static constexpr uint32_t STRIDE = Config::STRIDE;
    static constexpr uint32_t WINDOW_SIZE = Config::WINDOW_SIZE;
    using T = typename Config::T;
    using TilingConfigT = typename Config::TilingT;
    using SlideWin = SlideWindows<STRIDE, WINDOW_SIZE>;
    using UnfoldPolicy = WinoTransformDetail::UnfoldIntf<T, typename Config::UnfoldImpl>;


    __aicore__ inline WinoTransformer(
        __gm__ T* in5HD,
        const uint32_t srcH,
        const uint32_t srcW,
        const uint32_t srcC,
        const uint16_t padH,
        const uint16_t padW)
        : srcH_(srcH),
          srcW_(srcW),
          srcC_(srcC),
          padH_(padH),
          padW_(padW)
    {
        gm_.SetGlobalBuffer(in5HD);
    }


    __aicore__ inline TileBox CalculateSrcBox(const HWBox& tile, uint32_t cIdx, uint32_t cLength) const
    {
        TileBox box = {tile, {}, {}, {}};
        SlideWin::CalculateSrcBox(
            box.tile, srcH_, srcW_, padH_, padW_,
            box.src, box.pad);
        box.c.idx = cIdx;
        box.c.length = cLength;
        box.c.c1 = Ops::Base::CeilDiv(cLength, C0<T>());
        return box;
    }

    __aicore__ inline uint32_t SrcH() const
    {
        return srcH_;
    }

    __aicore__ inline uint32_t SrcW() const
    {
        return srcW_;
    }

    __aicore__ inline uint32_t SrcC() const
    {
        return srcC_;
    }

    __aicore__ inline void CopyIn(
        const AscendC::LocalTensor<T>& srcBuf,
        const TileBox& box,
        const uint32_t batchIdx) const
    {
        const HWBox& src = box.src;
        if (unlikely(src.elements==0)) {
            return;
        }

        uint32_t srcC1 = Ops::Base::CeilDiv(srcC_, C0<T>());
        uint64_t srcWC0 = srcW_ * C0<T>();
        uint64_t srcHWC0 = srcH_ * srcWC0;
        //nc1hwc0搬入
        uint64_t gmOffset =
            static_cast<uint64_t>(batchIdx) * srcC1 * srcHWC0
            + static_cast<uint64_t>(box.c.C1Idx<T>()) * srcHWC0
            + static_cast<uint64_t>(src.hIdx) * srcWC0
            + static_cast<uint64_t>(src.wIdx) * C0<T>();

        uint32_t srcFullLenH = SlideWin::Tiles2SrcLength(box.tile.hLength);
        uint32_t srcFullLenW = SlideWin::Tiles2SrcLength(box.tile.wLength);
        AscendC::LoopModeParams loop;
        loop.loop1Size = box.c.c1;
        loop.loop1SrcStride = srcHWC0 * sizeof(T);
        loop.loop1DstStride = srcFullLenH * srcFullLenW * C0<T>() * sizeof(T);
        loop.loop2Size = 1;
        loop.loop2SrcStride = 0;
        loop.loop2DstStride = 0;

        if constexpr (BlockConfig::SingleTransformC1<TilingConfigT>() > 1) {
            if (box.c.c1 > 1) {
                SetLoopModePara(loop, AscendC::DataCopyMVType::OUT_TO_UB);
            }
        }
        AscendC::DataCopyParams params;
        params.blockCount = src.hLength;
        params.blockLen = src.wLength;
        params.srcGap = srcW_ - src.wLength;
        params.dstGap = srcFullLenW - src.wLength;
        //留出位置给pad补0
        uint32_t hPadOffset = (box.pad.hTop * srcFullLenW + box.pad.wLeft) * C0<T>();
        AscendC::DataCopy(srcBuf[hPadOffset], gm_[gmOffset], params);

        if constexpr (BlockConfig::SingleTransformC1<TilingConfigT>() > 1) {
            if (box.c.c1 > 1) {
                AscendC::ResetLoopModePara(AscendC::DataCopyMVType::OUT_TO_UB);
            }
        }
    }


    __aicore__ inline void Compute(
        AscendC::LocalTensor<T>& srcBuf,
        AscendC::LocalTensor<T>& outBuf,
        AscendC::LocalTensor<T>& tmpBuf,
        const TileBox& box) const
    {
        constexpr uint32_t srcBufSizeC0 = WinoTransformDetail::GetInputBufSizeC0<Config>();
        uint32_t outBufSizeC0 = WinoTransformDetail::GetTransformBufSizeC0<T>(box.tile.elements);
        const HWBox& src = box.src;

        if (unlikely(src.elements==0)) {
            //整个tile都由padding区域产生,不做计算直接置0,
            AscendC::Duplicate(outBuf, static_cast<T>(0), outBufSizeC0 * box.c.c1);
            return;
        }

        const auto params = UnfoldPolicy::InitUnfoldParams(box);
        const typename UnfoldPolicy::UnfoldColParamsT& ucp = AscendC::Std::get<0>(params);
        const typename UnfoldPolicy::UnfoldRowParamsT& urp = AscendC::Std::get<1>(params);

        __ubuf__ T* tmpBufAddr = reinterpret_cast<__ubuf__ T*>(tmpBuf.GetPhyAddr());
        __ubuf__ T* srcBufAddr = reinterpret_cast<__ubuf__ T*>(srcBuf.GetPhyAddr());
        __ubuf__ T* outBufAddr = reinterpret_cast<__ubuf__ T*>(outBuf.GetPhyAddr());

        const bool isTail = box.tile.wLength < BlockConfig::SingleShapeTileW<TilingConfigT>() ||
                            box.tile.hLength < BlockConfig::SingleShapeTileH<TilingConfigT>();

        if constexpr (BlockConfig::SingleTransformC1<TilingConfigT>() == 1) {
            //TODO 当前需要优化的点主要集中在列变换，列变换是不是可以不管尾块统一按标准块处理？
            if (isTail) {
                UnfoldVf<true>(
                    outBufAddr, tmpBufAddr, srcBufAddr,
                    ucp, urp,
                    box.pad,
                    src.hLength, src.wLength);
            } else {
                UnfoldVf<false>(
                    outBufAddr, tmpBufAddr, srcBufAddr,
                    ucp, urp,
                    box.pad,
                    src.hLength, src.wLength);
            }
        } else {
            for (uint16_t c1Idx = 0; c1Idx < box.c.c1; c1Idx++) {
                if (isTail) {
                    UnfoldVf<true>(
                        outBufAddr, tmpBufAddr, srcBufAddr,
                        ucp, urp,
                        box.pad,
                        src.hLength, src.wLength);
                } else {
                    UnfoldVf<false>(
                        outBufAddr, tmpBufAddr, srcBufAddr,
                        ucp, urp,
                        box.pad,
                        src.hLength, src.wLength);
                }
                outBufAddr += outBufSizeC0;
                srcBufAddr += srcBufSizeC0;
            }
        }
    }

    __aicore__ inline void SetNK1C1K0C0CopyParams(
        NK1C1K0C0::CopyK0Params& copyParams,
        const TileBox& box) const
    {
        copyParams.tiles = box.tile.elements;
        copyParams.srcBufWidthBlockStride = WinoTransformDetail::Cal16TileHWBufWidth(box.tile.elements);
        copyParams.c1Idx = box.c.C1Idx<T>();
        copyParams.c1Length = box.c.c1;
    }

private:

    template <bool IsTailTile>
    __simd_vf__ static inline void UnfoldVf(
        __ubuf__ T* outBuf,
        __ubuf__ T* colUnfoldBuf,
        __ubuf__ T* srcBuf,
        const typename UnfoldPolicy::UnfoldColParamsT ucp,
        const typename UnfoldPolicy::UnfoldRowParamsT urp,
        HWPad pad, uint16_t srcH, uint16_t srcW)
    {
        Padding(srcBuf, pad, srcH, srcW);

        AscendC::MicroAPI::LocalMemBar<
            AscendC::MicroAPI::MemType::VEC_STORE,
            AscendC::MicroAPI::MemType::VEC_LOAD>();

        UnfoldPolicy::template UnfoldColsVf<IsTailTile>(colUnfoldBuf, srcBuf, ucp);

        AscendC::MicroAPI::LocalMemBar<
            AscendC::MicroAPI::MemType::VEC_STORE,
            AscendC::MicroAPI::MemType::VEC_LOAD>();

        UnfoldPolicy::UnfoldRowsVf(outBuf, colUnfoldBuf, urp);
    }

    __simd_callee__ static inline void Padding(
        __ubuf__ T* srcBuf, const HWPad& pad, uint16_t srcH, uint16_t srcW)
    {
        using namespace MicroAPI;
        RegTensor<T> paddingValue;
        Duplicate(paddingValue, 0);

        const uint16_t padHTop = pad.hTop;
        const uint16_t padHButton = pad.hBottom;
        const uint16_t padWLeft = pad.wLeft;
        const uint16_t padWRight = pad.wRight;

        const uint16_t wBlocks = srcW + padWLeft + padWRight;
        const uint32_t wElements = wBlocks * C0<T>();

        const uint32_t hTopElements = wElements * padHTop;
        const uint16_t hTopRepeatTimes = CeilDivision(hTopElements, VL<T>());

        __ubuf__ T* src = srcBuf;
        uint32_t hTopMaskValue = hTopElements;
        for (uint16_t i = 0; i < hTopRepeatTimes; i++) {
            MaskReg mask = MicroAPI::UpdateMask<T>(hTopMaskValue);
            StoreAlign<T, PostLiteral::POST_MODE_UPDATE>(src, paddingValue, VL<T>(), mask);
        }

        const uint32_t hBtnElements = wElements * padHButton;
        const uint16_t hBtnRepeatTimes = CeilDivision(hBtnElements, VL<T>());

        src = srcBuf + (padHTop + srcH) * wElements;
        uint32_t hBtnMaskValue = hBtnElements;
        for (uint16_t i = 0; i < hBtnRepeatTimes; i++) {
            MaskReg mask = MicroAPI::UpdateMask<T>(hBtnMaskValue);
            StoreAlign<T, PostLiteral::POST_MODE_UPDATE>(src, paddingValue, VL<T>(), mask);
        }

        const uint32_t hElements = srcH * C0<T>();
        const uint16_t hRepeatTimes = CeilDivision(hElements, VL<T>());
        const uint16_t wPadStride = (VL<T>() / C0<T>()) * wBlocks;

        src = srcBuf + padHTop * wElements;
        for (uint16_t i = 0; i < padWLeft; i++) {
            uint32_t maskValue = hElements;
            __ubuf__ T* src0 = src + C0<T>() * i;
            for (uint16_t h = 0; h < hRepeatTimes; h++) {
                MaskReg mask = MicroAPI::UpdateMask<T>(maskValue);
                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    src0, paddingValue, wBlocks, wPadStride, mask);
            }
        }

        src = srcBuf + padHTop * wElements + (srcW + padWLeft) * C0<T>();
        for (uint16_t i = 0; i < padWRight; i++) {
            uint32_t maskValue = hElements;
            __ubuf__ T* src0 = src + C0<T>() * i;
            for (uint16_t h = 0; h < hRepeatTimes; h++) {
                MaskReg mask = MicroAPI::UpdateMask<T>(maskValue);
                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    src0, paddingValue, wBlocks, wPadStride, mask);
            }
        }
    }


    AscendC::GlobalTensor<T> gm_;
    const uint32_t srcH_;
    const uint32_t srcW_;
    const uint32_t srcC_;
    const uint16_t padH_;
    const uint16_t padW_;
};

namespace WinoTransformDetail {
constexpr uint32_t F23_FMAP_STRIDE = 2;
constexpr uint32_t F23_FMAP_WINDOWS = 4;
constexpr uint32_t F23_DY_STRIDE = 2;
constexpr uint32_t F23_DY_WINDOWS = 2;


using namespace AscendC::MicroAPI;

struct DefaultUnfoldColParams {
    uint32_t wValidElements;
    uint32_t tileBufWidthBlocks;
    uint16_t wRepeatTimes;
    uint16_t tileH;
};

struct DefaultUnfoldRowParams {
    uint32_t hValidElements;
    uint32_t srcTileBufWidth;
    uint32_t dstTileBufWidthBlocks;
    uint16_t hRepeatTimes;
    uint16_t tileW;
};

template <typename T, uint32_t F23_STRIDE, uint32_t F23_WINDOW>
static inline __aicore__ void InitDefaultUnfoldParams(
    const TileBox& box,
    DefaultUnfoldColParams& ucp,
    DefaultUnfoldRowParams& urp)
{
    ucp.wValidElements = SlideWindows<F23_STRIDE, F23_WINDOW>::Tiles2SrcLength(box.tile.wLength) * C0<T>();
    ucp.tileBufWidthBlocks = CalColUnfoldBufWidth(box.tile.hLength);
    ucp.wRepeatTimes = Ops::Base::CeilDiv(ucp.wValidElements, VL<T>());
    ucp.tileH = box.tile.hLength;

    urp.hValidElements = TileUnfoldSize(box.tile.hLength) * C0<T>();
    urp.srcTileBufWidth = ucp.tileBufWidthBlocks * C0<T>();
    urp.dstTileBufWidthBlocks = Cal16TileHWBufWidth(box.tile.elements);
    urp.hRepeatTimes = Ops::Base::CeilDiv(urp.hValidElements, VL<T>());
    urp.tileW = box.tile.wLength;
}


struct Unfold16TileHWStorer {
    template <typename T>
    struct StoreInfo {
        __ubuf__ T* dst0;
        __ubuf__ T* dst1;
        __ubuf__ T* dst2;
        __ubuf__ T* dst3;
        __ubuf__ T* dst4;
        __ubuf__ T* dst5;
        __ubuf__ T* dst6;
        __ubuf__ T* dst7;

        uint16_t dstTileBufWidthBlocks;

        MaskReg maskAll;
        MaskReg lowHalfPartMask;
        MaskReg highHalfPartMask;
    };


    template <typename T>
    static __simd_callee__ inline void CreateStoreInfo(
        StoreInfo<T>& p,
        __ubuf__ T* out,
        uint16_t tileW,
        uint16_t dstTileBufWidthBlocks)
    {
        uint32_t dstStride = C0<T>() * F23_TRANSFORM_TILE_SIZE_4 * dstTileBufWidthBlocks;

        p.dstTileBufWidthBlocks = dstTileBufWidthBlocks;
        p.dst0 = out;
        p.dst1 = p.dst0 + dstStride;
        p.dst2 = p.dst1 + dstStride;
        p.dst3 = p.dst2 + dstStride;

        //这些地址用于reg的后半部分写入，需要减掉前半部分的地址偏移
        //TODO 测试地址减掉后越界
        p.dst4 = p.dst0 + tileW * C0<T>() - C0<T>() * 4 * dstTileBufWidthBlocks;
        p.dst5 = p.dst4 + dstStride;
        p.dst6 = p.dst5 + dstStride;
        p.dst7 = p.dst6 + dstStride;

        p.lowHalfPartMask = CreateMask<T, MaskPattern::H>();
        p.maskAll = CreateMask<T, MaskPattern::ALL>();
        Not(p.highHalfPartMask, p.lowHalfPartMask, p.maskAll);
    }

    template <typename T>
    static __simd_callee__ inline void UpdateStoreInfo(StoreInfo<T>& p, uint16_t tileW)
    {
        uint32_t step = C0<T>() * tileW;
        p.dst0 += step;
        p.dst1 += step;
        p.dst2 += step;
        p.dst3 += step;
        p.dst4 += step;
        p.dst5 += step;
        p.dst6 += step;
        p.dst7 += step;
    }

    template <typename T>
    static __simd_callee__ inline void GetHighHalfPartMask(
        StoreInfo<T>& p,
        MaskReg& highHalfPartMask,
        MaskReg& tileHMask)
    {
        And(highHalfPartMask, tileHMask, p.highHalfPartMask, p.maskAll);
    }

    template <typename T>
    static __simd_callee__ inline void store(
        StoreInfo<T>& p,
        RegTensor<T>& r0,
        RegTensor<T>& r1,
        RegTensor<T>& r2,
        RegTensor<T>& r3,
        MaskReg& highHalfPartMask)
    {
        //TODO 尝试先gather在select完成block级别的交织

        StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
            p.dst0, r0, p.dstTileBufWidthBlocks, 1, p.lowHalfPartMask);
        StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
            p.dst1, r1, p.dstTileBufWidthBlocks, 1, p.lowHalfPartMask);

        StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
            p.dst2, r2, p.dstTileBufWidthBlocks, 1, p.lowHalfPartMask);
        StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
            p.dst3, r3, p.dstTileBufWidthBlocks, 1, p.lowHalfPartMask);

        StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
            p.dst4, r0, p.dstTileBufWidthBlocks, 1, highHalfPartMask);
        StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
            p.dst5, r1, p.dstTileBufWidthBlocks, 1, highHalfPartMask);

        StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
            p.dst6, r2, p.dstTileBufWidthBlocks, 1, highHalfPartMask);
        StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
            p.dst7, r3, p.dstTileBufWidthBlocks, 1, highHalfPartMask);
    }
};


template <typename T, typename TilingT>
struct Dy {
    using UnfoldRowParamsT = DefaultUnfoldRowParams;
    using UnfoldColParamsT = DefaultUnfoldColParams;

    using ParamsTuple = AscendC::Std::tuple<UnfoldColParamsT, UnfoldRowParamsT>;

    static ParamsTuple inline __aicore__ InitUnfoldParams(const TileBox& box)
    {
        DefaultUnfoldRowParams urp = {};
        DefaultUnfoldColParams ucp = {};

        InitDefaultUnfoldParams<T, F23_DY_STRIDE, F23_DY_WINDOWS>(box, ucp, urp);

        return AscendC::Std::make_tuple(ucp, urp);
    }

    template <bool isTailTile>
    static __simd_callee__ inline void UnfoldColsVf(
        __ubuf__ T* tileBuf,
        __ubuf__ T* dyBuf,
        const DefaultUnfoldColParams& params)
    {
        constexpr uint32_t TileH = BlockConfig::SingleShapeTileH<TilingT>();
        constexpr uint32_t TileW = BlockConfig::SingleShapeTileW<TilingT>();

        using sw = SlideWindows<F23_DY_STRIDE, F23_DY_WINDOWS>;
        constexpr uint32_t SrcW = sw::Tiles2SrcLength(TileW);

        //大轴在内，小轴在外，scalar都在第一层循环，所以外循环越小scalar占比越低
        constexpr uint32_t LoopH = TileH;
        constexpr uint32_t LoopW = ConstexprMaths::CeilDiv(SrcW * C0<T>(), VL<T>());

        if constexpr (LoopH > LoopW) {
            UnfoldColsHFirstVf(tileBuf, dyBuf, params);
        } else {
            UnfoldColsWFirstVf(tileBuf, dyBuf, params);
        }
    }


    static __simd_callee__ inline void UnfoldColsWFirstVf(
        __ubuf__ T* tileBuf,
        __ubuf__ T* dyBuf,
        const DefaultUnfoldColParams& params)
    {
        const uint32_t wValidElements = params.wValidElements;
        const uint32_t tileBufWidthBlocks = params.tileBufWidthBlocks;
        const uint16_t wRepeatTimes = params.wRepeatTimes;
        const uint16_t tileH = params.tileH;
        const uint16_t dstStride = tileBufWidthBlocks * (VL<T>() / C0<T>()) - F23_TRANSFORM_TILE_SIZE_4 + 1;

        RegTensor<bfloat16_t> bf16NegativeOne;
        if constexpr (Std::is_same_v<T,bfloat16_t>) {
            Duplicate(bf16NegativeOne, -1);
        }

        for (uint16_t th = 0; th < tileH; th++) {
            constexpr uint32_t thStride = F23_TRANSFORM_TILE_SIZE_4 * C0<T>();

            __ubuf__ T* dst = tileBuf + thStride * th;
            __ubuf__ T* src0 = dyBuf + wValidElements * th * F23_DY_STRIDE;
            __ubuf__ T* src1 = src0 + wValidElements;

            uint32_t maskValue = wValidElements;
            for (uint16_t i = 0; i < wRepeatTimes; i++) {
                MaskReg mask = UpdateMask<T>(maskValue);

                RegTensor<T> s0;
                RegTensor<T> s1;
                LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(s0, src0, VL<T>());
                LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(s1, src1, VL<T>());

                RegTensor<T> d0;
                RegTensor<T> d1;
                RegTensor<T> d2;
                TransformVf(bf16NegativeOne,s0, s1, d0, d1, d2, mask);

                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    dst, s0, tileBufWidthBlocks, 1, mask);
                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    dst, d0, tileBufWidthBlocks, 1, mask);
                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    dst, d1, tileBufWidthBlocks, 1, mask);
                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    dst, d2, tileBufWidthBlocks, dstStride, mask);
            }
        }
    }

    static __simd_callee__ inline void UnfoldColsHFirstVf(
        __ubuf__ T* tileBuf,
        __ubuf__ T* dyBuf,
        const DefaultUnfoldColParams& params)
    {
        const uint32_t wValidElements = params.wValidElements;
        const uint32_t tileBufWidthBlocks = params.tileBufWidthBlocks;
        const uint16_t wRepeatTimes = params.wRepeatTimes;
        const uint16_t tileH = params.tileH;

        RegTensor<bfloat16_t> bf16NegativeOne;
        if constexpr (Std::is_same_v<T, bfloat16_t>) {
            Duplicate(bf16NegativeOne, -1);
        }

        uint32_t maskValue = wValidElements;
        for (uint16_t w = 0; w < wRepeatTimes; w++) {
            MaskReg mask = UpdateMask<T>(maskValue);

            __ubuf__ T* src = dyBuf + w * VL<T>();
            __ubuf__ T* dst = tileBuf + tileBufWidthBlocks * VL<T>() * w;

            for (uint16_t i = 0; i < tileH; i++) {
                RegTensor<T> s0, s1, d0, d1, d2;

                LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(s0, src, wValidElements);
                LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(s1, src, wValidElements);

                TransformVf(bf16NegativeOne, s0, s1, d0, d1, d2, mask);

                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    dst, s0, tileBufWidthBlocks, 1, mask);
                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    dst, d0, tileBufWidthBlocks, 1, mask);
                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    dst, d1, tileBufWidthBlocks, 1, mask);
                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    dst, d2, tileBufWidthBlocks, 1, mask);
            }
        }
    }

    static __simd_callee__ inline void UnfoldRowsVf(
        __ubuf__ T* out,
        __ubuf__ T* buf,
        const DefaultUnfoldRowParams& params)
    {
        const uint32_t srcTileBufWidth = params.srcTileBufWidth;
        const uint16_t dstTileBufWidthBlocks = params.dstTileBufWidthBlocks;
        const uint16_t hRepeatTimes = params.hRepeatTimes;
        const uint16_t tileW = params.tileW;

        RegTensor<bfloat16_t> bf16NegativeOne;
        if constexpr (Std::is_same_v<T, bfloat16_t>) {
            Duplicate(bf16NegativeOne, -1);
        }

        Unfold16TileHWStorer::StoreInfo<T> s;
        Unfold16TileHWStorer::CreateStoreInfo(s, out, tileW, dstTileBufWidthBlocks);

        uint32_t maskValue = params.hValidElements;
        for (uint16_t i = 0; i < hRepeatTimes; i++) {
            __ubuf__ T* src = buf + i * VL<T>();
            MaskReg mask = UpdateMask<T>(maskValue);
            MaskReg storeMask;
            Unfold16TileHWStorer::GetHighHalfPartMask(s, storeMask, mask);

            for (uint16_t th = 0; th < tileW; th++) {
                RegTensor<T> s0;
                RegTensor<T> s1;

                LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(s0, src, srcTileBufWidth);
                LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(s1, src, srcTileBufWidth);

                RegTensor<T> d0;
                RegTensor<T> d1;
                RegTensor<T> d2;
                TransformVf(bf16NegativeOne, s0, s1, d0, d1, d2, mask);

                Unfold16TileHWStorer::store(s, s0, d0, d1, d2, storeMask);
            }
            Unfold16TileHWStorer::UpdateStoreInfo(s, tileW);
        }
    }

    static __simd_callee__ inline void TransformVf(
        RegTensor<bfloat16_t>& negativeOne,
        RegTensor<T>& s0, RegTensor<T>& s1,
        RegTensor<T>& d0, RegTensor<T>& d1, RegTensor<T>& d2,
        MaskReg& mask)
    {
        Add(d0, s0, s1, mask);
        Sub(d1, s0, s1, mask);
        if constexpr (Std::is_same_v<bfloat16_t, T>) {
            //bf16不支持Neg指令，用乘-1替代
            Mul(d2, s1, negativeOne, mask);
        } else {
            Neg(d2, s1, mask);
        }

    }
};

template <typename T, typename TilingT>
struct Fmap {
    //fmap一个循环里展开2个tile,所以额外添加首位轮参数
    struct UnfoldFmapRowParams : DefaultUnfoldRowParams {
        uint16_t tileWMainRepeatTimes;
        uint16_t tileWTailRepeatTimes;
    };

    struct UnfoldFmapColParams : DefaultUnfoldColParams {
        uint16_t tileHMainRepeatTimes;
        uint16_t tileHTailRepeatTimes;
    };

    using UnfoldRowParamsT = UnfoldFmapRowParams;
    using UnfoldColParamsT = UnfoldFmapColParams;

    using ParamsTuple = AscendC::Std::tuple<UnfoldColParamsT, UnfoldRowParamsT>;

    static ParamsTuple inline __aicore__ InitUnfoldParams(const TileBox& box)
    {
        UnfoldFmapColParams ucp = {};
        UnfoldFmapRowParams urp = {};

        InitDefaultUnfoldParams<T, F23_FMAP_STRIDE, F23_FMAP_WINDOWS>(box, ucp, urp);

        ucp.tileHMainRepeatTimes = ucp.tileH >> 1;
        ucp.tileHTailRepeatTimes = ucp.tileH & 1;

        urp.tileWMainRepeatTimes = urp.tileW >> 1;
        urp.tileWTailRepeatTimes = urp.tileW & 1;

        return AscendC::Std::make_tuple(ucp, urp);
    }

    template <bool IsTailTile>
    static __simd_callee__ inline void UnfoldColsVf(
        __ubuf__ T* tileBuf,
        __ubuf__ T* fmapBuf,
        const UnfoldFmapColParams& params)
    {
        if constexpr (IsTailTile) {
            UnfoldColsDefaultVf(tileBuf, fmapBuf, params);
        } else {
            constexpr uint32_t TileH = BlockConfig::SingleShapeTileH<TilingT>();
            constexpr uint32_t TileW = BlockConfig::SingleShapeTileW<TilingT>();
            if constexpr (TileH <= 4) {
                //fmap的正变换为了复用Tile间的重叠数据，是W循环在外，H循环在内
                //如果H太小则W循环带来的scalar占比会比较大，所以TileH较小的情况下将TileH进行展开
                UnfoldColsUnRollTileHVf<TileH, TileW>(tileBuf, fmapBuf);
            } else {
                UnfoldColsDefaultVf(tileBuf, fmapBuf, params);
            }
        }
    }

    template <uint16_t TileH, uint16_t SrcH, uint32_t WElements, uint32_t TileBufWidthBlocks, uint32_t DstStride>
    struct TileHUnRollHelper {
        __ubuf__ T* src[SrcH];
        __ubuf__ T* dst[TileH];
        RegTensor<T> s[SrcH];
        RegTensor<T> d[TileH * F23_TRANSFORM_TILE_SIZE_4];

        template <uint16_t Idx>
        static __simd_callee__ inline void InitSrcAddrImpl(TileHUnRollHelper& helper,__ubuf__ T* fmapBuf)
        {
            helper.src[Idx] = fmapBuf + WElements * Idx;
        }

        template <uint16_t N = SrcH>
        static __simd_callee__ inline void InitSrcAddr(TileHUnRollHelper& helper,__ubuf__ T* fmapBuf)
        {
            if constexpr (N > 0) {
                // 🌟 技巧 2：先调 N-1 的递归，再执行 N-1 的动作。
                // 这样就能保证是从 0, 1, 2... 顺序执行的！
                InitSrcAddr<N - 1>(helper, fmapBuf);
                InitSrcAddrImpl<N - 1>(helper, fmapBuf);
            }
        }

        template <uint16_t Idx>
        static __simd_callee__ inline void InitDstAddrImpl(TileHUnRollHelper& helper,__ubuf__ T* tileBuf)
        {
            helper.dst[Idx] = tileBuf + F23_TRANSFORM_TILE_SIZE_4 * C0<T>() * Idx;
        }

        template <uint16_t N = TileH>
        static __simd_callee__ inline void InitDstAddr(TileHUnRollHelper& helper,__ubuf__ T* tileBuf)
        {
            if constexpr (N > 0) {
                InitDstAddr<N - 1>(helper, tileBuf);
                InitDstAddrImpl<N - 1>(helper, tileBuf);
            }
        }

        template <uint16_t Idx>
        static __simd_callee__ inline void LoadSrcImpl(TileHUnRollHelper& helper)
        {
            LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(helper.s[Idx], helper.src[Idx], VL<T>());
        }

        template <uint16_t N = SrcH>
        static __simd_callee__ inline void LoadHSrc(TileHUnRollHelper& helper)
        {
            if constexpr (N > 0) {
                LoadHSrc<N - 1>(helper);
                LoadSrcImpl<N - 1>(helper);
            }
        }

        template <uint16_t Idx>
        static __simd_callee__ inline void TransformAndStoreImpl(
            TileHUnRollHelper& helper, MaskReg& mask)
        {
            constexpr uint16_t sIdx = Idx * F23_FMAP_STRIDE;
            constexpr uint16_t dIdx = Idx * F23_TRANSFORM_TILE_SIZE_4;
            RegTensor<T>& s0 = helper.s[sIdx];
            RegTensor<T>& s1 = helper.s[sIdx + 1];
            RegTensor<T>& s2 = helper.s[sIdx + 2];
            RegTensor<T>& s3 = helper.s[sIdx + 3];

            RegTensor<T>& d0 = helper.d[dIdx];
            RegTensor<T>& d1 = helper.d[dIdx + 1];
            RegTensor<T>& d2 = helper.d[dIdx + 2];
            RegTensor<T>& d3 = helper.d[dIdx + 3];

            TransformVf(s0, s1, s2, s3, d0, d1, d2, d3, mask);

            StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                helper.dst[Idx], d0, TileBufWidthBlocks, 1, mask);
            StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                helper.dst[Idx], d1, TileBufWidthBlocks, 1, mask);
            StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                helper.dst[Idx], d2, TileBufWidthBlocks, 1, mask);
            StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                helper.dst[Idx], d3, TileBufWidthBlocks, DstStride, mask);
        }

        template <uint16_t N = TileH>
        static __simd_callee__ inline void TransformAndStore(TileHUnRollHelper& helper, MaskReg& mask)
        {
            if constexpr (N > 0) {
                TransformAndStore<N - 1>(helper, mask);
                TransformAndStoreImpl<N - 1>(helper, mask);
            }
        }
    };

    template <uint16_t TileH, uint16_t TileW>
    static __simd_callee__ inline void UnfoldColsUnRollTileHVf(
        __ubuf__ T* tileBuf,
        __ubuf__ T* fmapBuf)
    {
        using SlideWin = SlideWindows<F23_FMAP_STRIDE, F23_FMAP_WINDOWS>;
        constexpr uint16_t SrcH = SlideWin::Tiles2SrcLength(TileH);
        constexpr uint32_t WElements = SlideWin::Tiles2SrcLength(TileW) * C0<T>();
        constexpr uint16_t WRepeatTimes = ConstexprMaths::CeilDiv(WElements, VL<T>());
        constexpr uint16_t tileBufWidthBlocks = CalColUnfoldBufWidth(TileH);
        constexpr uint32_t DstStride = tileBufWidthBlocks * (VL<T>() / C0<T>()) - F23_TRANSFORM_TILE_SIZE_4 + 1;

        //vf里面直接用pragma roll发现会有编译失败，不展开循环似乎没法用数组，所以手动用模板把代码展开
        using HelperT = TileHUnRollHelper<TileH, SrcH, WElements, tileBufWidthBlocks, DstStride>;
        HelperT helper;

        HelperT::InitSrcAddr(helper, fmapBuf);
        HelperT::InitDstAddr(helper, tileBuf);

        uint32_t maskValue = WElements;
        for (uint16_t i = 0; i < WRepeatTimes; i++) {
            MaskReg mask = UpdateMask<T>(maskValue);
            HelperT::LoadHSrc(helper);
            HelperT::TransformAndStore(helper, mask);
        }
    }

    static __simd_callee__ inline void UnfoldColsDefaultVf(
        __ubuf__ T* tileBuf,
        __ubuf__ T* fmapBuf,
        const UnfoldFmapColParams& params)
    {
        const uint32_t wValidElements = params.wValidElements;
        const uint32_t tileBufWidthBlocks = params.tileBufWidthBlocks;
        const uint16_t wRepeatTimes = params.wRepeatTimes;
        const uint16_t tileHMainRepeatTimes = params.tileHMainRepeatTimes;
        const uint16_t tileHTailRepeatTimes = params.tileHTailRepeatTimes;

        uint32_t maskValue = wValidElements;
        for (uint16_t i = 0; i < wRepeatTimes; i++) {
            MaskReg mask = UpdateMask<T>(maskValue);
            RegTensor<T> s0;
            RegTensor<T> s1;
            RegTensor<T> s2;
            RegTensor<T> s3;

            RegTensor<T> d0;
            RegTensor<T> d1;
            RegTensor<T> d2;
            RegTensor<T> d3;

            // 从最上方的tile开始滑窗
            // 先读取fmap首2行,每次循环往下读2行凑成4行执行变换
            // 但若一个滑窗在fmap的1-4行分别读入s0,s1,s2,s3
            // 在下一个滑窗s2,s3就变成1-2行,不考虑重新读取的话2-3行就只能读入s0,s1,滑窗1-4行就变成s2,s3,s0,s1
            // 如果将s0,s1的数据拷贝到s2,s3可能会产生多余的mov指令
            // 所以这里按照最朴素的方式展开循环一个循环内处理2个连续滑窗,
            // 如果滑窗为奇数,则通过tileHTailRepeatTimes额外执行一次滑窗

            //循环fmapW
            const uint32_t wOffset = i * VL<T>();

            __ubuf__ T* src = fmapBuf + wOffset;

            LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(s0, src, wValidElements);
            LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(s1, src, wValidElements);

            __ubuf__ T* dst = tileBuf + tileBufWidthBlocks * wOffset;

            for (uint16_t th = 0; th < tileHMainRepeatTimes; th++) {
                LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(s2, src, wValidElements);
                LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(s3, src, wValidElements);

                TransformVf(s0, s1, s2, s3, d0, d1, d2, d3, mask);

                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    dst, d0, tileBufWidthBlocks, 1, mask);
                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    dst, d1, tileBufWidthBlocks, 1, mask);
                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    dst, d2, tileBufWidthBlocks, 1, mask);
                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    dst, d3, tileBufWidthBlocks, 1, mask);

                LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(s0, src, wValidElements);
                LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(s1, src, wValidElements);

                TransformVf(s2, s3, s0, s1, d0, d1, d2, d3, mask);

                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    dst, d0, tileBufWidthBlocks, 1, mask);
                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    dst, d1, tileBufWidthBlocks, 1, mask);
                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    dst, d2, tileBufWidthBlocks, 1, mask);
                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    dst, d3, tileBufWidthBlocks, 1, mask);
            }

            for (uint16_t th = 0; th < tileHTailRepeatTimes; th++) {
                LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(s2, src, wValidElements);
                LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(s3, src, wValidElements);
                TransformVf(s0, s1, s2, s3, d0, d1, d2, d3, mask);

                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    dst, d0, tileBufWidthBlocks, 1, mask);
                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    dst, d1, tileBufWidthBlocks, 1, mask);
                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    dst, d2, tileBufWidthBlocks, 1, mask);
                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    dst, d3, tileBufWidthBlocks, 1, mask);
            }
        }
    }

    static __simd_callee__ inline void UnfoldRowsVf(
        __ubuf__ T* out,
        __ubuf__ T* buf,
        const UnfoldFmapRowParams& params)
    {
        const uint32_t srcTileBufWidth = params.srcTileBufWidth;
        const uint16_t dstTileBufWidthBlocks = params.dstTileBufWidthBlocks;
        const uint16_t hRepeatTimes = params.hRepeatTimes;
        const uint16_t tileWMainRepeatTimes = params.tileWMainRepeatTimes;
        const uint16_t tileWTailRepeatTimes = params.tileWTailRepeatTimes;
        const uint16_t tileW = params.tileW;

        Unfold16TileHWStorer::StoreInfo<T> s;
        Unfold16TileHWStorer::CreateStoreInfo(s, out, tileW, dstTileBufWidthBlocks);

        uint32_t maskValue = params.hValidElements;
        for (uint16_t i = 0; i < hRepeatTimes; i++) {
            MaskReg mask = UpdateMask<T>(maskValue);
            MaskReg storeMask;
            Unfold16TileHWStorer::GetHighHalfPartMask(s, storeMask, mask);

            RegTensor<T> s0;
            RegTensor<T> s1;
            RegTensor<T> s2;
            RegTensor<T> s3;

            RegTensor<T> d0;
            RegTensor<T> d1;
            RegTensor<T> d2;
            RegTensor<T> d3;

            __ubuf__ T* src = buf + VL<T>() * i;

            LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(s0, src, srcTileBufWidth);
            LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(s1, src, srcTileBufWidth);

            for (uint16_t tw = 0; tw < tileWMainRepeatTimes; tw++) {
                LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(s2, src, srcTileBufWidth);
                LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(s3, src, srcTileBufWidth);

                TransformVf(s0, s1, s2, s3, d0, d1, d2, d3, mask);

                Unfold16TileHWStorer::store(s, d0, d1, d2, d3, storeMask);

                LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(s0, src, srcTileBufWidth);
                LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(s1, src, srcTileBufWidth);

                TransformVf(s2, s3, s0, s1, d0, d1, d2, d3, mask);

                Unfold16TileHWStorer::store(s, d0, d1, d2, d3, storeMask);
            }

            for (uint16_t th = 0; th < tileWTailRepeatTimes; th++) {
                LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(s2, src, srcTileBufWidth);
                LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(s3, src, srcTileBufWidth);

                TransformVf(s0, s1, s2, s3, d0, d1, d2, d3, mask);

                Unfold16TileHWStorer::store(s, d0, d1, d2, d3, storeMask);
            }

            Unfold16TileHWStorer::UpdateStoreInfo(s, tileW);
        }
    }

    static __simd_callee__ inline void TransformVf(
        RegTensor<T>& s0, RegTensor<T>& s1, RegTensor<T>& s2, RegTensor<T>& s3,
        RegTensor<T>& d0, RegTensor<T>& d1, RegTensor<T>& d2, RegTensor<T>& d3,
        MaskReg& mask)
    {
        Sub(d0, s0, s2, mask);
        Add(d1, s1, s2, mask);
        Sub(d2, s2, s1, mask);
        Sub(d3, s1, s3, mask);
    }
};

template <typename T, typename TilingT>
using FmapConfig = TransformConfig<T,
    F23_FMAP_STRIDE,
    F23_FMAP_WINDOWS,
    Fmap<T, TilingT>, TilingT>;

template <typename T, typename TilingT>
using DyConfig = TransformConfig<T,
    F23_DY_STRIDE,
    F23_DY_WINDOWS,
    Dy<T, TilingT>, TilingT>;
}


template <typename T, typename TilingT>
using WinoFmapFwdTransformer = WinoTransformer<WinoTransformDetail::FmapConfig<T, TilingT> >;


template <typename T, typename TilingT>
using WinoDyFwdTransformer = WinoTransformer<WinoTransformDetail::DyConfig<T, TilingT> >;

#endif //CONV_BP_WINO_TRANSFORM_H