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
};

struct TileBox {
    HWBox tile;
    HWBox src;
    HWPad pad;
    CSlice c;
};

namespace WinoTransformDetail {
inline uint32_t __aicore__ CalColUnfoldBufWidth(uint32_t th)
{
    //要补一个pad到奇数
    return TileUnfoldSize(th) | TILE_BUF_BANK_CONFLICT_PADDING;
}

inline uint32_t __aicore__ Cal16TileHWBufWidth(uint32_t tileHW)
{
    //补一个pad到奇数
    return tileHW | TILE_BUF_BANK_CONFLICT_PADDING;
}

//列变换空间大小[srcW,th4],将th4转置方便行变换时顺序读取
template <uint32_t STRIDE, uint32_t WINDOW_SIZE>
inline uint32_t __aicore__ CalColUnfoldBufSize(uint32_t th, uint32_t tw)
{
    uint32_t srcW = SlideWindows<STRIDE, WINDOW_SIZE>::Tiles2SrcLength(tw);
    return srcW * CalColUnfoldBufWidth(th);
}

//将每个滑窗内16个点拖到外轴后占用的空间[16,TileHW]
inline uint32_t __aicore__ Cal16TileHWBufSize(uint32_t tileHW)
{
    return F23_TRANSFORM_TILE_ELEMENTS_16 * Cal16TileHWBufWidth(tileHW);
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

    template <bool HasHPadding>
    static __simd_callee__ inline void UnfoldColsVf(
        __ubuf__ T* unfoldColBuf,
        __ubuf__ T* srcBuf,
        const UnfoldColParamsT& params)
    {
        Impl::template UnfoldColsVf<HasHPadding>(unfoldColBuf, srcBuf, params);
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
    typename UnfoldImplType>
struct TransformConfig {
    using T = Type;
    using UnfoldImpl = UnfoldImplType;

    static constexpr uint32_t STRIDE = STRIDE_VAL;
    static constexpr uint32_t WINDOW_SIZE = WINDOWS_SIZE_VAL;
};


template <typename Config>
class WinoTransformer {
public:
    static constexpr uint32_t STRIDE = Config::STRIDE;
    static constexpr uint32_t WINDOW_SIZE = Config::WINDOW_SIZE;
    using T = typename Config::T;
    using SlideWin = SlideWindows<STRIDE, WINDOW_SIZE>;
    using UnfoldPolicy = WinoTransformDetail::UnfoldIntf<T, typename Config::UnfoldImpl>;
    struct TransposeCacheShape;

    __aicore__ inline WinoTransformer(
        __gm__ T* in,
        __gm__ T* transposeCache,
        const TransposeCacheShape& transposeCacheShape,
        const uint32_t srcH,
        const uint32_t srcW,
        const uint32_t srcC,
        const uint16_t padH,
        const uint16_t padW)
        : transposeCacheShape_(transposeCacheShape),
          srcH_(srcH),
          srcW_(srcW),
          srcC_(srcC),
          padH_(padH),
          padW_(padW)
    {
        transposeCache_.SetGlobalBuffer(transposeCache);
        gm_.SetGlobalBuffer(in);
    }

    __aicore__ static inline uint32_t GetOutputBufSize(
        uint32_t c,
        uint32_t tileH,
        uint32_t tileW)
    {
        //output buf会同时存正变换结果并且在计算过程总复用一部分存放转置数据
        //常规情况正变换结果时原始输入的4倍，转置的空间为原始hw对齐到16，所以正变换结果肯定更大
        //因此这样复用可以减少一些空间
        uint32_t tileBufSize = GetTransformBufSize(c, tileH * tileW);
        //转置占用空间当前和输入相同align(srcHW,16)*c1c0
        uint32_t transposeBufSize = TilesToC1SrcHW16C0(c, tileH, tileW);
        return Std::max(transposeBufSize, tileBufSize);
    }

    __aicore__ static inline uint32_t GetInputBufSize(
        uint32_t c,
        uint32_t tileH,
        uint32_t tileW)
    {
        return TilesToC1SrcHW16C0(c, tileH, tileW);
    }

    __aicore__ static inline uint32_t GetTmpBufLength(uint32_t tileH, uint32_t tileW)
    {
        return WinoTransformDetail::CalColUnfoldBufSize<STRIDE, WINDOW_SIZE>(
                   tileH, tileW) * C0<T>();
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

    template <bool PreTransposeSrc>
    __aicore__ inline void CopyIn(
        const AscendC::LocalTensor<T>& srcBuf,
        const TileBox& box,
        const uint32_t batchIdx,
        const uint32_t kIdx) const
    {
        const HWBox& src = box.src;
        if (unlikely(src.elements==0)) {
            return;
        }

        if constexpr (PreTransposeSrc) {
            uint64_t gmOffset = transposeCacheShape_.GetOffset(batchIdx, kIdx, box.c.idx / C0<T>());
            AscendC::DataCopyParams params;
            params.blockCount = box.c.c1;
            params.blockLen = src.elements;
            params.srcGap = transposeCacheShape_.k0 - src.elements;
            params.dstGap = 0;
            AscendC::DataCopy(srcBuf, transposeCache_[gmOffset], params);
        } else {
            AscendC::DataCopyExtParams params;
            params.blockCount = src.hLength;
            params.blockLen = src.wLength * sizeof(T);
            params.srcStride = (srcW_ - src.wLength) * sizeof(T);
            params.dstStride = 0;

            uint32_t srcHW16 = Ops::Base::CeilAlign(src.elements, HW_SRC_ALIGNED_16);
            AscendC::LoopModeParams loop;
            loop.loop1Size = box.c.length;
            loop.loop1SrcStride = static_cast<uint64_t>(srcW_) * srcH_ * sizeof(T);
            loop.loop1DstStride = srcHW16 * sizeof(T);
            loop.loop2Size = 1;
            loop.loop2SrcStride = 0;
            loop.loop2DstStride = 0;

            SetLoopModePara(loop, AscendC::DataCopyMVType::OUT_TO_UB);

            AscendC::DataCopyPad<T, AscendC::PaddingMode::Compact>(
                srcBuf,
                gm_[static_cast<uint64_t>(batchIdx) * srcC_ * srcH_ * srcW_ +
                    static_cast<uint64_t>(box.c.idx) * srcH_ * srcW_ +
                    static_cast<uint64_t>(src.hIdx) * srcW_ +
                    src.wIdx],
                params,
                {false, 0, 0, 0});
            AscendC::ResetLoopModePara(AscendC::DataCopyMVType::OUT_TO_UB);
        }
    }

    template <bool PreTransposeSrc>
    __aicore__ inline void Compute(
        AscendC::LocalTensor<T>& srcBuf,
        AscendC::LocalTensor<T>& outBuf,
        AscendC::LocalTensor<T>& tmpBuf,
        const TileBox& box) const
    {
        uint16_t c1 = box.c.c1;
        uint32_t outBufSize = GetOutputBufSize(c1 * C0<T>(), box.tile.hLength, box.tile.wLength);
        const HWBox& src = box.src;
        if (unlikely(src.elements==0)) {
            //整个tile都由padding区域产生,不做计算直接置0,
            AscendC::Duplicate(outBuf, static_cast<T>(0), outBufSize);
            return;
        }

        const uint32_t srcHW16 = Ops::Base::CeilAlign(src.elements, HW_SRC_ALIGNED_16);
        const uint32_t srcHW16C0 = srcHW16 * C0<T>();
        const uint32_t transposeLength = srcHW16C0;
        const uint32_t srcC1Stride = PreTransposeSrc ? src.elements * C0<T>() : srcHW16C0;

        uint32_t transformBufC1Stride = GetTransformBufSize(C0<T>(), box.tile.elements);

        const auto params = UnfoldPolicy::InitUnfoldParams(box);
        const typename UnfoldPolicy::UnfoldColParamsT& ucp = AscendC::Std::get<0>(params);
        const typename UnfoldPolicy::UnfoldRowParamsT& urp = AscendC::Std::get<1>(params);

        LocalTensor<T> hwC0Buf;
        __ubuf__ T* hwC0BufAddr;
        if constexpr (!PreTransposeSrc) {
            //转置复用输出buf空间
            hwC0Buf = outBuf[outBufSize - transposeLength];
            hwC0BufAddr = reinterpret_cast<__ubuf__ T*>(hwC0Buf.GetPhyAddr());
        }
        __ubuf__ T* colTransBufAddr = reinterpret_cast<__ubuf__ T*>(tmpBuf.GetPhyAddr());
        __ubuf__ T* dstBufAddr = reinterpret_cast<__ubuf__ T*>(outBuf.GetPhyAddr());

        for (uint16_t c1Idx = 0; c1Idx < c1; c1Idx++) {
            LocalTensor<T> c1Src = srcBuf[c1Idx * srcC1Stride];

            if constexpr (PreTransposeSrc) {
                hwC0BufAddr = reinterpret_cast<__ubuf__ T*>(c1Src.GetPhyAddr());
            } else {
                TransposeCHW2C1HWC0(c1Src, hwC0Buf, srcHW16);
            }

            if (box.pad.hTop == 0 && box.pad.hBottom == 0) {
                UnfoldVf<false>(dstBufAddr, colTransBufAddr, hwC0BufAddr, ucp, urp);
            } else {
                UnfoldVf<true>(dstBufAddr, colTransBufAddr, hwC0BufAddr, ucp, urp);
            }

            dstBufAddr += transformBufC1Stride;
        }
    }


    __aicore__ inline void SetNK1C1K0C0CopyParams(
        NK1C1K0C0::CopyK0Params& copyParams,
        const TileBox& box) const
    {
        copyParams.tiles = box.tile.elements;
        copyParams.srcBufWidthBlockStride = WinoTransformDetail::Cal16TileHWBufWidth(box.tile.elements);
        copyParams.c1Idx = box.c.idx / C0<T>();
        copyParams.c1Length = box.c.c1;
    }

    __aicore__ inline void PreTranspose(
        LocalTensor<T>& srcBuf,
        LocalTensor<T>& outBuf,
        const TileBox& box) const
    {
        const HWBox& src = box.src;
        if (unlikely(src.elements==0)) {
            return;
        }

        const uint32_t srcHW16 = Ops::Base::CeilAlign(src.elements, HW_SRC_ALIGNED_16);
        const uint32_t srcHW16C0 = srcHW16 * C0<T>();
        const uint32_t transposeLength = srcHW16C0;
        const uint32_t srcC1Stride = srcHW16C0;

        for (uint16_t c1Idx = 0; c1Idx < box.c.c1; c1Idx++) {
            LocalTensor<T> c1Src = srcBuf[c1Idx * srcC1Stride];
            LocalTensor<T> c1Out = outBuf[c1Idx * transposeLength];
            TransposeCHW2C1HWC0(c1Src, c1Out, srcHW16);
        }
    }

    __aicore__ inline void CopyPreTransposeOut(
        LocalTensor<T>& outBuf,
        const TileBox& box,
        uint32_t batchIdx,
        uint32_t kIdx) const
    {
        const HWBox& src = box.src;
        if (unlikely(src.elements==0)) {
            return;
        }

        const uint32_t srcHW16 = Ops::Base::CeilAlign(src.elements, HW_SRC_ALIGNED_16);
        const uint32_t transposeLengthBlocks = srcHW16;

        DataCopyParams params;
        params.blockCount = box.c.c1;
        params.blockLen = box.src.elements;
        params.srcGap = transposeLengthBlocks - box.src.elements;
        params.dstGap = transposeCacheShape_.k0 - box.src.elements;

        uint64_t gmOffset = transposeCacheShape_.GetOffset(batchIdx, kIdx, box.c.idx / C0<T>());
        DataCopy(transposeCache_[gmOffset], outBuf, params);
    }

    struct TransposeCacheShape {
        // nk1c1k0c0格式
        const uint32_t k1;
        const uint32_t c1;
        const uint32_t k0;

        __aicore__ inline TransposeCacheShape(
            uint32_t c,
            uint32_t tileH,
            uint32_t tileW,
            uint32_t singleShapeTileH,
            uint32_t singleShapeTileW)
            : k1(Ops::Base::CeilDiv(tileH, singleShapeTileH) * Ops::Base::CeilDiv(tileW, singleShapeTileW)),
              c1(Ops::Base::CeilDiv(c, C0<T>())),
              k0(SlideWin::Tiles2SrcLength(singleShapeTileH) * SlideWin::Tiles2SrcLength(singleShapeTileW))
        {
        }

        __aicore__ inline uint64_t GetOffset(uint32_t batchIdx, uint32_t k1Idx, uint32_t c1Idx) const
        {
            uint64_t k0c0 = static_cast<uint64_t>(k0) * C0<T>();
            uint64_t c1k0c0 = c1 * k0c0;
            uint64_t k1c1k0c0 = k1 * c1k0c0;
            uint64_t gmOffset = batchIdx * k1c1k0c0 + k1Idx * c1k0c0 + c1Idx * k0c0;
            return gmOffset;
        }
    };

private:
    __aicore__ static inline uint32_t GetTransformBufSize(uint32_t c, uint32_t tileHW)
    {
        return WinoTransformDetail::Cal16TileHWBufSize(tileHW) * Ops::Base::CeilAlign(c, C0<T>());
    }

    __aicore__ static inline uint32_t TilesToC1SrcHW16C0(uint32_t c, uint32_t tileH, uint32_t tileW)
    {
        return C1SrcHW16C0(c, SlideWin::Tiles2SrcLength(tileH) * SlideWin::Tiles2SrcLength(tileW));
    }

    __aicore__ static inline uint32_t C1SrcHW16C0(uint32_t c, uint32_t hw)
    {
        return Ops::Base::CeilAlign(hw, HW_SRC_ALIGNED_16) *
               Ops::Base::CeilAlign(c, C0<T>());
    }

    template <bool HasHPadding>
    __simd_vf__ static inline void UnfoldVf(
        __ubuf__ T* outBuf,
        __ubuf__ T* colUnfoldBuf,
        __ubuf__ T* transposeBuf,
        const typename UnfoldPolicy::UnfoldColParamsT ucp,
        const typename UnfoldPolicy::UnfoldRowParamsT urp)
    {
        //将workspace中的原始数据变换到tileBuf中
        UnfoldPolicy::template UnfoldColsVf<HasHPadding>(colUnfoldBuf, transposeBuf, ucp);

        AscendC::MicroAPI::LocalMemBar<
            AscendC::MicroAPI::MemType::VEC_STORE,
            AscendC::MicroAPI::MemType::VEC_LOAD>();

        //将workspace的列变换结果在变换到tileBuf中
        UnfoldPolicy::UnfoldRowsVf(outBuf, colUnfoldBuf, urp);
    }

    __aicore__ static inline void TransposeCHW2C1HWC0(
        const AscendC::LocalTensor<T>& srcBuf,
        const AscendC::LocalTensor<T>& dstBuf,
        const uint32_t srcHWAligned16)
    {
        uint64_t srcList[16];
        uint64_t dstList[16];

        if constexpr (sizeof(T) == 2) {
#pragma unroll
            for (uint32_t i = 0; i < 16; i++) {
                uint32_t s = i * srcHWAligned16;
                uint32_t d = i * 16;
                srcList[i] = reinterpret_cast<uint64_t>(srcBuf[s].GetPhyAddr());
                dstList[i] = reinterpret_cast<uint64_t>(dstBuf[d].GetPhyAddr());
            }
            AscendC::TransDataTo5HDParams params;
            params.repeatTimes = srcHWAligned16 / 16;
            params.srcRepStride = params.repeatTimes == 1 ? 0 : 1;
            params.dstRepStride = params.repeatTimes == 1 ? 0 : 16;
            AscendC::TransDataTo5HD<T>(dstList, srcList, params);
        } else if constexpr (sizeof(T) == 4) {
#pragma unroll
            for (uint32_t i = 0; i < 8; i++) {
                uint32_t s = i * srcHWAligned16;
                uint32_t d = i * 8;
                srcList[i] = reinterpret_cast<uint64_t>(srcBuf[s].GetPhyAddr());
                srcList[i + 8] = reinterpret_cast<uint64_t>(srcBuf[s + 8].GetPhyAddr());
                dstList[i * 2] = reinterpret_cast<uint64_t>(dstBuf[d].GetPhyAddr());
                dstList[i * 2 + 1] = reinterpret_cast<uint64_t>(dstBuf[d + 8 * 8].GetPhyAddr());
            }

            AscendC::TransDataTo5HDParams params;
            params.repeatTimes = srcHWAligned16 / 16;
            params.srcRepStride = params.repeatTimes == 1 ? 0 : 2;
            params.dstRepStride = params.repeatTimes == 1 ? 0 : 16;
            AscendC::TransDataTo5HD<T>(dstList, srcList, params);
        }
    }


    AscendC::GlobalTensor<T> gm_;
    AscendC::GlobalTensor<T> transposeCache_;
    TransposeCacheShape transposeCacheShape_;
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
    uint32_t wPadLeftStoreOffset;
    uint16_t validHStart;
    uint16_t validHEnd;
    uint16_t wRepeatTimes;
    uint16_t tileH;
};

struct DefaultUnfoldRowParams {
    uint32_t srcTileBufWidth;
    uint32_t dstTileBufWidthBlocks;
    uint16_t hRepeatTimes;
    uint16_t tileW;
    uint16_t tileH;
};

template <typename T>
static inline __aicore__ void InitDefaultUnfoldParams(
    const TileBox& box,
    DefaultUnfoldColParams& ucp,
    DefaultUnfoldRowParams& urp)
{
    ucp.wValidElements = box.src.wLength * C0<T>();
    ucp.tileBufWidthBlocks = CalColUnfoldBufWidth(box.tile.hLength);
    ucp.wPadLeftStoreOffset = box.pad.wLeft * ucp.tileBufWidthBlocks * C0<T>();
    ucp.validHStart = box.pad.hTop;
    ucp.validHEnd = box.pad.hTop + box.src.hLength;
    ucp.wRepeatTimes = Ops::Base::CeilDiv(ucp.wValidElements, VL<T>());
    ucp.tileH = box.tile.hLength;

    urp.srcTileBufWidth = ucp.tileBufWidthBlocks * C0<T>();
    urp.dstTileBufWidthBlocks = Cal16TileHWBufWidth(box.tile.elements);
    urp.hRepeatTimes = Ops::Base::CeilDiv(TileUnfoldSize(box.tile.hLength) * C0<T>(), VL<T>());
    urp.tileW = box.tile.wLength;
    urp.tileH = box.tile.hLength;
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


    static __simd_callee__ inline void CalTileHMainTailRepeatTimes(
        uint16_t hRepeatTimes, uint16_t tileH,
        uint16_t& hMainRepeatTimes, uint16_t& hTailRepeatTimes)
    {
        bool hasTail = hRepeatTimes * AscendC::DEFAULT_BLK_NUM > tileH * F23_TRANSFORM_TILE_SIZE_4;
        hTailRepeatTimes = static_cast<uint16_t>(hasTail);
        hMainRepeatTimes = hRepeatTimes - hTailRepeatTimes;
    }

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

        p.dst4 = p.dst0 + tileW * C0<T>();
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
        uint32_t step = C0<T>() * 2 * tileW;
        p.dst0 += step;
        p.dst1 += step;
        p.dst2 += step;
        p.dst3 += step;
        p.dst4 += step;
        p.dst5 += step;
        p.dst6 += step;
        p.dst7 += step;
    }

    template <bool enableLowHalf, bool enableHighHalf, typename T>
    static __simd_callee__ inline void store(
        StoreInfo<T>& p,
        RegTensor<T>& r0,
        RegTensor<T>& r1,
        RegTensor<T>& r2,
        RegTensor<T>& r3)
    {
        if constexpr (enableLowHalf) {
            StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                p.dst0, r0, p.dstTileBufWidthBlocks, 1, p.lowHalfPartMask);
            StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                p.dst1, r1, p.dstTileBufWidthBlocks, 1, p.lowHalfPartMask);

            StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                p.dst2, r2, p.dstTileBufWidthBlocks, 1, p.lowHalfPartMask);
            StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                p.dst3, r3, p.dstTileBufWidthBlocks, 1, p.lowHalfPartMask);
        }

        if constexpr (enableHighHalf) {
            StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                p.dst4, r0, p.dstTileBufWidthBlocks, 1, p.highHalfPartMask);
            StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                p.dst5, r1, p.dstTileBufWidthBlocks, 1, p.highHalfPartMask);

            StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                p.dst6, r2, p.dstTileBufWidthBlocks, 1, p.highHalfPartMask);
            StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                p.dst7, r3, p.dstTileBufWidthBlocks, 1, p.highHalfPartMask);
        }
    }
};

template <bool hasPadding, bool incHIdxIfHasPadding = false, typename T>
static void __simd_callee__ inline LoadAlignInHPad(
    RegTensor<T>& reg,
    __ubuf__ T*& src,
    uint32_t repeatStride,
    uint16_t& hIdx,
    uint16_t validStart,
    uint16_t validEnd,
    RegTensor<T>& padding)
{
    if constexpr (hasPadding) {
        uint32_t maskValue = VL<T>() * (hIdx >= validStart && hIdx <= validEnd);
        MaskReg padMask = UpdateMask<T>(maskValue);
        RegTensor<T> tmp;
        LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(tmp, src, repeatStride);
        Select(reg, padding, tmp, padMask);

        if constexpr (incHIdxIfHasPadding) {
            hIdx++;
        }
    } else {
        LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(reg, src, repeatStride);
    }
}

template <typename T>
struct Dy {
    using UnfoldRowParamsT = DefaultUnfoldRowParams;
    using UnfoldColParamsT = DefaultUnfoldColParams;

    using ParamsTuple = AscendC::Std::tuple<UnfoldColParamsT, UnfoldRowParamsT>;

    static ParamsTuple inline __aicore__ InitUnfoldParams(const TileBox& box)
    {
        DefaultUnfoldRowParams urp = {};
        DefaultUnfoldColParams ucp = {};

        InitDefaultUnfoldParams<T>(box, ucp, urp);

        return AscendC::Std::make_tuple(ucp, urp);
    }

    template <bool HasHPadding>
    static __simd_callee__ inline void UnfoldColsVf(
        __ubuf__ T* tileBuf,
        __ubuf__ T* dyBuf,
        const DefaultUnfoldColParams& params)
    {
        const uint32_t wValidElements = params.wValidElements;
        const uint16_t validHStart = params.validHStart;
        const uint16_t validHEnd = params.validHEnd;
        const uint32_t wPadLeftStoreOffset = params.wPadLeftStoreOffset;
        const uint32_t tileBufWidthBlocks = params.tileBufWidthBlocks;
        const uint16_t wRepeatTimes = params.wRepeatTimes;
        const uint16_t tileH = params.tileH;

        RegTensor<T> padding;
        Duplicate(padding, 0);

        //地址会往前减去topPad,可能会产生翻转,但是后续应该能转回来
        dyBuf -= wValidElements * validHStart;
        tileBuf += wPadLeftStoreOffset;

        for (uint16_t th = 0; th < tileH; th++) {
            constexpr uint32_t thStride = F23_TRANSFORM_TILE_SIZE_4 * C0<T>();

            __ubuf__ T* dst = tileBuf + thStride * th;

            uint16_t hIdx0 = th * F23_DY_STRIDE;
            uint16_t hIdx1 = hIdx0 + 1;

            __ubuf__ T* src0 = dyBuf + wValidElements * hIdx0;
            __ubuf__ T* src1 = dyBuf + wValidElements * hIdx1;

            uint32_t maskValue = wValidElements;
            for (uint16_t i = 0; i < wRepeatTimes; i++) {
                MaskReg mask = UpdateMask<T>(maskValue);

                RegTensor<T> s0;
                RegTensor<T> s1;

                LoadAlignInHPad<HasHPadding>(s0, src0, VL<T>(), hIdx0, validHStart, validHEnd, padding);
                LoadAlignInHPad<HasHPadding>(s1, src1, VL<T>(), hIdx1, validHStart, validHEnd, padding);

                RegTensor<T> d0;
                RegTensor<T> d1;
                RegTensor<T> d2;
                TransformVf(s0, s1, d0, d1, d2, mask);

                __ubuf__ T* dst0 = dst;

                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    dst0, s0, tileBufWidthBlocks, 1, mask);
                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    dst0, d0, tileBufWidthBlocks, 1, mask);
                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    dst0, d1, tileBufWidthBlocks, 1, mask);
                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    dst0, d2, tileBufWidthBlocks, 1, mask);

                dst += tileBufWidthBlocks * VL<T>();
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
        const uint16_t tileH = params.tileH;

        Unfold16TileHWStorer::StoreInfo<T> s;
        Unfold16TileHWStorer::CreateStoreInfo(s, out, tileW, dstTileBufWidthBlocks);

        uint16_t hTailRepeatTimes;
        uint16_t hMainRepeatTimes;
        Unfold16TileHWStorer::CalTileHMainTailRepeatTimes(hRepeatTimes, tileH, hMainRepeatTimes, hTailRepeatTimes);

        UnfoldRowsVf_<false>(s, hMainRepeatTimes, srcTileBufWidth, tileW, buf);
        UnfoldRowsVf_<true>(s, hTailRepeatTimes, srcTileBufWidth, tileW, buf + hMainRepeatTimes * VL<T>());
    }

    template <bool TailH>
    static __simd_callee__ inline void UnfoldRowsVf_(
        Unfold16TileHWStorer::StoreInfo<T>& s,
        const uint16_t hRepeatTimes,
        const uint32_t srcTileBufWidth,
        const uint16_t tileW,
        __ubuf__ T* buf)
    {
        for (uint16_t i = 0; i < hRepeatTimes; i++) {
            const uint32_t hOffset = i * VL<T>();
            __ubuf__ T* src = buf + hOffset;

            for (uint16_t th = 0; th < tileW; th++) {
                RegTensor<T> s0;
                RegTensor<T> s1;

                LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(s0, src, srcTileBufWidth);
                LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(s1, src, srcTileBufWidth);

                RegTensor<T> d0;
                RegTensor<T> d1;
                RegTensor<T> d2;
                TransformVf(s0, s1, d0, d1, d2, s.maskAll);

                Unfold16TileHWStorer::store<true, !TailH>(s, s0, d0, d1, d2);
            }

            if constexpr (!TailH) {
                Unfold16TileHWStorer::UpdateStoreInfo(s, tileW);
            }
        }
    }

    static __simd_callee__ inline void TransformVf(
        RegTensor<T>& s0, RegTensor<T>& s1,
        RegTensor<T>& d0, RegTensor<T>& d1,
        RegTensor<T>& d2, MaskReg& mask)
    {
        Add(d0, s0, s1, mask);
        Sub(d1, s0, s1, mask);
        Neg(d2, s0, mask);
    }
};

template <typename T>
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

        InitDefaultUnfoldParams<T>(box, ucp, urp);

        ucp.tileHMainRepeatTimes = ucp.tileH >> 1;
        ucp.tileHTailRepeatTimes = ucp.tileH & 1;

        urp.tileWMainRepeatTimes = urp.tileW >> 1;
        urp.tileWTailRepeatTimes = urp.tileW & 1;

        return AscendC::Std::make_tuple(ucp, urp);
    }

    template <bool HasHPadding>
    static __simd_callee__ inline void UnfoldColsVf(
        __ubuf__ T* tileBuf,
        __ubuf__ T* fmapBuf,
        const UnfoldFmapColParams& params)
    {
        const uint32_t wValidElements = params.wValidElements;

        const uint16_t validHStart = params.validHStart;
        const uint16_t validHEnd = params.validHEnd;
        const uint32_t wPadLeftStoreOffset = params.wPadLeftStoreOffset;

        const uint32_t tileBufWidthBlocks = params.tileBufWidthBlocks;
        const uint16_t wRepeatTimes = params.wRepeatTimes;
        const uint16_t tileHMainRepeatTimes = params.tileHMainRepeatTimes;
        const uint16_t tileHTailRepeatTimes = params.tileHTailRepeatTimes;

        RegTensor<T> padding;
        Duplicate(padding, 0);
        tileBuf += wPadLeftStoreOffset;
        //减去padding,翻转应当不影响
        fmapBuf -= wValidElements * validHStart;

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

            uint16_t hIdx = 0;
            LoadAlignInHPad<HasHPadding, true>(
                s0, src, wValidElements,
                hIdx, validHStart, validHEnd, padding);
            LoadAlignInHPad<HasHPadding, true>(
                s1, src, wValidElements,
                hIdx, validHStart, validHEnd, padding);

            __ubuf__ T* dst = tileBuf + tileBufWidthBlocks * wOffset;

            for (uint16_t th = 0; th < tileHMainRepeatTimes; th++) {
                LoadAlignInHPad<HasHPadding, true>(
                    s2, src, wValidElements,
                    hIdx, validHStart, validHEnd, padding);
                LoadAlignInHPad<HasHPadding, true>(
                    s3, src, wValidElements,
                    hIdx, validHStart, validHEnd, padding);

                TransformVf(s0, s1, s2, s3, d0, d1, d2, d3, mask);

                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    dst, d0, tileBufWidthBlocks, 1, mask);
                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    dst, d1, tileBufWidthBlocks, 1, mask);
                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    dst, d2, tileBufWidthBlocks, 1, mask);
                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    dst, d3, tileBufWidthBlocks, 1, mask);

                LoadAlignInHPad<HasHPadding, true>(
                    s0, src, wValidElements,
                    hIdx, validHStart, validHEnd, padding);
                LoadAlignInHPad<HasHPadding, true>(
                    s1, src, wValidElements,
                    hIdx, validHStart, validHEnd, padding);

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
                LoadAlignInHPad<HasHPadding, true>(
                    s2, src, wValidElements,
                    hIdx, validHStart, validHEnd, padding);
                LoadAlignInHPad<HasHPadding, true>(
                    s3, src, wValidElements,
                    hIdx, validHStart, validHEnd, padding);

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
        const uint16_t tileH = params.tileH;
        const uint16_t tileW = params.tileW;

        Unfold16TileHWStorer::StoreInfo<T> s;
        Unfold16TileHWStorer::CreateStoreInfo(s, out, tileW, dstTileBufWidthBlocks);

        uint16_t hTailRepeatTimes;
        uint16_t hMainRepeatTimes;
        Unfold16TileHWStorer::CalTileHMainTailRepeatTimes(hRepeatTimes, tileH, hMainRepeatTimes, hTailRepeatTimes);

        UnfoldRowsVf_<false>(
            s,
            hMainRepeatTimes,
            tileW,
            tileWMainRepeatTimes,
            tileWTailRepeatTimes,
            srcTileBufWidth,
            buf);

        UnfoldRowsVf_<true>(
            s,
            hTailRepeatTimes,
            tileW,
            tileWMainRepeatTimes,
            tileWTailRepeatTimes,
            srcTileBufWidth,
            buf + hMainRepeatTimes * VL<T>());
    }

    template <bool TailH>
    static __simd_callee__ inline void UnfoldRowsVf_(
        Unfold16TileHWStorer::StoreInfo<T>& s,
        const uint16_t hRepeatTimes,
        const uint16_t tileW,
        const uint16_t tileWMainRepeatTimes,
        const uint16_t tileWTailRepeatTimes,
        const uint32_t srcTileBufWidth,
        __ubuf__ T* buf)
    {
        for (uint16_t i = 0; i < hRepeatTimes; i++) {
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

                TransformVf(s0, s1, s2, s3, d0, d1, d2, d3, s.maskAll);

                Unfold16TileHWStorer::store<true, !TailH>(s, d0, d1, d2, d3);

                LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(s0, src, srcTileBufWidth);
                LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(s1, src, srcTileBufWidth);

                TransformVf(s2, s3, s0, s1, d0, d1, d2, d3, s.maskAll);

                Unfold16TileHWStorer::store<true, !TailH>(s, d0, d1, d2, d3);
            }

            for (uint16_t th = 0; th < tileWTailRepeatTimes; th++) {
                LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(s2, src, srcTileBufWidth);
                LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(s3, src, srcTileBufWidth);

                TransformVf(s0, s1, s2, s3, d0, d1, d2, d3, s.maskAll);

                Unfold16TileHWStorer::store<true, !TailH>(s, d0, d1, d2, d3);
            }

            if constexpr (!TailH) {
                Unfold16TileHWStorer::UpdateStoreInfo(s, tileW);
            }
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

template <typename T>
using FmapConfig = TransformConfig<T,
    F23_FMAP_STRIDE,
    F23_FMAP_WINDOWS,
    Fmap<T> >;

template <typename T>
using DyConfig = TransformConfig<T,
    F23_DY_STRIDE,
    F23_DY_WINDOWS,
    Dy<T> >;
}


template <typename T>
using WinoFmapFwdTransformer = WinoTransformer<WinoTransformDetail::FmapConfig<T> >;


template <typename T>
using WinoDyFwdTransformer = WinoTransformer<WinoTransformDetail::DyConfig<T> >;

#endif //CONV_BP_WINO_TRANSFORM_H