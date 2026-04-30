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

template <uint32_t STRIDE, uint32_t WINDOW_SIZE>
inline uint32_t __aicore__ CalColUnfoldBufWidth(uint32_t tw)
{
    uint32_t srcW = SlideWindows<STRIDE, WINDOW_SIZE>::Tiles2SrcLength(tw);
    //宽为偶数要补一个pad到奇数
    return srcW | TILE_BUF_BANK_CONFLICT_PADDING;
}

inline uint32_t __aicore__ Cal16TileHWBufWidth(uint32_t tileHW)
{
    //宽为偶数要补一个pad到奇数
    return tileHW | TILE_BUF_BANK_CONFLICT_PADDING;
}

//列变换空间大小[th4,srcW]
template <uint32_t STRIDE, uint32_t WINDOW_SIZE>
inline uint32_t __aicore__ CalColUnfoldBufSize(uint32_t th, uint32_t tw)
{
    return TileUnfoldSize(th) * CalColUnfoldBufWidth<STRIDE, WINDOW_SIZE>(tw);
}

//将每个滑窗内16个点拖到外轴后占用的空间[16,TileHW]
inline uint32_t __aicore__ Cal16TileHWBufSize(uint32_t tileHW)
{
    return F23_TRANSFORM_TILE_ELEMENTS_16 * Cal16TileHWBufWidth(tileHW);
}
}

template <typename T>
inline uint32_t __aicore__ SingleShapeTileBufSize(uint32_t c, uint32_t tileHW)
{
    return WinoTransformDetail::Cal16TileHWBufSize(tileHW) * Ops::Base::CeilAlign(c, C0<T>());
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

    static __simd_callee__ inline void UnfoldColsVf(
        __ubuf__ T* unfoldColBuf,
        __ubuf__ T* srcBuf,
        const UnfoldColParamsT& params)
    {
        Impl::UnfoldColsVf(unfoldColBuf, srcBuf, params);
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

template <typename T,
    uint32_t STRIDE,
    uint32_t WINDOW_SIZE,
    typename UnfoldImpl,
    bool hasInputPadding>
class WinoTransformer {
public:
    using SlideWin = SlideWindows<STRIDE, WINDOW_SIZE>;
    using UnfoldPolicy = WinoTransformDetail::UnfoldIntf<T, UnfoldImpl>;

    __aicore__ inline WinoTransformer(
        __gm__ T* in,
        const uint32_t srcH,
        const uint32_t srcW,
        const uint16_t padH,
        const uint16_t padW)
        : srcH_(srcH),
          srcW_(srcW),
          padH_(padH),
          padW_(padW)
    {
        gm_.SetGlobalBuffer(in);
    }

    __aicore__ static inline uint32_t CalculateTmpBufLength(uint32_t tileH, uint32_t tileW)
    {
        uint32_t s0 = CalculateTransposeBufC0SafeElements(tileH, tileW);
        uint32_t s1 = WinoTransformDetail::CalColUnfoldBufSize<STRIDE, WINDOW_SIZE>(
                          tileH, tileW) * C0<T>();
        return s0 + s1;
    }

    __aicore__ static inline void assignSubTmpBuf(
        uint32_t tileH, uint32_t tileW,
        LocalTensor<T>& tmpBuf,
        LocalTensor<T>& transposeBufOut,
        LocalTensor<T>& colTransBufOut)
    {
        uint32_t s0 = CalculateTransposeBufC0SafeElements(tileH, tileW);
        transposeBufOut = tmpBuf[0];
        transposeBufOut.SetSize(s0);

        uint32_t s1 = WinoTransformDetail::CalColUnfoldBufSize<STRIDE, WINDOW_SIZE>(
                         tileH, tileW) * C0<T>();
        colTransBufOut = tmpBuf[s0];
        colTransBufOut.SetSize(s1);
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

    __aicore__ inline void CopyIn(
        const AscendC::LocalTensor<T>& buf,
        const TileBox& box,
        const uint64_t batchOffset) const
    {
        if (const HWBox& src = box.src; src.elements != 0) {
            AscendC::DataCopyExtParams params;
            params.blockCount = src.hLength;
            params.blockLen = src.wLength * sizeof(T);
            params.srcStride = (srcW_ - src.wLength) * sizeof(T);
            params.dstStride = 0;

            uint32_t fmapHWAligned16 = Ops::Base::CeilAlign(src.elements, HW_SRC_ALIGNED_16);

            AscendC::LoopModeParams loop;
            loop.loop1Size = box.c.length;
            loop.loop1SrcStride = static_cast<uint64_t>(srcW_) * srcH_ * sizeof(T);
            loop.loop1DstStride = fmapHWAligned16 * sizeof(T);
            loop.loop2Size = 1;
            loop.loop2SrcStride = 0;
            loop.loop2DstStride = 0;

            SetLoopModePara(loop, AscendC::DataCopyMVType::OUT_TO_UB);

            // 在大小为[c1,th4,tw4,c0]的buf里面从搬入大小为[c,aligned16(hw)]大小的数据
            // 由于c1c0 >= c
            // 且根据数学推到[th4,tw4] >= align16(hw)
            // 所以当前buf一定能放得下所有数据

            AscendC::DataCopyPad<T, AscendC::PaddingMode::Compact>(
                buf,
                gm_[batchOffset +
                    static_cast<uint64_t>(box.c.idx) * srcH_ * srcW_ +
                    static_cast<uint64_t>(src.hIdx) * srcW_ +
                    src.wIdx],
                params,
                {false, 0, 0, 0});
            AscendC::ResetLoopModePara(AscendC::DataCopyMVType::OUT_TO_UB);
        }
    }

    __aicore__ inline void Compute(
        AscendC::LocalTensor<T>& mainBuf,
        AscendC::LocalTensor<T>& tmpBuf,
        const TileBox& box) const
    {
        uint16_t c1 = box.c.c1;

        if (const HWBox& src = box.src; src.elements != 0) {
            uint32_t fmapHWAligned16 = Ops::Base::CeilAlign(src.elements, HW_SRC_ALIGNED_16);

            uint32_t padTopOffset = box.pad.hTop * src.wLength * C0<T>();
            uint32_t srcC1Stride = fmapHWAligned16 * C0<T>();
            uint32_t tileUnfoldC1Stride = SingleShapeTileBufSize<T>(C0<T>(), box.tile.elements);

            const auto params = UnfoldPolicy::InitUnfoldParams(box);
            const typename UnfoldPolicy::UnfoldColParamsT& ucp = AscendC::Std::get<0>(params);
            const typename UnfoldPolicy::UnfoldRowParamsT& urp = AscendC::Std::get<1>(params);

            LocalTensor<T> transposeBuf;
            LocalTensor<T> colTranBuf;
            assignSubTmpBuf(box.tile.hLength, box.tile.wLength, tmpBuf, transposeBuf, colTranBuf);
            for (uint16_t i = 0; i < c1; i++) {
                //从末端开始处理，由于mainBuf的头部搬入了整块fmap
                //如果顺序处理,那可能在变换第一个c1的fmap时,污染了第二个c1的fmap
                //逆序处理就没这个问题,因为尾部不会连着其他fmap
                const uint16_t c1Idx = c1 - i - 1;

                TransposeCHW2C1HWC0(
                    mainBuf[c1Idx * srcC1Stride],
                    transposeBuf[padTopOffset],
                    fmapHWAligned16);

                //TODO 处理pad
                AscendC::LocalTensor<T> outBuf = mainBuf[c1Idx * tileUnfoldC1Stride];
                UnfoldVf(
                    reinterpret_cast<__ubuf__ T*>(outBuf.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ T*>(colTranBuf.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ T*>(transposeBuf.GetPhyAddr()),
                    ucp, urp);
            }
        } else {
            //整个tile都由padding区域产生,不做计算直接置0,
            AscendC::Duplicate(
                mainBuf,
                static_cast<T>(0),
                SingleShapeTileBufSize<T>(c1 * C0<T>(), box.tile.elements));
        }
    }

    __aicore__ inline void CopyOut(
        AscendC::LocalTensor<T>& mainBuf,
        const NK1C1K0C0<T>& nk1c1k0c0,
        const TileBox& box,
        uint32_t batchIdx,
        uint32_t k1Idx) const
    {
        nk1c1k0c0.CopyK0Out(
            mainBuf,
            box.tile,
            WinoTransformDetail::Cal16TileHWBufWidth(box.tile.elements),
            batchIdx,
            k1Idx,
            box.c.idx / C0<T>(),
            box.c.c1);
    }

private:
    struct Format16TileHWParams {
        uint32_t srcTileBufWidth;
        uint32_t dstTileBufWidthBlocks;
        uint16_t tileH;
        uint16_t tileW;
        uint16_t wRepeatTimes;
    };

    __simd_vf__ static inline void UnfoldVf(
        __ubuf__ T* outBuf,
        __ubuf__ T* colUnfoldBuf,
        __ubuf__ T* transposeBuf,
        const typename UnfoldPolicy::UnfoldColParamsT ucp,
        const typename UnfoldPolicy::UnfoldRowParamsT urp)
    {
        //将workspace中的原始数据变换到tileBuf中
        UnfoldPolicy::UnfoldColsVf(colUnfoldBuf, transposeBuf, ucp);

        AscendC::MicroAPI::LocalMemBar<
            AscendC::MicroAPI::MemType::VEC_STORE,
            AscendC::MicroAPI::MemType::VEC_LOAD>();

        //将workspace的列变换结果在变换到tileBuf中
        UnfoldPolicy::UnfoldRowsVf(outBuf, colUnfoldBuf, urp);
    }

    __aicore__ static inline uint32_t CalculateTransposeBufC0SafeElements(
        uint32_t tileH, uint32_t tileW)
    {
        //需要[c0,align16(hw)]的空间给c0hw转置
        uint32_t h = SlideWin::Tiles2SrcLength(tileH);
        uint32_t w = SlideWin::Tiles2SrcLength(tileW);

        uint32_t safeElements;
        if constexpr (hasInputPadding) {
            //transposeBuf中会做H方向pad的清0
            //假设tileH,tileW都为1,那么fmap下算出来h,w都为4
            //假设存在pad_top为3,那么实际的h为1
            //由于转置指令TransDataTo5HD所需的空间是向上对齐到16的,那么实际转置需要的空间为align16(1*4)=16
            //那么需要 pad_top*4+align16(1*4) =12+16=28超过16
            //所以这里要额外+15防止转置指令写入的数据溢出
            //但如果不存在输入pad_top其实可以不做保护
            safeElements = Ops::Base::CeilAlign(h * w + 15, HW_SRC_ALIGNED_16);
        } else {
            safeElements = Ops::Base::CeilAlign(h * w, HW_SRC_ALIGNED_16);
        }
        return safeElements * C0<T>();
    }

    __aicore__ static inline void TransposeCHW2C1HWC0(
        const AscendC::LocalTensor<T>& srcBuf,
        const AscendC::LocalTensor<T>& dstBuf,
        const uint32_t fmapHWAligned16)
    {
        uint64_t srcList[16];
        uint64_t dstList[16];

        if constexpr (sizeof(T) == 2) {
#pragma unroll
            for (uint32_t i = 0; i < 16; i++) {
                uint32_t s = i * fmapHWAligned16;
                uint32_t d = i * 16;
                srcList[i] = reinterpret_cast<uint64_t>(srcBuf[s].GetPhyAddr());
                dstList[i] = reinterpret_cast<uint64_t>(dstBuf[d].GetPhyAddr());
            }
            AscendC::TransDataTo5HDParams params;
            params.repeatTimes = fmapHWAligned16 / 16;
            params.srcRepStride = params.repeatTimes == 1 ? 0 : 1;
            params.dstRepStride = params.repeatTimes == 1 ? 0 : 16;
            AscendC::TransDataTo5HD<T>(dstList, srcList, params);
        } else if constexpr (sizeof(T) == 4) {
#pragma unroll
            for (uint32_t i = 0; i < 8; i++) {
                uint32_t s = i * fmapHWAligned16;
                uint32_t d = i * 8;
                srcList[i] = reinterpret_cast<uint64_t>(srcBuf[s].GetPhyAddr());
                srcList[i + 8] = reinterpret_cast<uint64_t>(srcBuf[s + 8].GetPhyAddr());
                dstList[i * 2] = reinterpret_cast<uint64_t>(dstBuf[d].GetPhyAddr());
                dstList[i * 2 + 1] = reinterpret_cast<uint64_t>(dstBuf[d + 8 * 8].GetPhyAddr());
            }

            AscendC::TransDataTo5HDParams params;
            params.repeatTimes = fmapHWAligned16 / 16;
            params.srcRepStride = params.repeatTimes == 1 ? 0 : 2;
            params.dstRepStride = params.repeatTimes == 1 ? 0 : 16;
            AscendC::TransDataTo5HD<T>(dstList, srcList, params);
        }
    }

    AscendC::GlobalTensor<T> gm_;
    const uint32_t srcH_;
    const uint32_t srcW_;
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
    uint32_t wPadLeftOffset;
    uint32_t tileBufWidth;
    uint16_t wRepeatTimes;
    uint16_t tileH;
};

struct DefaultUnfoldRowParams {
    uint32_t hValidElements;
    uint32_t wLoadOffset;
    uint32_t srcTileBufWidthBlocks;
    uint32_t dstTileBufWidthBlocks;
    uint16_t hRepeatTimes;
    uint16_t tileW;
    uint16_t tileH;
};

template <typename T, uint32_t STRIDE, uint32_t WINDOW>
static inline __aicore__ void InitDefaultUnfoldParams(
    const TileBox& box,
    DefaultUnfoldColParams& ucp,
    DefaultUnfoldRowParams& urp)
{
    ucp.wValidElements = box.src.wLength * C0<T>();
    ucp.tileBufWidth = CalColUnfoldBufWidth<STRIDE, WINDOW>(box.tile.wLength) * C0<T>();
    ucp.wPadLeftOffset = box.pad.wLeft * C0<T>();
    ucp.wRepeatTimes = Ops::Base::CeilDiv(ucp.wValidElements, VL<T>());
    ucp.tileH = box.tile.hLength;

    urp.hValidElements = TileUnfoldSize(box.tile.hLength) * C0<T>();
    urp.srcTileBufWidthBlocks = ucp.tileBufWidth;
    urp.dstTileBufWidthBlocks = Cal16TileHWBufWidth(box.tile.elements);
    urp.hRepeatTimes = Ops::Base::CeilDiv(urp.hValidElements, VL<T>());
    urp.tileW = box.tile.wLength;
    urp.tileH = box.tile.hLength;
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

        InitDefaultUnfoldParams<T, F23_DY_STRIDE, F23_DY_WINDOWS>(box, ucp, urp);

        return AscendC::Std::make_tuple(ucp, urp);
    }

    static __simd_callee__ inline void UnfoldColsVf(
        __ubuf__ T* tileBuf,
        __ubuf__ T* dyBuf,
        const DefaultUnfoldColParams& params)
    {
        const uint32_t wValidElements = params.wValidElements;
        const uint32_t wPadLeftOffset = params.wPadLeftOffset;
        const uint32_t tileBufWidth = params.tileBufWidth;
        const uint16_t wRepeatTimes = params.wRepeatTimes;
        const uint16_t tileH = params.tileH;

        __ubuf__ T* dst0 = tileBuf + wPadLeftOffset;
        uint32_t maskValue = wValidElements;
        for (uint16_t i = 0; i < wRepeatTimes; i++) {
            MaskReg mask = UpdateMask<T>(maskValue);

            const uint32_t wOffset = i * VL<T>();

            __ubuf__ T* src = dyBuf + wOffset;
            __ubuf__ T* dst = dst0 + wOffset;

            for (uint16_t th = 0; th < tileH; th++) {
                RegTensor<T> s0;
                RegTensor<T> s1;

                LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(s0, src, wValidElements);
                LoadAlign(s1, src);

                RegTensor<T> d0;
                RegTensor<T> d1;
                RegTensor<T> d2;
                TransformVf(s0, s1, d0, d1, d2, mask);

                StoreAlign<T, PostLiteral::POST_MODE_UPDATE>(dst, s0, tileBufWidth, mask);
                StoreAlign<T, PostLiteral::POST_MODE_UPDATE>(dst, d0, tileBufWidth, mask);
                StoreAlign<T, PostLiteral::POST_MODE_UPDATE>(dst, d1, tileBufWidth, mask);
                StoreAlign(dst, d2, mask);
            }
        }
    }

    static __simd_callee__ inline void UnfoldRowsVf(
        __ubuf__ T* out,
        __ubuf__ T* buf,
        const DefaultUnfoldRowParams& params)
    {
        const uint32_t dstTileBufWidthBlocks = params.dstTileBufWidthBlocks;
        const uint32_t srcTileBufWidthBlocks = params.srcTileBufWidthBlocks;
        const uint32_t hValidElements = params.hValidElements;
        const uint16_t hRepeatTimes = params.hRepeatTimes;
        const uint16_t tileW = params.tileW;
        const uint16_t tileH = params.tileH;

        uint32_t maskValue = hValidElements;
        uint32_t storeMaskValue = tileH * VL<T>();
        __ubuf__ T* dst0 = out;
        __ubuf__ T* dst1 = out + dstTileBufWidthBlocks * C0<T>() * F23_TRANSFORM_TILE_SIZE_4 * 2;

        for (uint16_t i = 0; i < hRepeatTimes; i++) {
            MaskReg mask = UpdateMask<T>(maskValue);
            MaskReg storeMask0 = UpdateMask<T>(storeMaskValue);
            MaskReg storeMask1 = UpdateMask<T>(storeMaskValue);

            const uint32_t hOffset = srcTileBufWidthBlocks * i * VL<T>();

            __ubuf__ T* src = buf + hOffset;

            for (uint16_t th = 0; th < tileW; th++) {
                RegTensor<T> s0;
                RegTensor<T> s1;

                LoadAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    s0, src, srcTileBufWidthBlocks, 1, mask);
                LoadAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    s1, src, srcTileBufWidthBlocks, 1, mask);

                RegTensor<T> d0;
                RegTensor<T> d1;
                RegTensor<T> d2;
                TransformVf(s0, s1, d0, d1, d2, mask);

                RegTensor<T> t0;
                RegTensor<T> t1;
                RegTensor<T> t2;
                RegTensor<T> t3;
                Interleave(t0, t2, s0, d0);
                Interleave(t1, t3, d1, d2);

                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    dst0, t0, dstTileBufWidthBlocks, 1, storeMask0);
                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    dst1, t1, dstTileBufWidthBlocks, 1, storeMask0);

                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    dst0, t2, dstTileBufWidthBlocks, 1, storeMask1);
                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    dst1, t3, dstTileBufWidthBlocks, 1, storeMask1);
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

        InitDefaultUnfoldParams<T, F23_FMAP_STRIDE, F23_FMAP_WINDOWS>(box, ucp, urp);

        ucp.tileHMainRepeatTimes = ucp.tileH >> 1;
        ucp.tileHTailRepeatTimes = ucp.tileH & 1;

        urp.tileWMainRepeatTimes = urp.tileW >> 1;
        urp.tileWTailRepeatTimes = urp.tileW & 1;

        return AscendC::Std::make_tuple(ucp, urp);
    }

    static __simd_callee__ inline void UnfoldColsVf(
        __ubuf__ T* tileBuf,
        __ubuf__ T* fmapBuf,
        const UnfoldFmapColParams& params)
    {
        const uint32_t wValidElements = params.wValidElements;
        const uint32_t wPadLeftOffset = params.wPadLeftOffset;
        const uint32_t tileBufWidth = params.tileBufWidth;
        const uint16_t wRepeatTimes = params.wRepeatTimes;
        const uint16_t tileHMainRepeatTimes = params.tileHMainRepeatTimes;
        const uint16_t tileHTailRepeatTimes = params.tileHTailRepeatTimes;

        __ubuf__ T* dst0 = tileBuf + wPadLeftOffset;

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

            __ubuf__ T* dst = dst0 + wOffset;
            for (uint16_t th = 0; th < tileHMainRepeatTimes; th++) {
                LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(s2, src, wValidElements);
                LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(s3, src, wValidElements);

                TransformVf(s0, s1, s2, s3, d0, d1, d2, d3, mask);

                StoreAlign<T, PostLiteral::POST_MODE_UPDATE>(dst, d0, tileBufWidth, mask);
                StoreAlign<T, PostLiteral::POST_MODE_UPDATE>(dst, d1, tileBufWidth, mask);
                StoreAlign<T, PostLiteral::POST_MODE_UPDATE>(dst, d2, tileBufWidth, mask);
                StoreAlign<T, PostLiteral::POST_MODE_UPDATE>(dst, d3, tileBufWidth, mask);

                LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(s0, src, wValidElements);
                LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(s1, src, wValidElements);

                TransformVf(s2, s3, s0, s1, d0, d1, d2, d3, mask);

                StoreAlign<T, PostLiteral::POST_MODE_UPDATE>(dst, d0, tileBufWidth, mask);
                StoreAlign<T, PostLiteral::POST_MODE_UPDATE>(dst, d1, tileBufWidth, mask);
                StoreAlign<T, PostLiteral::POST_MODE_UPDATE>(dst, d2, tileBufWidth, mask);
                StoreAlign<T, PostLiteral::POST_MODE_UPDATE>(dst, d3, tileBufWidth, mask);
            }

            for (uint16_t th = 0; th < tileHTailRepeatTimes; th++) {
                LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(s2, src, wValidElements);
                LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(s3, src, wValidElements);

                TransformVf(s0, s1, s2, s3, d0, d1, d2, d3, mask);

                StoreAlign<T, PostLiteral::POST_MODE_UPDATE>(dst, d0, tileBufWidth, mask);
                StoreAlign<T, PostLiteral::POST_MODE_UPDATE>(dst, d1, tileBufWidth, mask);
                StoreAlign<T, PostLiteral::POST_MODE_UPDATE>(dst, d2, tileBufWidth, mask);
                StoreAlign<T, PostLiteral::POST_MODE_UPDATE>(dst, d3, tileBufWidth, mask);
            }
        }
    }

    static __simd_callee__ inline void UnfoldRowsVf(
        __ubuf__ T* out,
        __ubuf__ T* buf,
        const UnfoldFmapRowParams& params)
    {
        const uint32_t dstTileBufWidthBlocks = params.dstTileBufWidthBlocks;
        const uint32_t srcTileBufWidthBlocks = params.srcTileBufWidthBlocks;
        const uint32_t hValidElements = params.hValidElements;
        const uint16_t hRepeatTimes = params.hRepeatTimes;
        const uint16_t tileWMainRepeatTimes = params.tileWMainRepeatTimes;
        const uint16_t tileWTailRepeatTimes = params.tileWTailRepeatTimes;
        const uint16_t tileH = params.tileH;

        uint32_t maskValue = hValidElements;
        uint32_t storeMaskValue = tileH * VL<T>();
        __ubuf__ T* dst0 = out;
        __ubuf__ T* dst1 = out + dstTileBufWidthBlocks * C0<T>() * F23_TRANSFORM_TILE_SIZE_4 * 2;

        for (uint16_t i = 0; i < hRepeatTimes; i++) {
            RegTensor<T> s0;
            RegTensor<T> s1;
            RegTensor<T> s2;
            RegTensor<T> s3;

            RegTensor<T> d0;
            RegTensor<T> d1;
            RegTensor<T> d2;
            RegTensor<T> d3;

            RegTensor<T> t0;
            RegTensor<T> t1;
            RegTensor<T> t2;
            RegTensor<T> t3;

            MaskReg mask = UpdateMask<T>(maskValue);
            MaskReg storeMask0 = UpdateMask<T>(storeMaskValue);
            MaskReg storeMask1 = UpdateMask<T>(storeMaskValue);

            const uint32_t hOffset = srcTileBufWidthBlocks * i * VL<T>();

            __ubuf__ T* src = buf + hOffset;

            LoadAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                s0, src, srcTileBufWidthBlocks, 1, mask);
            LoadAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                s1, src, srcTileBufWidthBlocks, 1, mask);

            for (uint16_t tw = 0; tw < tileWMainRepeatTimes; tw++) {
                LoadAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    s2, src, srcTileBufWidthBlocks, 1, mask);
                LoadAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    s3, src, srcTileBufWidthBlocks, 1, mask);

                TransformVf(s0, s1, s2, s3, d0, d1, d2, d3, mask);

                Interleave(t0, t2, d0, d1);
                Interleave(t1, t3, d2, d3);

                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    dst0, t0, dstTileBufWidthBlocks, 1, storeMask0);
                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    dst1, t1, dstTileBufWidthBlocks, 1, storeMask0);

                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    dst0, t2, dstTileBufWidthBlocks, 1, storeMask1);
                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    dst1, t3, dstTileBufWidthBlocks, 1, storeMask1);

                LoadAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    s0, src, srcTileBufWidthBlocks, 1, mask);
                LoadAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    s1, src, srcTileBufWidthBlocks, 1, mask);

                TransformVf(s2, s3, s0, s1, d0, d1, d2, d3, mask);

                Interleave(t0, t2, d0, d1);
                Interleave(t1, t3, d2, d3);

                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    dst0, t0, dstTileBufWidthBlocks, 1, storeMask0);
                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    dst1, t1, dstTileBufWidthBlocks, 1, storeMask0);

                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    dst0, t2, dstTileBufWidthBlocks, 1, storeMask1);
                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    dst1, t3, dstTileBufWidthBlocks, 1, storeMask1);
            }

            for (uint16_t th = 0; th < tileWTailRepeatTimes; th++) {
                LoadAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    s2, src, srcTileBufWidthBlocks, 1, mask);
                LoadAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    s3, src, srcTileBufWidthBlocks, 1, mask);

                TransformVf(s0, s1, s2, s3, d0, d1, d2, d3, mask);

                Interleave(t0, t2, d0, d1);
                Interleave(t1, t3, d2, d3);

                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    dst0, t0, dstTileBufWidthBlocks, 1, storeMask0);
                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    dst1, t1, dstTileBufWidthBlocks, 1, storeMask0);

                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    dst0, t2, dstTileBufWidthBlocks, 1, storeMask1);
                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    dst1, t3, dstTileBufWidthBlocks, 1, storeMask1);
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
}


template <typename T>
using WinoFmapTransformer = WinoTransformer<T,
    WinoTransformDetail::F23_FMAP_STRIDE,
    WinoTransformDetail::F23_FMAP_WINDOWS,
    WinoTransformDetail::Fmap<T>, true>;

template <typename T>
using WinoDyTransformer = WinoTransformer<T,
    WinoTransformDetail::F23_DY_STRIDE,
    WinoTransformDetail::F23_DY_WINDOWS,
    WinoTransformDetail::Dy<T>, false>;

#endif //CONV_BP_WINO_TRANSFORM_H