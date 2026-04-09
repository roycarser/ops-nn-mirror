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

template <typename T, typename Impl>
struct UnfoldIntf {
    using UnfoldColParamsT = typename Impl::UnfoldColParamsT;
    using UnfoldRowParamsT = typename Impl::UnfoldRowParamsT;

    static __aicore__ AscendC::Std::tuple<UnfoldColParamsT, UnfoldRowParamsT> InitUnfoldParams(const TileBox& box)
    {
        return Impl::InitUnfoldParams(box);
    }

    static __simd_callee__ inline void UnfoldColsFromSrcBufVf(
        __ubuf__ T* __restrict__ tileBuf,
        __ubuf__ T* __restrict__ srcBuf,
        const UnfoldColParamsT& params)
    {
        Impl::UnfoldColsFromSrcBufVf(tileBuf, srcBuf, params);
    }

    static __simd_callee__ inline void UnfoldRowsVf(
        __ubuf__ T* buf,
        const UnfoldRowParamsT& params)
    {
        Impl::UnfoldRowsVf(buf, params);
    }
};

template <typename T,
    uint32_t STRIDE,
    uint32_t WINDOW_SIZE,
    typename UnfoldImpl,
    bool hasInputPadding >
class WinoTransformer {
public:
    using SlideWin = SlideWindows<STRIDE, WINDOW_SIZE>;
    using UnfoldPolicy = UnfoldIntf<T, UnfoldImpl>;

    __aicore__ inline WinoTransformer(
        __gm__ T* gm,
        const uint32_t srcH,
        const uint32_t srcW,
        const uint16_t padH,
        const uint16_t padW)
        : srcH_(srcH),
          srcW_(srcW),
          padH_(padH),
          padW_(padW)
    {
        gm_.SetGlobalBuffer(gm);
    }

    __aicore__ static inline uint32_t CalculateTransposeBufC0Length(
        uint32_t tileH, uint32_t tileW)
    {
        //额外申请[c0,align16(hw)]的空间给c0hw转置
        uint32_t h = SlideWin::Tiles2SrcLength(tileH);
        uint32_t w = SlideWin::Tiles2SrcLength(tileW);

        uint32_t safeElements;
        if constexpr (hasInputPadding) {
            //transposeBuf中会做H方向pad的清0
            //令src中有效数据为valid_h,valid_w
            //那么需要 pad_top*w+ align16(valid_h*valid_w) <= buf_length
            //所以这里要额外+15防止转置指令写入的数据溢出
            //防止pad_top=3,valid_h=1,w=4,这种情况
            //会写入的空间为3*4+16=28超过16
            //但如果不存在输入pad_top其实可以不做保护
            safeElements = Ops::Base::CeilAlign(h * w + 15, HW_SRC_ALIGNED_16);
        } else {
            safeElements = Ops::Base::CeilAlign(h * w, HW_SRC_ALIGNED_16);
        }
        return safeElements * C0<T>();
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

    __aicore__ inline void CopyIn(
        const AscendC::LocalTensor<T>& buf,
        const TileBox& box,
        const uint32_t batchOffset) const
    {
        if (const HWBox& src = box.src; src.elements != 0) {
            AscendC::DataCopyExtParams params;
            params.blockCount = src.hLength;
            params.blockLen = src.wLength * sizeof(T);
            params.srcStride = srcW_ * sizeof(T);
            params.dstStride = 0;

            uint32_t fmapHWAligned16 = Ops::Base::CeilAlign(src.elements, HW_SRC_ALIGNED_16);

            AscendC::LoopModeParams loop;
            loop.loop1Size = box.c.length;
            loop.loop1SrcStride = srcW_ * srcH_ * sizeof(T);
            loop.loop1DstStride = fmapHWAligned16 * sizeof(T);
            loop.loop2Size = 1;
            loop.loop1SrcStride = 0;
            loop.loop2DstStride = 0;

            SetLoopModePara(loop, AscendC::DataCopyMVType::OUT_TO_UB);

            // 在大小为[c1,th4,tw4,c0]的buf里面从搬入大小为[c,aligned16(hw)]大小的数据
            // 由于c1c0 >= c
            // 且根据数学推到[th4,tw4] >= align16(hw)
            // 所以当前buf一定能放得下所有数据

            AscendC::DataCopyPad<T, AscendC::PaddingMode::Compact>(
                buf,
                gm_[box.c.idx * srcH_ * srcW_ + src.hIdx * srcW_ + src.wIdx + batchOffset],
                params,
                {false, 0, 0, 0});
            AscendC::ResetLoopModePara(AscendC::DataCopyMVType::OUT_TO_UB);
        }
    }

    __aicore__ inline void Compute(
        AscendC::LocalTensor<T>& mainBuf,
        AscendC::LocalTensor<T>& transposeBuf,
        const TileBox& box) const
    {
        uint16_t c1 = box.c.c1;
        uint32_t tileBufW = TileUnfoldSize(box.tile.wLength) + TILE_BUF_BANK_CONFLICT_PADDING;
        uint32_t tileBufH = TileUnfoldSize(box.tile.hLength);
        uint32_t tileBufHW = tileBufW * tileBufH;

        if (const HWBox& src = box.src; src.elements != 0) {
            uint32_t fmapHWAligned16 = Ops::Base::CeilAlign(src.elements, HW_SRC_ALIGNED_16);

            if (const uint32_t tailC0 = c1 * C0<T>() - box.c.length; tailC0 != 0) {
                //c轴不是c0对齐的,给他补0补到c0对齐
                AscendC::Duplicate(
                    mainBuf[box.c.length * fmapHWAligned16],
                    static_cast<T>(0),
                    tailC0 * fmapHWAligned16);
            }

            uint32_t padTopOffset = box.pad.hTop * src.wLength * C0<T>();
            uint32_t srcC1Stride = fmapHWAligned16 * C0<T>();
            uint32_t tileUnfoldC1Stride = tileBufHW * C0<T>();

            const auto params = UnfoldPolicy::InitUnfoldParams(box);
            const typename UnfoldPolicy::UnfoldColParamsT& ucp = AscendC::Std::get<0>(params);
            const typename UnfoldPolicy::UnfoldRowParamsT& urp = AscendC::Std::get<1>(params);

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
                AscendC::LocalTensor<T> tileBuf = mainBuf[c1Idx * tileUnfoldC1Stride];
                UnfoldFromSrcBufVf(
                    reinterpret_cast<__ubuf__ T*>(tileBuf.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ T*>(transposeBuf.GetPhyAddr()),
                    ucp, urp);
            }
        } else {
            //整个tile都由padding区域产生,不做计算直接置0,
            AscendC::Duplicate(
                mainBuf,
                static_cast<T>(0),
                tileBufHW * c1 * C0<T>());
        }
    }

    __aicore__ inline void CopyOut(
        AscendC::LocalTensor<T>& mainBuf,
        AscendC::LocalTensor<T>& out,
        const TileBox& box) const
    {
        AscendC::DataCopyParams params;
        params.blockCount = TileUnfoldSize(box.tile.hLength) * box.c.c1;
        params.blockLen = TileUnfoldSize(box.tile.wLength);
        //拷贝时忽略右侧1 pad
        params.srcGap = TILE_BUF_BANK_CONFLICT_PADDING;
        params.dstGap = 0;
        AscendC::DataCopy(out, mainBuf, params);
    }

private:
    __simd_vf__ static inline void UnfoldFromSrcBufVf(
        __ubuf__ T* __restrict__ tileBuf,
        __ubuf__ T* __restrict__ srcBuf,
        const typename UnfoldPolicy::UnfoldColParamsT ucp,
        const typename UnfoldPolicy::UnfoldRowParamsT urp)
    {
        UnfoldPolicy::UnfoldColsFromSrcBufVf(tileBuf, srcBuf, ucp);

        AscendC::MicroAPI::LocalMemBar<
            AscendC::MicroAPI::MemType::VEC_LOAD,
            AscendC::MicroAPI::MemType::VEC_STORE>();

        UnfoldPolicy::UnfoldRowsVf(tileBuf, urp);
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

namespace UnfoldImpl {
using namespace AscendC::MicroAPI;

struct DefaultUnfoldRowParams {
    uint32_t hValidElements;
    uint32_t wLoadOffset;
    uint32_t tileBufWidthBlocks;
    uint16_t hRepeatTimes;
    uint16_t tileW;
};

struct DefaultUnfoldColParams {
    uint32_t wValidElements;
    uint32_t wStoreOffset;
    uint32_t tileBufWidth;
    uint16_t wRepeatTimes;
    uint16_t tileH;
};

template <typename T>
static inline __aicore__ void InitDefaultUnfoldParams(
    const TileBox& box,
    DefaultUnfoldColParams& ucp,
    DefaultUnfoldRowParams& urp)
{
    uint32_t tileWSize = TileUnfoldSize(box.tile.wLength);
    uint32_t tileBufWidthBlocks = tileWSize + TILE_BUF_BANK_CONFLICT_PADDING;

    ucp.wValidElements = box.src.wLength * C0<T>();
    ucp.tileBufWidth = tileBufWidthBlocks * C0<T>();
    //行展开后放在tileBuf的右侧,
    ucp.wStoreOffset = (tileWSize - box.src.wLength - box.pad.wRight) * C0<T>();
    ucp.wRepeatTimes = Ops::Base::CeilDiv(ucp.wValidElements, VL<T>());
    ucp.tileH = box.tile.hLength;

    urp.hValidElements = TileUnfoldSize(box.tile.hLength) * C0<T>();
    urp.tileBufWidthBlocks = tileBufWidthBlocks;
    //读取的起始位置,在行展开的写入位置在往前走padLeft的大小
    urp.wLoadOffset = ucp.wStoreOffset - box.pad.wLeft * C0<T>();
    urp.hRepeatTimes = Ops::Base::CeilDiv(urp.hValidElements, VL<T>());
    urp.tileW = box.tile.wLength;
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

    static __simd_callee__ inline void UnfoldColsFromSrcBufVf(
        __ubuf__ T* __restrict__ tileBuf,
        __ubuf__ T* __restrict__ dyBuf,
        const DefaultUnfoldColParams& params)
    {
        const uint32_t wValidElements = params.wValidElements;
        const uint32_t wStoreOffset = params.wStoreOffset;
        const uint32_t tileBufWidth = params.tileBufWidth;
        const uint16_t wRepeatTimes = params.wRepeatTimes;
        const uint16_t tileH = params.tileH;

        __ubuf__ T* dst0 = tileBuf + wStoreOffset;
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
        __ubuf__ T* buf,
        const DefaultUnfoldRowParams& params)
    {
        const uint32_t tileBufWidthBlocks = params.tileBufWidthBlocks;
        const uint32_t hValidElements = params.hValidElements;
        const uint32_t wLoadOffset = params.wLoadOffset;
        const uint16_t hRepeatTimes = params.hRepeatTimes;
        const uint16_t tileW = params.tileW;

        __ubuf__ T* src0 = buf + wLoadOffset;

        uint32_t maskValue = hValidElements;
        for (uint16_t i = 0; i < hRepeatTimes; i++) {
            RegTensor<T> s0;
            RegTensor<T> s1;

            RegTensor<T> d0;
            RegTensor<T> d1;
            RegTensor<T> d2;
            RegTensor<T> d3;

            MaskReg mask = UpdateMask<T>(maskValue);
            const uint32_t hOffset = tileBufWidthBlocks * i * VL<T>();

            __ubuf__ T* src = src0 + hOffset;
            __ubuf__ T* dst = buf + hOffset;

            for (uint16_t th = 0; th < tileW; th++) {
                LoadAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    s0, src, tileBufWidthBlocks, 1, mask);
                LoadAlign<T, DataCopyMode::DATA_BLOCK_COPY>(
                    s1, src, tileBufWidthBlocks, mask);

                TransformVf(s0, s1, d0, d1, d2, mask);

                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    dst, s0, tileBufWidthBlocks, 1, mask);
                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    dst, d0, tileBufWidthBlocks, 1, mask);
                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    dst, d1, tileBufWidthBlocks, 1, mask);
                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY>(
                    dst, d2, tileBufWidthBlocks, mask);
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

    static __simd_callee__ inline void UnfoldColsFromSrcBufVf(
        __ubuf__ T* __restrict__ tileBuf,
        __ubuf__ T* __restrict__ fmapBuf,
        const UnfoldFmapColParams& params)
    {
        const uint32_t wValidElements = params.wValidElements;
        const uint32_t wStoreOffset = params.wStoreOffset;
        const uint32_t tileBufWidth = params.tileBufWidth;
        const uint16_t wRepeatTimes = params.wRepeatTimes;
        const uint16_t tileHMainRepeatTimes = params.tileHMainRepeatTimes;
        const uint16_t tileHTailRepeatTimes = params.tileHTailRepeatTimes;

        __ubuf__ T* dst0 = tileBuf + wStoreOffset;

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
        __ubuf__ T* buf,
        const UnfoldFmapRowParams& params)
    {
        const uint32_t tileBufWidthBlocks = params.tileBufWidthBlocks;
        const uint32_t wLoadOffset = params.wLoadOffset;
        const uint32_t hValidElements = params.hValidElements;
        const uint16_t hRepeatTimes = params.hRepeatTimes;
        const uint16_t tileWMainRepeatTimes = params.tileWMainRepeatTimes;
        const uint16_t tileWTailRepeatTimes = params.tileWTailRepeatTimes;

        //fmap靠在整块buf的右侧,所以读取时需要加个左边的偏移
        __ubuf__ T* src0 = buf + wLoadOffset;

        uint32_t maskValue = hValidElements;
        for (uint16_t i = 0; i < hRepeatTimes; i++) {
            RegTensor<T> s0;
            RegTensor<T> s1;
            RegTensor<T> s2;
            RegTensor<T> s3;

            RegTensor<T> d0;
            RegTensor<T> d1;
            RegTensor<T> d2;
            RegTensor<T> d3;

            MaskReg mask = UpdateMask<T>(maskValue);
            const uint32_t hOffset = tileBufWidthBlocks * i * VL<T>();

            __ubuf__ T* src = src0 + hOffset;

            LoadAlign<T, DataCopyMode::DATA_BLOCK_COPY,
                PostLiteral::POST_MODE_UPDATE>(
                s0, src, tileBufWidthBlocks, 1, mask);
            LoadAlign<T, DataCopyMode::DATA_BLOCK_COPY,
                PostLiteral::POST_MODE_UPDATE>(
                s1, src, tileBufWidthBlocks, 1, mask);

            __ubuf__ T* dst = buf + hOffset;

            for (uint16_t tw = 0; tw < tileWMainRepeatTimes; tw++) {
                LoadAlign<T, DataCopyMode::DATA_BLOCK_COPY,
                    PostLiteral::POST_MODE_UPDATE>(
                    s2, src, tileBufWidthBlocks, 1, mask);
                LoadAlign<T, DataCopyMode::DATA_BLOCK_COPY,
                    PostLiteral::POST_MODE_UPDATE>(
                    s3, src, tileBufWidthBlocks, 1, mask);

                TransformVf(s0, s1, s2, s3, d0, d1, d2, d3, mask);

                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY,
                    PostLiteral::POST_MODE_UPDATE>(
                    dst, d0, tileBufWidthBlocks, 1, mask);
                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY,
                    PostLiteral::POST_MODE_UPDATE>(
                    dst, d1, tileBufWidthBlocks, 1, mask);
                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY,
                    PostLiteral::POST_MODE_UPDATE>(
                    dst, d2, tileBufWidthBlocks, 1, mask);
                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY,
                    PostLiteral::POST_MODE_UPDATE>(
                    dst, d3, tileBufWidthBlocks, 1, mask);

                LoadAlign<T, DataCopyMode::DATA_BLOCK_COPY,
                    PostLiteral::POST_MODE_UPDATE>(
                    s0, src, tileBufWidthBlocks, 1, mask);
                LoadAlign<T, DataCopyMode::DATA_BLOCK_COPY,
                    PostLiteral::POST_MODE_UPDATE>(
                    s1, src, tileBufWidthBlocks, 1, mask);

                TransformVf(s2, s3, s0, s1, d0, d1, d2, d3, mask);
                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY,
                    PostLiteral::POST_MODE_UPDATE>(
                    dst, d0, tileBufWidthBlocks, 1, mask);
                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY,
                    PostLiteral::POST_MODE_UPDATE>(
                    dst, d1, tileBufWidthBlocks, 1, mask);
                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY,
                    PostLiteral::POST_MODE_UPDATE>(
                    dst, d2, tileBufWidthBlocks, 1, mask);
                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY,
                    PostLiteral::POST_MODE_UPDATE>(
                    dst, d3, tileBufWidthBlocks, 1, mask);
            }

            for (uint16_t th = 0; th < tileWTailRepeatTimes; th++) {
                LoadAlign<T, DataCopyMode::DATA_BLOCK_COPY,
                    PostLiteral::POST_MODE_UPDATE>(
                    s2, src, tileBufWidthBlocks, 1, mask);
                LoadAlign<T, DataCopyMode::DATA_BLOCK_COPY,
                    PostLiteral::POST_MODE_UPDATE>(
                    s3, src, tileBufWidthBlocks, 1, mask);

                TransformVf(s0, s1, s2, s3, d0, d1, d2, d3, mask);

                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY,
                    PostLiteral::POST_MODE_UPDATE>(
                    dst, d0, tileBufWidthBlocks, 1, mask);
                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY,
                    PostLiteral::POST_MODE_UPDATE>(
                    dst, d1, tileBufWidthBlocks, 1, mask);
                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY,
                    PostLiteral::POST_MODE_UPDATE>(
                    dst, d2, tileBufWidthBlocks, 1, mask);
                StoreAlign<T, DataCopyMode::DATA_BLOCK_COPY,
                    PostLiteral::POST_MODE_UPDATE>(
                    dst, d3, tileBufWidthBlocks, 1, mask);
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
using WinoFmapTransformer = WinoTransformer<T, 2, 4, UnfoldImpl::Fmap<T>, true>;

template <typename T>
using WinoDyTransformer = WinoTransformer<T, 2, 2, UnfoldImpl::Dy<T>, false>;

#endif //CONV_BP_WINO_TRANSFORM_H