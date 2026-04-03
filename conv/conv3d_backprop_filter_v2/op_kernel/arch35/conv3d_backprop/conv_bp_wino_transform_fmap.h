// /**
// * Copyright (c) 2025 Huawei Technologies Co., Ltd.
//  * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
//  * CANN Open Software License Agreement Version 2.0 (the "License").
//  * Please refer to the License for details. You may not use this file except in compliance with the License.
//  * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
//  * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
//  * See LICENSE in the root of the software repository for the full text of the License.
//  */
//
// /*!
//  * \file conv_bp_wino_transform_fmap.h
//  * \brief
//  */
//
//
// #ifndef CONV_BP_WINO_TRANSFORM_FMAP_H
// #define CONV_BP_WINO_TRANSFORM_FMAP_H
//
// #include "conv_bp_wino_util.h"
//
// using namespace AscendC;
//
// //fmap在搬运时hw轴按照16元素对齐
// //这里主要是为了方便Transpose5HD在float32做转置
// static constexpr uint32_t FMAP_HW_ALIGNED_16 = 16;
//
// using F23FmapTile = SlideWindows<2, 4>;
//
// template <typename T>
// class WinoFmapTransformer {
// public:
//     __aicore__ inline WinoFmapTransformer(
//         __gm__ T* gm,
//         const uint32_t cin,
//         const uint32_t hi,
//         const uint32_t wi,
//         const uint16_t padH,
//         const uint16_t padW)
//         : cin_(cin),
//           hi_(hi),
//           wi_(wi),
//           padH_(padH),
//           padW_(padW)
//     {
//     }
//
//     struct TileBox {
//         HWBox tile;
//         HWBox fmap;
//         HWPad pad;
//     };
//
//
//     __aicore__ inline void CopyOut(
//         LocalTensor<T>& buf,
//         LocalTensor<T>& transposeBuf,
//         const TileBox& box,
//         uint16_t cLength,
//         GlobalTensor<T>& out)
//     {
//         DataCopy(out, buf, buf.GetSize());
//         // DataCopy(out, transposeBuf, transposeBuf.GetSize());
//     }
//
//
//     __aicore__ inline TileBox CopyIn(
//         const LocalTensor<T>& buf,
//         const HWBox& tile,
//         uint32_t batchIdx,
//         uint32_t cIdx,
//         uint32_t cLength) const
//     {
//         TileBox box = {tile, {}, {}};
//         F23FmapTile::CalculateSrcBox(
//             box.tile, hi_, wi_, padH_, padW_,
//             box.fmap, box.pad);
//
//         if (const HWBox& fmap = box.fmap; fmap.elements != 0) {
//             DataCopyExtParams params;
//             params.blockCount = fmap.hLength;
//             params.blockLen = fmap.wLength * sizeof(T);
//             params.srcStride = wi_ * sizeof(T);
//             params.dstStride = 0;
//
//             uint32_t fmapHWAligned16 = Ops::Base::CeilAlign(fmap.elements, FMAP_HW_ALIGNED_16);
//
//             LoopModeParams loop;
//             loop.loop1Size = cLength;
//             loop.loop1SrcStride = wi_ * hi_ * sizeof(T);
//             loop.loop1DstStride = fmapHWAligned16 * sizeof(T);
//             loop.loop2Size = 1;
//             loop.loop1SrcStride = 0;
//             loop.loop2DstStride = 0;
//
//             SetLoopModePara(loop, DataCopyMVType::OUT_TO_UB);
//
//             // 在大小为[c1,th4,tw4,c0]的buf里面从搬入大小为[c,aligned16(hw)]大小的fmap
//             // 由于c1c0 >= c
//             // 且根据数学推到[th4,tw4] >= align16(hw)
//             // 所以当前buf一定能放得下所有数据
//
//             DataCopyPad<T, PaddingMode::Compact>(
//                 buf,
//                 fmapGm_[(batchIdx * cin_ + cIdx) * hi_ * wi_ + fmap.hIdx * wi_ + fmap.wIdx],
//                 params,
//                 {false, 0, 0, 0});
//             ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
//         }
//
//         return box;
//     }
//
//
//     __aicore__ inline void Compute(
//         LocalTensor<T>& mainBuf,
//         LocalTensor<T>& transposeBuf,
//         uint32_t cLength,
//         const TileBox& box) const
//     {
//         uint32_t c1 = Ops::Base::CeilAlign(cLength, C0<T>());
//         if (const HWBox& fmap = box.fmap; fmap.elements != 0) {
//             uint32_t fmapHWAligned16 = Ops::Base::CeilAlign(fmap.elements, FMAP_HW_ALIGNED_16);
//
//             if (uint32_t tailC0 = c1 * C0<T>() - cLength; tailC0 != 0) {
//                 //cLength不是c0对齐的,给他补0补到c0对齐
//                 Duplicate(
//                     mainBuf[cLength * fmapHWAligned16],
//                     static_cast<T>(0),
//                     tailC0 * fmapHWAligned16);
//             }
//
//             uint32_t tileBufWidthC0Blocks = TileUnfoldSize(box.tile.wLength);
//             uint32_t tileBufWidth = tileBufWidthC0Blocks * C0<T>();
//
//             UnfoldRowParamsV2 urp = {};
//             urp.fmapWidth = fmap.wLength * C0<T>();
//             //行展开后放在tileBuf的右侧,
//             urp.wStoreOffset = tileBufWidth - (fmap.wLength + box.pad.wRight) * C0<T>();
//             urp.wRepeatTimes = Ops::Base::CeilDiv(urp.fmapWidth, VL<T>());
//             urp.tileHTailRepeatTimes = box.tile.hLength & 1;
//             urp.tileHMainRepeatTimes = box.tile.hLength >> 1;
//
//             uint32_t fmapLeftBoundOffset = (box.tile.wLength * 2 - 2) * C0<T>();
//
//             UnfoldColParamsV2 ucp = {};
//             ucp.hValidElements = TileUnfoldSize(box.tile.hLength) * C0<T>();
//             ucp.fmapLeftBoundOffset = fmapLeftBoundOffset;
//             ucp.hRepeatTimes = Ops::Base::CeilDiv(ucp.hValidElements, VL<T>());
//             ucp.tileWTailRepeatTimes = box.tile.wLength & 1;
//             ucp.tileWMainRepeatTimes = box.tile.wLength >> 1;
//
//             uint32_t tileBufC1Stride = TileUnfoldElements(box.tile.elements) * C0<T>();
//             uint32_t fmapAligned16C1Stride = fmapHWAligned16 * C0<T>();
//             for (uint32_t i = 0; i < c1; i++) {
//                 //从末端开始处理，由于mainBuf的头部搬入了整块fmap
//                 //如果顺序处理,那可能在变换第一个c1的fmap时,污染了第二个c1的fmap
//                 //逆序处理就没这个问题,因为尾部不会连着其他fmap
//                 uint32_t c1Idx = c1 - i - 1;
//
//                 TransposeCHW2C1HWC0(
//                     mainBuf[c1Idx * fmapAligned16C1Stride],
//                     transposeBuf[box.pad.hTop * box.fmap.wLength * C0<T>()],
//                     fmapHWAligned16);
//                 //TODO 处理pad
//                 LocalTensor<T> tileBuf = mainBuf[c1Idx * tileBufC1Stride];
//                 //行变换
//                 UnfoldFromFmapBufVf(
//                     reinterpret_cast<__ubuf__ T*>(tileBuf.GetPhyAddr()),
//                     reinterpret_cast<__ubuf__ T*>(transposeBuf.GetPhyAddr()),
//                     tileBufWidth,
//                     tileBufWidthC0Blocks,
//                     urp, ucp);
//             }
//         } else {
//             //整个tile都由padding区域产生,不做计算直接置0,
//             Duplicate(
//                 mainBuf,
//                 static_cast<T>(0),
//                 TileUnfoldElements(box.tile.elements) * c1 * C0<T>());
//         }
//     }
//
// private:
//     __aicore__ static inline void TransposeCHW2C1HWC0(
//         const LocalTensor<T>& srcBuf,
//         const LocalTensor<T>& dstBuf,
//         uint32_t fmapHWAligned16)
//     {
//         uint64_t srcList[16];
//         uint64_t dstList[16];
//
//         if constexpr (sizeof(T) == 2) {
// #pragma unroll
//             for (uint32_t i = 0; i < 16; i++) {
//                 uint32_t s = i * fmapHWAligned16;
//                 uint32_t d = i * 16;
//                 srcList[i] = reinterpret_cast<uint64_t>(srcBuf[s].GetPhyAddr());
//                 dstList[i] = reinterpret_cast<uint64_t>(dstBuf[d].GetPhyAddr());
//             }
//             TransDataTo5HDParams params;
//             params.repeatTimes = fmapHWAligned16 / 16;
//             params.srcRepStride = params.repeatTimes == 1 ? 0 : 1;
//             params.dstRepStride = params.repeatTimes == 1 ? 0 : 16;
//             TransDataTo5HD<T>(dstList, srcList, params);
//         } else if constexpr (sizeof(T) == 4) {
// #pragma unroll
//             for (uint32_t i = 0; i < 8; i++) {
//                 uint32_t s = i * fmapHWAligned16;
//                 uint32_t d = i * 8;
//                 srcList[i] = reinterpret_cast<uint64_t>(srcBuf[s].GetPhyAddr());
//                 srcList[i + 8] = reinterpret_cast<uint64_t>(srcBuf[s + 8].GetPhyAddr());
//                 dstList[i * 2] = reinterpret_cast<uint64_t>(dstBuf[d].GetPhyAddr());
//                 dstList[i * 2 + 1] = reinterpret_cast<uint64_t>(dstBuf[d + 8 * 8].GetPhyAddr());
//             }
//
//             TransDataTo5HDParams params;
//             params.repeatTimes = fmapHWAligned16 / 16;
//             params.srcRepStride = params.repeatTimes == 1 ? 0 : 2;
//             params.dstRepStride = params.repeatTimes == 1 ? 0 : 16;
//             TransDataTo5HD<T>(dstList, srcList, params);
//         }
//     }
//
//     struct UnfoldColParamsV2 {
//         uint32_t hValidElements;
//         uint32_t fmapLeftBoundOffset;
//         uint16_t hRepeatTimes;
//         uint16_t tileWMainRepeatTimes;
//         uint16_t tileWTailRepeatTimes;
//     };
//
//     struct UnfoldRowParamsV2 {
//         uint32_t fmapWidth;
//         uint32_t wStoreOffset;
//         uint16_t wRepeatTimes;
//         uint16_t tileHMainRepeatTimes;
//         uint16_t tileHTailRepeatTimes;
//     };
//
//     static __simd_vf__ inline void UnfoldFromFmapBufVf(
//         __ubuf__ T* __restrict__ tileBuf,
//         __ubuf__ T* __restrict__ fmapBuf,
//         const uint32_t tileBufWidth,
//         const uint16_t tileBufWidthC0Blocks,
//         const UnfoldRowParamsV2 urp,
//         const UnfoldColParamsV2 ucp)
//     {
//         __ubuf__ T* t = tileBuf + urp.wStoreOffset;
//         UnfoldRowsFromFmapBufVf(t, fmapBuf, tileBufWidth, urp);
//
//         MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_LOAD, MicroAPI::MemType::VEC_STORE>();
//
//         UnfoldColsVf(tileBuf, tileBufWidthC0Blocks, ucp);
//     }
//
//     // 执行行变换
//     // 原始数据:
//     //
//     //      w0 w1 w2 w3 w4 w5
//     //     +--+--+--+--+--+--+
//     //  h0 |xx|xx|xx|xx|xx|xx|
//     //     +--+--+--+--+--+--+
//     //  h1 |xx|xx|xx|xx|xx|xx|
//     //     +--+--+--+--+--+--+
//     //  h2 |xx|xx|xx|xx|xx|xx|
//     //     +--+--+--+--+--+--+
//     //  h3 |xx|xx|xx|xx|xx|xx|
//     //     +--+--+--+--+--+--+
//     //  h4 |xx|xx|xx|xx|xx|xx|
//     //     +--+--+--+--+--+--+
//     //  h5 |xx|xx|xx|xx|xx|xx|
//     //     +--+--+--+--+--+--+
//     //
//     // 变换后内存:
//     //
//     //           w0         w1    w2-w4     w5     w6 w7
//     //      +---------+---------+-------+---------+--+--+
//     //  Th0 |h0w0-h2w0|h0w1-h2w1|.......|h0w5-h2w5|  |  |
//     //      +---------+---------+-------+---------+--+--+
//     //  Th1 |h1w0+h2w0|h1w1+h2w1|.......|h1w5+h2w5|  |  |
//     //      +---------+---------+-------+---------+--+--+
//     //  Th2 |h2w0-h1w0|h2w1-h1w1|.......|h2w5-h1w5|  |  |
//     //      +---------+---------+-------+---------+--+--+
//     //  Th3 |h1w0-h3w0|h1w1-h3w1|.......|h1w5-h3w5|  |  |
//     //      +---------+---------+-------+---------+--+--+
//     //  Th4 |h2w0-h4w0|h2w1-h4w1|.......|h2w5-h4w5|  |  |
//     //      +---------+---------+-------+---------+--+--+
//     //  Th5 |h4w0+h3w0|h4w1+h3w1|.......|h4w5+h3w5|  |  |
//     //      +---------+---------+-------+---------+--+--+
//     //  Th6 |h4w0-h3w0|h4w1-h3w1|.......|h4w5-h3w5|  |  |
//     //      +---------+---------+-------+---------+--+--+
//     //  Th7 |h3w0-h5w0|h3w1-h5w1|.......|h3w5-h5w5|  |  |
//     //      +---------+---------+-------+---------+--+--+
//     // 数据在H方向上展开后写入
//     //
//     static __simd_callee__ inline void UnfoldRowsFromFmapBufVf(
//         __ubuf__ T* __restrict__ tileBuf,
//         __ubuf__ T* __restrict__ fmapBuf,
//         const uint32_t tileBufWidth,
//         const UnfoldRowParamsV2& params)
//     {
//         const uint32_t fmapWidth = params.fmapWidth;
//         const uint16_t tileHMainRepeatTimes = params.tileHMainRepeatTimes;
//         const uint16_t tileHTailRepeatTimes = params.tileHTailRepeatTimes;
//         const uint16_t wRepeatTimes = params.wRepeatTimes;
//
//         uint32_t maskValue = fmapWidth;
//         for (uint16_t i = 0; i < wRepeatTimes; i++) {
//             MicroAPI::MaskReg mask = MicroAPI::UpdateMask<T>(maskValue);
//             MicroAPI::RegTensor<T> s0;
//             MicroAPI::RegTensor<T> s1;
//             MicroAPI::RegTensor<T> s2;
//             MicroAPI::RegTensor<T> s3;
//
//             MicroAPI::RegTensor<T> d0;
//             MicroAPI::RegTensor<T> d1;
//             MicroAPI::RegTensor<T> d2;
//             MicroAPI::RegTensor<T> d3;
//
//             // 从最上方的tile开始滑窗
//             // 先读取fmap首2行,每次循环往下读2行凑成4行执行变换
//             // 但若一个滑窗在fmap的1-4行分别读入s0,s1,s2,s3
//             // 在下一个滑窗s2,s3就变成1-2行,不考虑重新读取的话2-3行就只能读入s0,s1,滑窗1-4行就变成s2,s3,s0,s1
//             // 如果将s0,s1的数据拷贝到s2,s3可能会产生多余的指令并且产生数据依赖降低性能
//             // vf内scalar能力较弱不确定怎么样的代码编译器能正常优化
//             // 所以这里按照最朴素的方式展开循环一个循环内处理2个连续滑窗,
//             // 如果滑窗为奇数,则通过tileHTailRepeatTimes额外执行一次滑窗
//
//             //循环fmapW
//             const uint32_t wOffset = i * VL<T>();
//
//             __ubuf__ T* src = fmapBuf + wOffset;
//             MicroAPI::LoadAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(s0, src, fmapWidth);
//             MicroAPI::LoadAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(s1, src, fmapWidth);
//
//             __ubuf__ T* dst = tileBuf + wOffset;
//             for (uint16_t th = 0; th < tileHMainRepeatTimes; th++) {
//                 MicroAPI::LoadAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(s2, src, fmapWidth);
//                 MicroAPI::LoadAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(s3, src, fmapWidth);
//
//                 TransformVf(s0, s1, s2, s3, d0, d1, d2, d3, mask);
//
//                 MicroAPI::StoreAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(dst, d0, tileBufWidth, mask);
//                 MicroAPI::StoreAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(dst, d1, tileBufWidth, mask);
//                 MicroAPI::StoreAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(dst, d2, tileBufWidth, mask);
//                 MicroAPI::StoreAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(dst, d3, tileBufWidth, mask);
//
//                 MicroAPI::LoadAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(s0, src, fmapWidth);
//                 MicroAPI::LoadAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(s1, src, fmapWidth);
//
//                 TransformVf(s2, s3, s0, s1, d0, d1, d2, d3, mask);
//
//                 MicroAPI::StoreAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(dst, d0, tileBufWidth, mask);
//                 MicroAPI::StoreAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(dst, d1, tileBufWidth, mask);
//                 MicroAPI::StoreAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(dst, d2, tileBufWidth, mask);
//                 MicroAPI::StoreAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(dst, d3, tileBufWidth, mask);
//             }
//
//             for (uint16_t th = 0; th < tileHTailRepeatTimes; th++) {
//                 MicroAPI::LoadAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(s2, src, fmapWidth);
//                 MicroAPI::LoadAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(s3, src, fmapWidth);
//
//                 TransformVf(s0, s1, s2, s3, d0, d1, d2, d3, mask);
//
//                 MicroAPI::StoreAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(dst, d0, tileBufWidth, mask);
//                 MicroAPI::StoreAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(dst, d1, tileBufWidth, mask);
//                 MicroAPI::StoreAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(dst, d2, tileBufWidth, mask);
//                 MicroAPI::StoreAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(dst, d3, tileBufWidth, mask);
//             }
//         }
//     }
//
//     // 在行变换的结果上执行变换
//     // 原始布局S:
//     //
//     //      w0 w1 w2 w3 w4 w5 w6 w7
//     //     +--+--+--+--+--+--+--+--+
//     //  h0 |  |  |xx|xx|xx|xx|xx|xx|
//     //     +--+--+--+--+--+--+--+--+
//     //  h1 |  |  |xx|xx|xx|xx|xx|xx|
//     //     +--+--+--+--+--+--+--+--+
//     //  h2 |  |  |xx|xx|xx|xx|xx|xx|
//     //     +--+--+--+--+--+--+--+--+
//     //  h3 |  |  |xx|xx|xx|xx|xx|xx|
//     //     +--+--+--+--+--+--+--+--+
//     //  h4 |  |  |xx|xx|xx|xx|xx|xx|
//     //     +--+--+--+--+--+--+--+--+
//     //  h5 |  |  |xx|xx|xx|xx|xx|xx|
//     //     +--+--+--+--+--+--+--+--+
//     //  h6 |  |  |xx|xx|xx|xx|xx|xx|
//     //     +--+--+--+--+--+--+--+--+
//     //  h7 |  |  |xx|xx|xx|xx|xx|xx|
//     //     +--+--+--+--+--+--+--+--+
//     //
//     // 变换后内存:
//     //
//     //           w0        w1     w2-w6     w7
//     //      +---------+---------+-------+---------+
//     //  Th0 |h0w2-h0w4|h0w3+h0w4|.......|h0w5-h0w7|
//     //      +---------+---------+-------+---------+
//     //  Th1 |h1w2-h1w4|h1w3+h1w4|.......|h1w5-h1w7|
//     //      +---------+---------+-------+---------+
//     //  Th2 |h2w2-h2w4|h2w3+h2w4|.......|h2w5-h2w7|
//     //      +---------+---------+-------+---------+
//     //  Th3 |h3w2-h3w4|h3w3+h3w4|.......|h3w5-h3w7|
//     //      +---------+---------+-------+---------+
//     //  Th4 |h4w2-h4w4|h4w3+h4w4|.......|h4w5-h4w7|
//     //      +---------+---------+-------+---------+
//     //  Th5 |h5w2-h5w4|h5w3+h5w4|.......|h5w5-h5w7|
//     //      +---------+---------+-------+---------+
//     //  Th6 |h6w2-h6w4|h6w3+h6w4|.......|h6w5-h6w7|
//     //      +---------+---------+-------+---------+
//     //  Th7 |h7w2-h7w4|h7w3+h7w4|.......|h7w5-h7w7|
//     //      +---------+---------+-------+---------+
//     //
//
//     static __simd_callee__ inline void UnfoldColsVf(
//         __ubuf__ T* buf,
//         const uint32_t tileBufWidthC0Blocks,
//         const UnfoldColParamsV2& params)
//     {
//         const uint32_t hValidElements = params.hValidElements;
//         const uint16_t hRepeatTimes = params.hRepeatTimes;
//         const uint16_t tileWMainRepeatTimes = params.tileWMainRepeatTimes;
//         const uint16_t tileWTailRepeatTimes = params.tileWTailRepeatTimes;
//
//
//         __ubuf__ T* src0 = buf + params.fmapLeftBoundOffset;
//         uint32_t maskValue = hValidElements;
//         for (uint16_t i = 0; i < hRepeatTimes; i++) {
//             MicroAPI::RegTensor<T> s0;
//             MicroAPI::RegTensor<T> s1;
//             MicroAPI::RegTensor<T> s2;
//             MicroAPI::RegTensor<T> s3;
//
//             MicroAPI::RegTensor<T> d0;
//             MicroAPI::RegTensor<T> d1;
//             MicroAPI::RegTensor<T> d2;
//             MicroAPI::RegTensor<T> d3;
//
//             MicroAPI::MaskReg mask = MicroAPI::UpdateMask<T>(maskValue);
//             const uint32_t hOffset = tileBufWidthC0Blocks * i * VL<T>();
//
//             __ubuf__ T* src = src0 + hOffset;
//
//             MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
//                 MicroAPI::PostLiteral::POST_MODE_UPDATE>(
//                 s0, src, tileBufWidthC0Blocks, 1, mask);
//             MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
//                 MicroAPI::PostLiteral::POST_MODE_UPDATE>(
//                 s1, src, tileBufWidthC0Blocks, 1, mask);
//
//             __ubuf__ T* dst = buf + hOffset;
//
//             for (uint16_t tw = 0; tw < tileWMainRepeatTimes; tw++) {
//                 MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
//                     MicroAPI::PostLiteral::POST_MODE_UPDATE>(
//                     s2, src, tileBufWidthC0Blocks, 1, mask);
//                 MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
//                     MicroAPI::PostLiteral::POST_MODE_UPDATE>(
//                     s3, src, tileBufWidthC0Blocks, 1, mask);
//
//                 TransformVf(s0, s1, s2, s3, d0, d1, d2, d3, mask);
//
//                 MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
//                     MicroAPI::PostLiteral::POST_MODE_UPDATE>(
//                     dst, d0, tileBufWidthC0Blocks, 1, mask);
//                 MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
//                     MicroAPI::PostLiteral::POST_MODE_UPDATE>(
//                     dst, d1, tileBufWidthC0Blocks, 1, mask);
//                 MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
//                     MicroAPI::PostLiteral::POST_MODE_UPDATE>(
//                     dst, d2, tileBufWidthC0Blocks, 1, mask);
//                 MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
//                     MicroAPI::PostLiteral::POST_MODE_UPDATE>(
//                     dst, d3, tileBufWidthC0Blocks, 1, mask);
//
//                 MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
//                     MicroAPI::PostLiteral::POST_MODE_UPDATE>(
//                     s0, src, tileBufWidthC0Blocks, 1, mask);
//                 MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
//                     MicroAPI::PostLiteral::POST_MODE_UPDATE>(
//                     s1, src, tileBufWidthC0Blocks, 1, mask);
//
//                 TransformVf(s2, s3, s0, s1, d0, d1, d2, d3, mask);
//                 MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
//                     MicroAPI::PostLiteral::POST_MODE_UPDATE>(
//                     dst, d0, tileBufWidthC0Blocks, 1, mask);
//                 MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
//                     MicroAPI::PostLiteral::POST_MODE_UPDATE>(
//                     dst, d1, tileBufWidthC0Blocks, 1, mask);
//                 MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
//                     MicroAPI::PostLiteral::POST_MODE_UPDATE>(
//                     dst, d2, tileBufWidthC0Blocks, 1, mask);
//                 MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
//                     MicroAPI::PostLiteral::POST_MODE_UPDATE>(
//                     dst, d3, tileBufWidthC0Blocks, 1, mask);
//             }
//
//             for (uint16_t th = 0; th < tileWTailRepeatTimes; th++) {
//                 MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
//                     MicroAPI::PostLiteral::POST_MODE_UPDATE>(
//                     s2, src, tileBufWidthC0Blocks, 1, mask);
//                 MicroAPI::LoadAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
//                     MicroAPI::PostLiteral::POST_MODE_UPDATE>(
//                     s3, src, tileBufWidthC0Blocks, 1, mask);
//
//                 TransformVf(s0, s1, s2, s3, d0, d1, d2, d3, mask);
//
//                 MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
//                     MicroAPI::PostLiteral::POST_MODE_UPDATE>(
//                     dst, d0, tileBufWidthC0Blocks, 1, mask);
//                 MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
//                     MicroAPI::PostLiteral::POST_MODE_UPDATE>(
//                     dst, d1, tileBufWidthC0Blocks, 1, mask);
//                 MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
//                     MicroAPI::PostLiteral::POST_MODE_UPDATE>(
//                     dst, d2, tileBufWidthC0Blocks, 1, mask);
//                 MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
//                     MicroAPI::PostLiteral::POST_MODE_UPDATE>(
//                     dst, d3, tileBufWidthC0Blocks, 1, mask);
//             }
//         }
//     }
//
//
//     //        wElementsInPadHBlock(fmapW+wLeft+wRight)*c0
//     //       +---------------------------------+
//     //   hTop|             Padding             |
//     //       +--wLeft--+-------------+---------+
//     //       |         |  NoPadding  |         |
//     //       | Padding |fmapH x      | Padding |hElementsInPadWBlock(fmapH)*c0
//     //       |         |    fmapW    |         |
//     //       +---------+-------------+--wRight-+
//     //       |             Padding             |hButton
//     //       +---------------------------------+
//
//     // template <bool hTop, bool hBottom>
//     // static __simd_vf__ inline void PadH(
//     //     __ubuf__ T* buf,
//     //     const uint16_t repeatTimes,
//     //     const uint32_t repeatStride,
//     //     const uint32_t buttonOffset,
//     //     const uint16_t hTopLength,
//     //     const uint16_t hButtonLength,
//     //     const uint16_t dataBlocksInW,
//     //     const uint32_t dataBlocksSkippedInButton)
//     // {
//     //     MicroAPI::RegTensor<T> value;
//     //     MicroAPI::Duplicate(value, static_cast<T>(0));
//     //     constexpr uint8_t blockElements = GetDataBlockSizeInBytes() / sizeof(T);
//     //
//     //     uint32_t hTopPaddingElements = 0;
//     //     uint16_t hTopRepeatTimes = 0;
//     //     if constexpr (hTop) {
//     //         hTopPaddingElements = hTopLength * dataBlocksInW * blockElements;
//     //         hTopRepeatTimes = CeilDivision(hTopPaddingElements, VL<T>());
//     //     }
//     //
//     //     uint32_t hButtonPaddingElements = 0;
//     //     uint16_t hButtonRepeatTimes = 0;
//     //     __ubuf__ T* buttonBuf = buf;
//     //     if constexpr (hBottom) {
//     //         hButtonPaddingElements = (hButtonLength * dataBlocksInW - dataBlocksSkippedInButton) * blockElements;
//     //         hButtonRepeatTimes = CeilDivision(hButtonPaddingElements, VL<T>());
//     //         //TODO
//     //         buttonBuf = buf + buttonOffset+dataBlocksSkippedInButton*blockElements;
//     //     }
//     //
//     //     for (uint16_t i0 = 0; i0 < repeatTimes; i0++) {
//     //         if constexpr (hTop) {
//     //             uint32_t maskValueHTop = hTopPaddingElements;
//     //             for (uint16_t i = 0; i < hTopRepeatTimes; i++) {
//     //                 MicroAPI::MaskReg mask = MicroAPI::UpdateMask<T>(maskValueHTop);
//     //                 MicroAPI::StoreAlign(buf + i0 * repeatStride + i * VL<T>(), value, mask);
//     //             }
//     //         }
//     //
//     //         if constexpr (hBottom) {
//     //             uint32_t maskValueHButton = hButtonPaddingElements;
//     //             for (uint16_t i = 0; i < hButtonRepeatTimes; i++) {
//     //                 MicroAPI::MaskReg mask = MicroAPI::UpdateMask<T>(maskValueHButton);
//     //                 MicroAPI::StoreAlign(buttonBuf + i0 * repeatStride + i * VL<T>(), value, mask);
//     //             }
//     //         }
//     //     }
//     // }
//
//     // struct PadParams {
//     //     uint32_t tileBufC1Stride;
//     //     uint32_t bufWidth;
//     //     uint32_t bufWidthC0Blocks;
//     //     uint32_t hElementsInPadWBlock;
//     //     uint32_t wElementsInPadHBlock;
//     //     uint32_t fmapH;
//     //     uint32_t fmapW;
//     //     uint16_t hRepeatTimesInPadWBlock;
//     //     uint16_t wRepeatTimesInPadHBlock;
//     //     HWPad pad;
//     // };
//     //
//     // template <bool hTop, bool hBottom, bool wLeft, bool wRight>
//     // static __simd_vf__ inline void PadFmap(
//     //     __ubuf__ T* buf,
//     //     const uint16_t c1Length,
//     //     const PadParams params)
//     // {
//     //     const uint32_t bufWidth = params.bufWidth;
//     //     const uint32_t bufWidthC0Blocks = params.bufWidthC0Blocks;
//     //     const uint32_t hElementsInPadWBlock = params.hElementsInPadWBlock;
//     //     const uint32_t wElementsInPadHBlock = params.wElementsInPadHBlock;
//     //     const uint32_t fmapH = params.fmapH;
//     //     const uint32_t fmapW = params.fmapW;
//     //     const uint16_t hRepeatTimesInPadWBlock = params.hRepeatTimesInPadWBlock;
//     //     const uint16_t wRepeatTimesInPadHBlock = params.wRepeatTimesInPadHBlock;
//     //     const HWPad& pad = params.pad;
//     //
//     //     MicroAPI::RegTensor<T> value;
//     //     MicroAPI::Duplicate(value, static_cast<T>(0));
//     //
//     //     for (uint16_t c1 = 0; c1 < c1Length; c1++) {
//     //         if constexpr (hTop) {
//     //             for (uint16_t i = 0; i < pad.hTop; i++) {
//     //                 uint32_t maskValue = wElementsInPadHBlock;
//     //                 uint32_t offset = i * bufWidth;
//     //                 for (uint16_t w = 0; w < wRepeatTimesInPadHBlock; w++) {
//     //                     MicroAPI::MaskReg mask = MicroAPI::UpdateMask<T>(maskValue);
//     //                     MicroAPI::StoreAlign(buf + offset + w * VL<T>(), value, mask);
//     //                 }
//     //             }
//     //         }
//     //
//     //         if constexpr (hBottom) {
//     //             for (uint16_t i = 0; i < pad.hBottom; i++) {
//     //                 uint32_t maskValue = wElementsInPadHBlock;
//     //                 uint32_t offset = (i + pad.hTop + fmapH) * bufWidth;
//     //                 for (uint16_t w = 0; w < wRepeatTimesInPadHBlock; w++) {
//     //                     MicroAPI::MaskReg mask = MicroAPI::UpdateMask<T>(maskValue);
//     //                     MicroAPI::StoreAlign(buf + offset + w * VL<T>(), value, mask);
//     //                 }
//     //             }
//     //         }
//     //
//     //         if constexpr (wLeft) {
//     //             for (uint16_t i = 0; i < pad.wLeft; i++) {
//     //                 uint32_t maskValue = hElementsInPadWBlock;
//     //                 uint32_t offset = pad.hTop * bufWidth + i * C0<T>();
//     //                 for (uint16_t w = 0; w < hRepeatTimesInPadWBlock; w++) {
//     //                     MicroAPI::MaskReg mask = MicroAPI::UpdateMask<T>(maskValue);
//     //                     MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
//     //                         buf + offset + w * BLK_COUNT_IN_VL * bufWidth, value,
//     //                         bufWidthC0Blocks, mask);
//     //                 }
//     //             }
//     //         }
//     //
//     //         if constexpr (wRight) {
//     //             for (uint16_t i = 0; i < pad.wRight; i++) {
//     //                 uint32_t maskValue = hElementsInPadWBlock;
//     //                 uint32_t offset = pad.hTop * bufWidth + (i + pad.wLeft + fmapW) * C0<T>();
//     //                 for (uint16_t w = 0; w < hRepeatTimesInPadWBlock; w++) {
//     //                     MicroAPI::MaskReg mask = MicroAPI::UpdateMask<T>(maskValue);
//     //                     MicroAPI::StoreAlign<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
//     //                         buf + offset + w * BLK_COUNT_IN_VL * bufWidth, value,
//     //                         bufWidthC0Blocks, mask);
//     //                 }
//     //             }
//     //         }
//     //
//     //         buf += params.tileBufC1Stride;
//     //     }
//     // }
//
//
//     static __simd_callee__ inline void TransformVf(
//         MicroAPI::RegTensor<T>& s0, MicroAPI::RegTensor<T>& s1, MicroAPI::RegTensor<T>& s2, MicroAPI::RegTensor<T>& s3,
//         MicroAPI::RegTensor<T>& d0, MicroAPI::RegTensor<T>& d1, MicroAPI::RegTensor<T>& d2, MicroAPI::RegTensor<T>& d3,
//         MicroAPI::MaskReg& mask)
//     {
//         MicroAPI::Sub(d0, s0, s2, mask);
//         MicroAPI::Add(d1, s1, s2, mask);
//         MicroAPI::Sub(d2, s2, s1, mask);
//         MicroAPI::Sub(d3, s1, s3, mask);
//     }
//
//     __aicore__ inline LocalTensor<T> GetPingPongBuffer()
//     {
//         LocalTensor<T> mainBuf = vBuf_.GetWithOffset<T>(
//             singleShapeBufLength_, static_cast<uint32_t>(pingFlag_) * singleShapeBufLength_ * sizeof(T));
//         pingFlag_ = !pingFlag_;
//         return mainBuf;
//     }
//
//     __aicore__ inline LocalTensor<T> GetTransposeBuffer()
//     {
//         return vBuf_.GetWithOffset<T>(
//             transposeBufCnt_ * transposeBufLength_, singleShapeBufLength_ * 2 * sizeof(T));
//     }
//
//     TBuf<TPosition::VECIN> vBuf_;
//     GlobalTensor<T> fmapGm_;
//     const uint32_t cin_;
//     const uint32_t hi_;
//     const uint32_t wi_;
//     const uint16_t padH_;
//     const uint16_t padW_;
//     //TODO tiles/singleShapeTile挪到外部统一管理
//     const uint32_t tilesH_;
//     const uint32_t tilesW_;
//     const uint16_t singleShapeCin_;
//     const uint16_t singleShapeTilesH_;
//     const uint16_t singleShapeTilesW_;
//     const uint16_t transposeBufCnt_;
//     const uint32_t transposeBufLength_;
//     const uint32_t singleShapeBufLength_;
//     bool pingFlag_ = true;
// };
//
// #endif //CONV_BP_WINO_TRANSFORM_FMAP_H