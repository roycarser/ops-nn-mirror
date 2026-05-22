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
 * \file conv_bp_wino_out_transform.h
 * \brief
 */

#ifndef CONV_BP_WINO_OUT_TRANSFORM_H
#define CONV_BP_WINO_OUT_TRANSFORM_H

#include "kernel_operator.h"
#include "conv_bp_wino_util.h"


using namespace AscendC;

class WinoOutputTransformer {
public:
    //需要申请18个CoutCin空间,逆变换前16个用来放原始数据,逆变换后,9个用来放逆变换后的数据,剩下9个放转置后的数据
    static constexpr uint32_t COUT_CIN_BUF_CNT = 18;

    void Init()
    {
        v2mte3_ = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
    }

    __aicore__ inline void PartitionProcess(
        const GlobalTensor<float>& yGm,
        const LocalTensor<float>& buf,
        uint32_t coutIdx,
        uint32_t cinIdx,
        uint32_t coutLength,
        uint32_t cinLength,
        uint32_t cinSrc)
    {
        uint32_t singlePointAlignElements = AivPartitioner::Get2DAlignBufLength<float>(
            coutLength, cinLength);

        uint32_t partitionCoutIdx;
        uint32_t partitionCoutLength;
        AivPartitioner::GetPartition(coutLength, partitionCoutIdx, partitionCoutLength);
        uint32_t partitionCoutCinLength = partitionCoutLength * cinLength;
        //将4*4的dw变换为3*3的dw
        TransformVf(
            reinterpret_cast<__ubuf__ float*>(buf.GetPhyAddr()),
            singlePointAlignElements,
            Ops::Base::CeilDiv(singlePointAlignElements, VL<float>()),
            partitionCoutCinLength,
            Ops::Base::CeilDiv(partitionCoutCinLength, VL<float>()));

        SetFlag<HardEvent::V_MTE3>(v2mte3_);
        WaitFlag<HardEvent::V_MTE3>(v2mte3_);

        uint64_t gmOffset = static_cast<uint64_t>(coutIdx + partitionCoutIdx) * cinIdx * KERNEL_3x3;

        DataCopyExtParams params;
        params.blockCount = partitionCoutLength;
        params.blockLen = cinLength * KERNEL_3x3 * sizeof(float);
        params.srcStride = 0;
        params.dstStride = (static_cast<int64_t>(cinSrc) - cinLength) * KERNEL_3x3 * sizeof(float);

        DataCopyPad<float, PaddingMode::Compact>(
            yGm[gmOffset],
            buf[KERNEL_3x3 * singlePointAlignElements],
            params);
    }

private:
    static constexpr uint32_t KERNEL_3 = 3;
    static constexpr uint32_t KERNEL_3x3 = 9;

    __simd_vf__ static inline void TransformVf(
        __ubuf__ float* buf,
        uint32_t singlePointAlignElements,
        uint16_t loopCnt0,
        uint32_t partitionCoutCinLength,
        uint16_t loopCnt1)
    {
        Transform4x4To3x3Vf(buf, singlePointAlignElements, loopCnt0);
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
        Transpose2NCHW(buf, singlePointAlignElements, partitionCoutCinLength, loopCnt1);
    }

    __simd_callee__ static inline void Transform4x4To3x3Vf(
        __ubuf__ float* buf,
        uint32_t singlePointAlignElements,
        uint16_t loopCnt)
    {
        using namespace MicroAPI;
        RegTensor<float> value0P5;
        Duplicate(value0P5, 0.5f);

        uint32_t srcRowStride = singlePointAlignElements * F23_TRANSFORM_TILE_SIZE_4;

        for (uint16_t col = 0; col < 4; col++) {
            __ubuf__ float* src0 = buf + col * singlePointAlignElements;
            __ubuf__ float* src1 = src0 + srcRowStride;
            __ubuf__ float* src2 = src1 + srcRowStride;
            __ubuf__ float* src3 = src2 + srcRowStride;

            uint32_t maskValue = singlePointAlignElements;
            for (uint16_t i = 0; i < loopCnt; i++) {
                MaskReg mask = UpdateMask<float>(maskValue);

                RegTensor<float> s0;
                RegTensor<float> s1;
                RegTensor<float> s2;
                RegTensor<float> s3;
                RegTensor<float> d0;
                RegTensor<float> d1;
                RegTensor<float> d2;

                LoadAlign(s0, src0);
                LoadAlign(s1, src1);
                LoadAlign(s2, src2);
                LoadAlign<float, PostLiteral::POST_MODE_UPDATE>(s3, src3, VL<float>());

                TransformVf(value0P5, s0, s1, s2, s3, d0, d1, d2, mask);

                StoreAlign<float, PostLiteral::POST_MODE_UPDATE>(src0, d0, VL<float>(), mask);
                StoreAlign<float, PostLiteral::POST_MODE_UPDATE>(src1, d1, VL<float>(), mask);
                StoreAlign<float, PostLiteral::POST_MODE_UPDATE>(src2, d2, VL<float>(), mask);
            }
        }

        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

        for (uint16_t row = 0; row < 4; row++) {
            __ubuf__ float* src0 = buf + row * srcRowStride;
            __ubuf__ float* src1 = src0 + singlePointAlignElements;
            __ubuf__ float* src2 = src1 + singlePointAlignElements;
            __ubuf__ float* src3 = src2 + singlePointAlignElements;

            //让变换结果变成连在一起
            __ubuf__ float* dst0 = buf + singlePointAlignElements * row * 3;
            __ubuf__ float* dst1 = dst0 + singlePointAlignElements;
            __ubuf__ float* dst2 = dst1 + singlePointAlignElements;

            uint32_t maskValue = singlePointAlignElements;
            for (uint16_t i = 0; i < loopCnt; i++) {
                MaskReg mask = UpdateMask<float>(maskValue);

                RegTensor<float> s0;
                RegTensor<float> s1;
                RegTensor<float> s2;
                RegTensor<float> s3;
                RegTensor<float> d0;
                RegTensor<float> d1;
                RegTensor<float> d2;

                LoadAlign<float, PostLiteral::POST_MODE_UPDATE>(s0, src0, VL<float>());
                LoadAlign<float, PostLiteral::POST_MODE_UPDATE>(s1, src1, VL<float>());
                LoadAlign<float, PostLiteral::POST_MODE_UPDATE>(s2, src2, VL<float>());
                LoadAlign<float, PostLiteral::POST_MODE_UPDATE>(s3, src3, VL<float>());

                TransformVf(value0P5, s0, s1, s2, s3, d0, d1, d2, mask);

                StoreAlign<float, PostLiteral::POST_MODE_UPDATE>(dst0, d0, VL<float>(), mask);
                StoreAlign<float, PostLiteral::POST_MODE_UPDATE>(dst1, d1, VL<float>(), mask);
                StoreAlign<float, PostLiteral::POST_MODE_UPDATE>(dst2, d2, VL<float>(), mask);
            }
        }
    }

    __simd_callee__ static inline void TransformVf(
        MicroAPI::RegTensor<float>& value0P5,
        MicroAPI::RegTensor<float>& s0,
        MicroAPI::RegTensor<float>& s1,
        MicroAPI::RegTensor<float>& s2,
        MicroAPI::RegTensor<float>& s3,
        MicroAPI::RegTensor<float>& d0,
        MicroAPI::RegTensor<float>& d1,
        MicroAPI::RegTensor<float>& d2,
        MicroAPI::MaskReg& mask)
    {
        MicroAPI::RegTensor<float> tmpAdd;
        MicroAPI::RegTensor<float> tmpSub;
        MicroAPI::RegTensor<float> tmpAddHalf;

        MicroAPI::Add(tmpAdd, s1, s2, mask);
        MicroAPI::Sub(tmpSub, s1, s2, mask);
        MicroAPI::Mul(tmpAddHalf, tmpAdd, value0P5, mask);
        MicroAPI::Mul(d0, s0, tmpAddHalf, mask);
        MicroAPI::Mul(d1, tmpSub, value0P5, mask);
        MicroAPI::Mul(d2, s3, tmpAddHalf, mask);
    }

    __simd_callee__ static inline void Transpose2NCHW(
        __ubuf__ float* buf,
        uint32_t singlePointAlignElements,
        uint32_t partitionCoutCinLength,
        uint16_t loopCnt)
    {
        using namespace MicroAPI;

        RegTensor<uint32_t> seq;
        RegTensor<uint32_t> tmp9;
        RegTensor<uint32_t> index;

        Arange(reinterpret_cast<RegTensor<int32_t>&>(seq), 0);
        Duplicate(tmp9, 9);
        MaskReg maskAll = CreateMask<uint32_t, MaskPattern::ALL>();
        Mul(index, seq, tmp9, maskAll);

        RegTensor<float> regCoutCin;
        __ubuf__ float* dst = buf + KERNEL_3x3 * singlePointAlignElements;

        for (uint16_t n = 0; n < KERNEL_3x3; n++) {
            __ubuf__ float* src = buf + n * singlePointAlignElements;
            __ubuf__ float* dst0 = dst;

            uint32_t maskValue = partitionCoutCinLength;
            for (uint16_t i = 0; i < loopCnt; i++) {
                MaskReg mask = UpdateMask<float>(maskValue);
                LoadAlign<float, PostLiteral::POST_MODE_UPDATE>(regCoutCin, src, VL<float>());
                Scatter(dst0, regCoutCin, index, mask);
                dst0 += VL<float>() * KERNEL_3x3;
            }

            dst++;
        }
    }

    TEventID v2mte3_ = 0;
};

#endif //CONV_BP_WINO_OUT_TRANSFORM_H