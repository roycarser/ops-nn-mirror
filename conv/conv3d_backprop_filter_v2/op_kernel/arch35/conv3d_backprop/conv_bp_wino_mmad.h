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
 * \file conv_bp_wino_mmad.h
 * \brief
 */

#ifndef CONV_BP_WINO_MMAD_H
#define CONV_BP_WINO_MMAD_H

#include "conv_bp_wino_util.h"

using namespace AscendC;

template <typename T>
class WinoMMAD {
public:
    __aicore__ WinoMMAD(bool hf32Flag)
        : hf32Flag_(hf32Flag)
    {
    }

    __aicore__ inline void Init(uint32_t singleShapeCout, uint32_t singleShapeCin, uint32_t singleShapeTilesHW)
    {
        TPipe* pipe = GetTPipePtr();
        pipe->InitBuffer(l1Buf_, TOTAL_L1_SIZE);

        uint32_t tile16Size = singleShapeTilesHW * F23_TRANSFORM_TILE_ELEMENTS_16;
        l1aLength_ = singleShapeCout * tile16Size;
        l1bLength_ = singleShapeCin * tile16Size;

        uint32_t l0aBufSize;
        uint32_t l0bBufSize;

        //计算L0上单次计算最多同时能处理多少个Winograd点
        CalcWinoPointL0Group(
            singleShapeCout,
            singleShapeCin,
            singleShapeTilesHW,
            l0PointGroup_,
            l0PointPerGroup_,
            l0aBufSize,
            l0bBufSize);

        pipe->InitBuffer(l0aBuf_[0], l0aBufSize);
        pipe->InitBuffer(l0aBuf_[1], l0aBufSize);
        pipe->InitBuffer(l0bBuf_[0], l0bBufSize);
        pipe->InitBuffer(l0bBuf_[1], l0bBufSize);
        pipe->InitBuffer(l0cBuf_, TOTAL_L0C_SIZE);

        EventFlag::template Alloc<HardEvent::MTE2_MTE1, HardEvent::MTE1_MTE2>(pipe, mte2mte1Flag_[0]);
        EventFlag::template Alloc<HardEvent::MTE2_MTE1, HardEvent::MTE1_MTE2>(pipe, mte2mte1Flag_[1]);
        EventFlag::template Alloc<HardEvent::MTE1_M, HardEvent::M_MTE1>(pipe, mte1madFlag_[0]);
        EventFlag::template Alloc<HardEvent::MTE1_M, HardEvent::M_MTE1>(pipe, mte1madFlag_[1]);

        SetFlag<HardEvent::MTE1_MTE2>(mte2mte1Flag_[0].dst2src);
        SetFlag<HardEvent::MTE1_MTE2>(mte2mte1Flag_[1].dst2src);
        SetFlag<HardEvent::M_MTE1>(mte1madFlag_[0].dst2src);
        SetFlag<HardEvent::M_MTE1>(mte1madFlag_[1].dst2src);
        SetHF32Mode(hf32Flag_);
    }

    __aicore__ inline void End()
    {
        SetHF32Mode(false);
        WaitFlag<HardEvent::MTE1_MTE2>(mte2mte1Flag_[0].dst2src);
        WaitFlag<HardEvent::MTE1_MTE2>(mte2mte1Flag_[1].dst2src);
        WaitFlag<HardEvent::M_MTE1>(mte1madFlag_[0].dst2src);
        WaitFlag<HardEvent::M_MTE1>(mte1madFlag_[1].dst2src);
    }

    __aicore__ inline void LoadL1(
        const HWBox& tiles, uint32_t batchIdx, uint32_t k1Idx,
        const NK1C1K0C0<T>& nk1c1k0c0Dy, uint32_t coutC1Idx, uint32_t coutC1Length,
        const NK1C1K0C0<T>& nk1c1k0c0Fmap, uint32_t cinC1Idx, uint32_t cinC1Length,
        bool l1PingPongFlag)
    {
        EventFlag& mte2mte1 = mte2mte1Flag_[l1PingPongFlag];
        WaitFlag<HardEvent::MTE1_MTE2>(mte2mte1.dst2src);

        auto l1Buf = GetL1Buf(l1PingPongFlag);
        LocalTensor<T>& l1a = Std::get<0>(l1Buf);
        LocalTensor<T>& l1b = Std::get<1>(l1Buf);
        //TODO L1 要留一个(16-tile.elements%16)的空间
        // 让load2d取最后一个点的最后一个分形时凑满512字节
        nk1c1k0c0Dy.CopyK0In(l1a, tiles, batchIdx, k1Idx, coutC1Idx, coutC1Length);
        nk1c1k0c0Fmap.CopyK0In(l1b, tiles, batchIdx, k1Idx, cinC1Idx, cinC1Length);

        SetFlag<HardEvent::MTE2_MTE1>(mte2mte1.src2dst);
    }

    __aicore__ inline void Compute(
        const HWBox& tiles, uint32_t coutC1, uint32_t cinC1,
        bool firstK, bool l1PingPongFlag)
    {
        EventFlag& mte2mte1Flag = mte2mte1Flag_[l1PingPongFlag];
        WaitFlag<HardEvent::MTE2_MTE1>(mte2mte1Flag.src2dst);

        //l0c均分16片给每个点使用
        constexpr uint32_t l0cBufSize = TOTAL_L0C_SIZE / F23_TRANSFORM_TILE_ELEMENTS_16 / sizeof(float);
        LocalTensor<float> l0c = l0cBuf_.Get<float>();
        LocalTensor<T> l0aBuf[2] = {l0aBuf_[0].Get<T>(), l0aBuf_[1].Get<T>()};
        LocalTensor<T> l0bBuf[2] = {l0bBuf_[0].Get<T>(), l0bBuf_[1].Get<T>()};

        auto l1Buf = GetL1Buf(l1PingPongFlag);
        LocalTensor<T>& l1a = Std::get<0>(l1Buf);
        LocalTensor<T>& l1b = Std::get<1>(l1Buf);

        MmadParams mad;
        mad.m = coutC1 * C0<T>();
        mad.n = cinC1 * C0<T>();
        mad.k = tiles.elements;
        mad.cmatrixInitVal = firstK;

        LoadData2DParamsV2 load2d;
        load2d.mStartPosition = 0;
        load2d.kStartPosition = 0;
        load2d.ifTranspose = true;
        load2d.srcStride = static_cast<int32_t>(tiles.elements);

        // 数据按照 [C1,16,tile.elements,C0]排布,使用load2d一次搬运一个point,即[C1,tile.elements,C0]的数据进L0
        //
        //                          C0             C0
        //                         ----      /----
        // tile.elements(point0)    ..      /  ..
        //                         ----    /  ----
        //     ...                 ----   /   ----
        //                         ----  /    ----
        // tile.elements(point15)   ..  /      ..
        //                         ----/      ----

        uint32_t l0aMStep;
        uint32_t l0aKStep;
        uint32_t l0bMStep;
        uint32_t l0bKStep;

        if constexpr (sizeof(T) == 2) {
            l0aMStep = Ops::Base::CeilDiv(tiles.elements, static_cast<uint32_t>(BLOCK_CUBE));
            l0aKStep = coutC1;
            l0bMStep = l0aMStep;
            l0bKStep = cinC1;
        } else {
            l0aMStep = Ops::Base::CeilDiv(tiles.elements, static_cast<uint32_t>(BLOCK_CUBE));
            l0aKStep = Ops::Base::CeilAlign(coutC1, 2u);
            l0bMStep = l0aMStep;
            l0bKStep = Ops::Base::CeilAlign(cinC1, 2u);
        }

        uint32_t l0aPointElements = l0aMStep * l0aKStep * (AscendC::BYTE_PER_FRACTAL / sizeof(T));
        uint32_t l0bPointElements = l0bMStep * l0bKStep * (AscendC::BYTE_PER_FRACTAL / sizeof(T));

        //不需要baseK循环,L1上左右Tensor在PingPong后最多一共占用256kb
        //除以16后单个点最多16kb,L0上一定能全载,除非singleShapeHW传进来为1
        //然后l0上对齐放大到16这类异常情况,但tiling阶段应该防止这种情况

        for (uint8_t g = 0; g < l0PointGroup_; g++) {
            //通过奇偶性判断l0PingPong
            const int l0pingFlag = g & 1;

            EventFlag& mte1madFlag = mte1madFlag_[l0pingFlag];
            WaitFlag<HardEvent::M_MTE1>(mte1madFlag.dst2src);

            uint8_t pointGroupOffset = g * l0PointPerGroup_;

            LocalTensor<T>& l0a = l0aBuf[l0pingFlag];
            LocalTensor<T>& l0b = l0bBuf[l0pingFlag];

            for (uint8_t i = 0; i < l0PointPerGroup_; i++) {
                uint32_t pointIdx = pointGroupOffset + i;
                uint32_t offsetL1 = pointIdx * tiles.elements * C0<T>();

                load2d.mStep = l0aMStep;
                load2d.kStep = l0aKStep;
                if constexpr (sizeof(T) == 4) {
                    load2d.dstStride = static_cast<int32_t>(l0aKStep) / 2;
                } else {
                    load2d.dstStride = static_cast<int32_t>(l0aKStep);
                }

                LoadData(l0a[i * l0aPointElements], l1a[offsetL1], load2d);

                load2d.mStep = l0bMStep;
                load2d.kStep = l0bKStep;
                if constexpr (sizeof(T) == 4) {
                    load2d.dstStride = static_cast<int32_t>(l0bKStep) / 2;
                } else {
                    load2d.dstStride = static_cast<int32_t>(l0bKStep);
                }

                LoadData(l0b[i * l0bPointElements], l1b[offsetL1], load2d);
            }

            SetFlag<HardEvent::MTE1_M>(mte1madFlag.src2dst);
            WaitFlag<HardEvent::MTE1_M>(mte1madFlag.src2dst);

            //通过将一批点位放一起执行，减少mad断流导致的PI缓起开销和头开销
            for (uint8_t i = 0; i < l0PointPerGroup_; i++) {
                uint32_t pointIdx = pointGroupOffset + i;

                uint32_t offsetC = pointIdx * l0cBufSize;
                uint32_t offsetA = i * l0aPointElements;
                uint32_t offsetB = i * l0bPointElements;

                AscendC::Mmad(l0c[offsetC], l0a[offsetA], l0b[offsetB], mad);
            }

            SetFlag<HardEvent::M_MTE1>(mte1madFlag.dst2src);
        }

        SetFlag<HardEvent::MTE1_MTE2>(mte2mte1Flag.dst2src);
    }

private:
    __aicore__ inline Std::tuple<LocalTensor<T>, LocalTensor<T> > GetL1Buf(bool flagPingPong)
    {
        //PingPong按L1/2为界
        //整个L1被均分为2个bank,PingPong以L1SIZE/2为界分配到2个bank上
        //确保PingBuf上执行mte1/mte2时不会和PongBuf上的mte1/mte2产生bank冲突

        constexpr uint32_t offsetPingPong = TOTAL_L1_SIZE / 2;
        uint32_t initOffset = offsetPingPong * flagPingPong;
        LocalTensor<T> l1a = l1Buf_.GetWithOffset<T>(l1aLength_, initOffset);
        LocalTensor<T> l1b = l1Buf_.GetWithOffset<T>(l1bLength_, initOffset + l1aLength_ * sizeof(T));

        return Std::make_tuple(l1a, l1b);
    }

    static __aicore__ inline void CalcWinoPointL0Group(
        uint32_t singleShapeCout,
        uint32_t singleShapeCin,
        uint32_t singleShapeTilesHW,
        uint8_t& outGroup,
        uint8_t& outPointPerGroup,
        uint32_t& outL0aSize,
        uint32_t& outL0bSize)
    {
        constexpr uint32_t blockCube = BLOCK_CUBE;

        uint32_t l0KSize = Ops::Base::CeilAlign(singleShapeTilesHW, blockCube) * sizeof(T);
        uint32_t singlePointL0ASize = l0KSize * Ops::Base::CeilAlign(singleShapeCout, blockCube);
        uint32_t singlePointL0BSize = l0KSize * Ops::Base::CeilAlign(singleShapeCin, blockCube);

        constexpr uint32_t l0BufLimit = TOTAL_L0A_SIZE / 2;
        uint32_t maxPointsL0A = l0BufLimit / singlePointL0ASize;
        uint32_t maxPointsL0B = l0BufLimit / singlePointL0BSize;
        uint32_t maxPointsL0 = Std::min(maxPointsL0A, maxPointsL0B);

        //计算最多几个点一起批跑,从1,2,4,8这几个数里挑选,确保整除不会有尾轮处理
        //因为开了PingPong所以最多8个点一批,16个点一批PingPong就没意义了
        uint32_t pointsPerGroup = maxPointsL0 >= 8 ?
                                      8 :
                                      maxPointsL0 >= 4 ?
                                      4 :
                                      maxPointsL0 >= 2 ?
                                      2 :
                                      1;

        outGroup = F23_TRANSFORM_TILE_ELEMENTS_16 / pointsPerGroup;
        outPointPerGroup = pointsPerGroup;
        outL0aSize = outPointPerGroup * singlePointL0ASize;
        outL0bSize = outPointPerGroup * singlePointL0BSize;
    }

    struct EventFlag {
        TEventID src2dst;
        TEventID dst2src;

        template <HardEvent Src2DstEvent, HardEvent Dst2SrcEvent>
        static __aicore__ inline void Alloc(TPipe* pipe, EventFlag& flag)
        {
            flag.src2dst = pipe->AllocEventID<Src2DstEvent>();
            flag.dst2src = pipe->AllocEventID<Dst2SrcEvent>();
        }
    };

    //L1上用TBuf规避bank冲突
    TBuf<TPosition::A1> l1Buf_;
    uint32_t l1aLength_ = 0;
    uint32_t l1bLength_ = 0;
    EventFlag mte2mte1Flag_[2];

    //TQue的scalar比较重,mte1上操作比较频繁直接用TBuf减少scalar开销
    EventFlag mte1madFlag_[2];
    TBuf<TPosition::A2> l0aBuf_[2];
    TBuf<TPosition::B2> l0bBuf_[2];
    TBuf<TPosition::CO1> l0cBuf_;

    uint8_t l0PointGroup_;
    uint8_t l0PointPerGroup_;
    const bool hf32Flag_;
};

#endif //CONV_BP_WINO_MMAD_H