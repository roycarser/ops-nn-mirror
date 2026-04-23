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
    __aicore__ WinoMMAD(uint16_t baseK, bool hf32Flag)
        : baseK_(baseK),
          hf32Flag_(hf32Flag)
    {
    }

    __aicore__ inline void Init(uint32_t singleShapeCout, uint32_t singleShapeCin, uint32_t singleShapeTilesHW)
    {
        TPipe* pipe = GetTPipePtr();
        pipe->InitBuffer(l1Buf_, TOTAL_L1_SIZE);

        uint32_t tile16Size = singleShapeTilesHW * F23_TRANSFORM_TILE_ELEMENTS_16;
        l1aLength_ = singleShapeCout * tile16Size;
        l1bLength_ = singleShapeCin * tile16Size;

        uint32_t baseKSize = baseK_ * sizeof(T);
        uint32_t l0aSize = singleShapeCout * baseKSize;
        pipe->InitBuffer(l0aBuf_[0], l0aSize);
        pipe->InitBuffer(l0aBuf_[1], l0aSize);

        uint32_t l0bSize = singleShapeCin * baseKSize;
        pipe->InitBuffer(l0bBuf_[0], l0bSize);
        pipe->InitBuffer(l0bBuf_[1], l0bSize);

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

        nk1c1k0c0Dy.CopyK0TileIn(l1a, tiles, batchIdx, k1Idx, coutC1Idx, coutC1Length);
        nk1c1k0c0Fmap.CopyK0TileIn(l1b, tiles, batchIdx, k1Idx, cinC1Idx, cinC1Length);

        SetFlag<HardEvent::MTE2_MTE1>(mte2mte1.src2dst);
    }

    __aicore__ inline void Compute(
        const HWBox& tiles, uint32_t coutC1, uint32_t cinC1,
        bool firstK, bool l1PingPongFlag)
    {
        EventFlag& mte2mte1Flag = mte2mte1Flag_[l1PingPongFlag];
        WaitFlag<HardEvent::MTE2_MTE1>(mte2mte1Flag.src2dst);

        //用load3d默认的配置,每个LoadData里面会设置load3d的FMatrix和PadValue这2个寄存器
        //导致L1每次搬运多2个MOVE_SPR的指令,这里直接外部统一设置优化下
        static constexpr IsResetLoad3dConfig load3dNotSetSPR = {false, false};
        constexpr uint8_t pad[4] = {0, 0, 0, 0};
        //设置寄存器也算在对应的mte指令里面,所以必须和loadData一样在WaitFlag之后执行
        SetFmatrix(tiles.elements, F23_TRANSFORM_TILE_ELEMENTS_16, pad, FmatrixMode::FMATRIX_LEFT);

        LoadData3DParamsV2<T> load3d;
        load3d.strideW = F23_TRANSFORM_TILE_ELEMENTS_16;
        load3d.strideH = 1;
        load3d.filterW = 1;
        load3d.filterH = 1;
        load3d.kStartPt = 0;

        //l0c均分16片给每个点使用
        constexpr uint32_t l0cBufSize = TOTAL_L0C_SIZE / F23_TRANSFORM_TILE_ELEMENTS_16 / sizeof(float);
        LocalTensor<float> l0c = l0cBuf_.Get<float>();
        LocalTensor<T> l0aBuf[2] = {l0aBuf_[0].Get<T>(), l0aBuf_[1].Get<T>()};
        LocalTensor<T> l0bBuf[2] = {l0bBuf_[0].Get<T>(), l0bBuf_[1].Get<T>()};

        auto l1Buf = GetL1Buf(l1PingPongFlag);
        LocalTensor<T>& l1a = Std::get<0>(l1Buf);
        LocalTensor<T>& l1b = Std::get<1>(l1Buf);

        uint32_t cinC1C0 = cinC1 * C0<T>();
        uint32_t coutC1C0 = coutC1 * C0<T>();

        for (uint32_t offsetK = 0; offsetK < tiles.elements; offsetK += baseK_) {
#pragma unroll
            for (uint8_t i = 0; i != F23_TRANSFORM_TILE_ELEMENTS_16; i++) {
                //通过奇偶性判断l0PingPong
                const int l0pingFlag = i & 1;

                uint16_t l1Offset = i * C0<T>();
                uint16_t k = Std::min(baseK_, tiles.elements - offsetK);

                load3d.mStartPt = offsetK;
                load3d.mExtension = k;

                EventFlag& mte1madFlag = mte1madFlag_[l0pingFlag];
                WaitFlag<HardEvent::M_MTE1>(mte1madFlag.dst2src);

                LocalTensor<T>& l0a = l0aBuf[l0pingFlag];
                load3d.kExtension = cinC1;
                load3d.enTranspose = true;
                load3d.channelSize = cinC1C0;
                //禁止load3d内部额外设置寄存器
                LoadData<T, load3dNotSetSPR>(l0a, l1a[l1Offset], load3d);

                LocalTensor<T>& l0b = l0bBuf[l0pingFlag];
                load3d.kExtension = coutC1;
                load3d.enTranspose = false;
                load3d.channelSize = coutC1C0;
                LoadData<T, load3dNotSetSPR>(l0b, l1b[l1Offset], load3d);

                SetFlag<HardEvent::MTE1_M>(mte1madFlag.src2dst);
                WaitFlag<HardEvent::MTE1_M>(mte1madFlag.src2dst);

                MmadParams params;
                params.m = coutC1C0;
                params.n = cinC1C0;
                params.k = k;
                params.cmatrixInitVal = firstK;
                AscendC::Mmad(l0c[i * l0cBufSize], l0a, l0b, params);
                SetFlag<HardEvent::M_MTE1>(mte1madFlag.dst2src);
            }

            firstK = false;
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

    const uint16_t baseK_;
    const bool hf32Flag_;
};

#endif //CONV_BP_WINO_MMAD_H