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

        uint32_t tile16Size = singleShapeTilesHW * F23_TRANSFORM_TILE_ELEMENTS_16 * sizeof(T);
        uint32_t baseKSize = baseK_ * sizeof(T);

        pipe->InitBuffer(l1aQue_, 2, singleShapeCout * tile16Size);
        pipe->InitBuffer(l1bQue_, 2, singleShapeCin * tile16Size);

        uint32_t l0aSize = singleShapeCout * baseKSize;
        pipe->InitBuffer(l0aBuf_[0], l0aSize);
        pipe->InitBuffer(l0aBuf_[1], l0aSize);

        uint32_t l0bSize = singleShapeCin * baseKSize;
        pipe->InitBuffer(l0bBuf_[0], l0bSize);
        pipe->InitBuffer(l0bBuf_[1], l0bSize);

        pipe->InitBuffer(l0cBuf_, TOTAL_L0C_SIZE);

        Mte1MadFlag::Alloc(pipe, mte1madFlag_[0]);
        Mte1MadFlag::Alloc(pipe, mte1madFlag_[1]);

        SetFlag<HardEvent::M_MTE1>(mte1madFlag_[0].mad2mte1);
        SetFlag<HardEvent::M_MTE1>(mte1madFlag_[1].mad2mte1);

        SetHF32Mode(hf32Flag_);
    }

    __aicore__ inline void End()
    {
        SetHF32Mode(false);
        WaitFlag<HardEvent::M_MTE1>(mte1madFlag_[0].mad2mte1);
        WaitFlag<HardEvent::M_MTE1>(mte1madFlag_[1].mad2mte1);
    }

    __aicore__ inline void LoadL1(
        const HWBox& tiles, uint32_t batchIdx, uint32_t k1Idx,
        const NK1C1K0C0<T>& nk1c1k0c0Dy, uint32_t coutC1Idx, uint32_t coutC1Length,
        const NK1C1K0C0<T>& nk1c1k0c0Fmap, uint32_t cinC1Idx, uint32_t cinC1Length)
    {
        LocalTensor<T> l1a = l1aQue_.AllocTensor<T>();
        nk1c1k0c0Dy.CopyK0TileIn(l1a, tiles, batchIdx, k1Idx, coutC1Idx, coutC1Length);
        l1aQue_.EnQue(l1a);

        LocalTensor<T> l1b = l1bQue_.AllocTensor<T>();
        nk1c1k0c0Fmap.CopyK0TileIn(l1b, tiles, batchIdx, k1Idx, cinC1Idx, cinC1Length);
        l1bQue_.EnQue(l1b);
    }

    __aicore__ inline void Compute(const HWBox& tiles, uint32_t coutC1, uint32_t cinC1, bool firstK)
    {
        LocalTensor<T> l1a = l1aQue_.DeQue<T>();
        LocalTensor<T> l1b = l1bQue_.DeQue<T>();

        //用load3d默认的配置,每个LoadData里面会设置load3d的FMatrix和PadValue这2个寄存器
        //这会导致mte1的issue queue快速占满,本来一次MAD只需要2个load3d搬运指令,现在需要6条指令
        //winograd的16个点位算完需要使用96条mte1指令使得scalar被IssueQueue卡主不能加载后续的mte2指令
        //所以这里调整配置让LoadData内部不去更新寄存器,由外部统一设置,减少mte1指令占用
        static constexpr IsResetLoad3dConfig load3dNotSetSPR = {false, false};
        constexpr uint8_t pad[4] = {0, 0, 0, 0};

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

        uint32_t cinC1C0 = cinC1 * C0<T>();
        uint32_t coutC1C0 = coutC1 * C0<T>();

        for (uint32_t offsetK = 0; offsetK < tiles.elements; offsetK += baseK_) {
#pragma unroll
            for (uint8_t i = 0; i != F23_TRANSFORM_TILE_ELEMENTS_16; i++) {
                //通过奇偶性判断PingPong
                const int pingFlag = i & 1;

                uint16_t l1Offset = i * C0<T>();
                uint16_t k = Std::min(baseK_, tiles.elements - offsetK);

                load3d.mStartPt = offsetK;
                load3d.mExtension = k;

                Mte1MadFlag& flag = mte1madFlag_[pingFlag];
                WaitFlag<HardEvent::M_MTE1>(flag.mad2mte1);

                LocalTensor<T>& l0a = l0aBuf[pingFlag];
                load3d.kExtension = cinC1;
                load3d.enTranspose = true;
                load3d.channelSize = cinC1C0;
                //禁止load3d内部额外设置寄存器
                LoadData<T, load3dNotSetSPR>(l0a, l1a[l1Offset], load3d);

                LocalTensor<T>& l0b = l0bBuf[pingFlag];
                load3d.kExtension = coutC1;
                load3d.enTranspose = false;
                load3d.channelSize = coutC1C0;
                LoadData<T, load3dNotSetSPR>(l0b, l1b[l1Offset], load3d);

                SetFlag<HardEvent::MTE1_M>(flag.mte12mad);
                WaitFlag<HardEvent::MTE1_M>(flag.mte12mad);

                MmadParams params;
                params.m = coutC1C0;
                params.n = cinC1C0;
                params.k = k;
                params.cmatrixInitVal = firstK;
                AscendC::Mmad(l0c[i * l0cBufSize], l0a, l0b, params);
                SetFlag<HardEvent::M_MTE1>(flag.mad2mte1);
            }

            firstK = false;
        }

        l1aQue_.FreeTensor(l1a);
        l1bQue_.FreeTensor(l1b);
    }

private:
    struct Mte1MadFlag {
        TEventID mad2mte1;
        TEventID mte12mad;

        __aicore__ inline static void Alloc(TPipe* pipe, Mte1MadFlag& flag)
        {
            flag.mad2mte1 = pipe->AllocEventID<HardEvent::M_MTE1>();
            flag.mte12mad = pipe->AllocEventID<HardEvent::MTE1_M>();
        }
    };

    TQue<TPosition::A1, 1> l1aQue_;
    TQue<TPosition::B1, 1> l1bQue_;

    //TQue的scalar比较重,mte1上操作比较频繁直接用TBuf减少scalar开销
    Mte1MadFlag mte1madFlag_[2];
    TBuf<TPosition::A2> l0aBuf_[2];
    TBuf<TPosition::B2> l0bBuf_[2];
    TBuf<TPosition::CO1> l0cBuf_;

    const uint16_t baseK_;
    const bool hf32Flag_;
};

#endif //CONV_BP_WINO_MMAD_H