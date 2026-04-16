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
        pipe->InitBuffer(l0aQue_, 2, singleShapeCout * baseKSize);
        pipe->InitBuffer(l0bQue_, 2, singleShapeCin * baseKSize);
        pipe->InitBuffer(l0cBuf, TOTAL_L0C_SIZE);

        SetHF32Mode(hf32Flag_);
    }

    __aicore__ inline void End()
    {
        SetHF32Mode(false);
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
        LocalTensor<T> l1a = l1aQue_.DeQue();
        LocalTensor<T> l1b = l1bQue_.DeQue();

        LoadData3DParamsV2<T> load3d;
        load3d.l1H = TileUnfoldElements(tiles.elements);
        load3d.l1W = F23_TRANSFORM_TILE_ELEMENTS_16;

        load3d.strideW = F23_TRANSFORM_TILE_ELEMENTS_16;
        load3d.strideH = 1;
        load3d.filterW = 1;
        load3d.filterH = 1;
        load3d.kStartPt = 0;

        for (uint32_t offsetK = 0; offsetK < tiles.elements; offsetK += baseK_) {
#pragma unroll
            for (uint8_t i = 0; i != F23_TRANSFORM_TILE_ELEMENTS_16; i++) {
                uint16_t l1Offset = i * C0<T>();
                uint16_t k = Std::min(baseK_, tiles.elements - offsetK);

                load3d.mStartPt = offsetK;
                load3d.mExtension = k;

                LocalTensor<T> l0a = l0aQue_.AllocTensor<T>();
                load3d.kExtension = cinC1 / C0<T>();
                load3d.enTranspose = true;
                load3d.channelSize = cinC1;
                LoadData(l0a, l1a[l1Offset], load3d);
                l0aQue_.EnQue(l0a);

                LocalTensor<T> l0b = l0bQue_.AllocTensor<T>();
                load3d.kExtension = coutC1 / C0<T>();
                load3d.enTranspose = false;
                load3d.channelSize = coutC1;
                LoadData(l0b, l1b[l1Offset], load3d);
                l0bQue_.EnQue(l0b);

                //l0c均分16片给每个点使用
                constexpr uint32_t l0cBufSize = TOTAL_L0C_SIZE / F23_TRANSFORM_TILE_ELEMENTS_16;
                LocalTensor<T> l0c;
                l0c = l0cBuf.GetWithOffset<T>(l0cBufSize, l0cBufSize * sizeof(T) * i);
                l0a = l0aQue_.DeQue(l0a);
                l0b = l0bQue_.DeQue(l0b);

                MmadParams params;
                params.m = coutC1 * C0<T>();
                params.n = cinC1 * C0<T>();
                params.k = k;
                params.cmatrixInitVal = firstK;
                AscendC::Mmad(l0c, l0a, l0b, params);

                l0aQue_.FreeTensor(l0a);
                l0bQue_.FreeTensor(l0b);
            }

            firstK = false;
        }

        l1aQue_.FreeTensor(l1a);
        l1bQue_.FreeTensor(l1b);
    }

private:
    TQue<TPosition::A1, 1> l1aQue_;
    TQue<TPosition::B1, 1> l1bQue_;
    TQue<TPosition::A2, 1> l0aQue_;
    TQue<TPosition::B2, 1> l0bQue_;
    TBuf<TPosition::CO1> l0cBuf;
    const uint16_t baseK_;
    const bool hf32Flag_;
};

#endif //CONV_BP_WINO_MMAD_H