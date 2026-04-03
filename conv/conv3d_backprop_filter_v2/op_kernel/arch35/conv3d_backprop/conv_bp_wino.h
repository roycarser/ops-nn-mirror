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
 * \file conv_bp_wino.h
 * \brief
 */

#ifndef CONV_BP_WINO_H
#define CONV_BP_WINO_H

#include "conv_bp_wino_transform.h"
#include "../../../../conv3d_backprop_input_v2/op_kernel/arch35/conv3d_backprop_input_v2/conv3d_backprop_input_v2_vec_transpose.h"
#include "../conv3d_backprop_filter_v2/conv3d_backprop_filter_v2_init_output.h"

using namespace AscendC;

template <typename T,
    TPosition pos,
    pipe_t srcPipe,
    pipe_t dstPipe,
    uint8_t SRC_READY_FLAG,
    uint8_t DST_READ_FLAG,
    bool enablePingPong = true>
class CVSyncQueue {
    static constexpr uint8_t MIX_BLOCK_SYNC = 2;

public:
    __aicore__ inline CVSyncQueue()
    {
        freeCount_ = enablePingPong ? 2 : 1;
    }

    __aicore__ inline void InitBuffer(TPipe* pipe, uint32_t bufSize)
    {
        bufSize_ = bufSize;
        pipe->InitBuffer(buf_, sizeof(T) * (enablePingPong ? bufSize_ * 2 : bufSize_));
    }

    //src申请buf写入
    __aicore__ inline LocalTensor<T> AllocTensor()
    {
        if (freeCount_ > 0) {
            //初始阶段可以直接写入
            freeCount_--;
        } else {
            CrossCoreWaitFlag<MIX_BLOCK_SYNC, srcPipe>(DST_READ_FLAG);
        }
        return GetBuf();
    }

    //src完成写入
    __aicore__ inline void EnQue()
    {
        CrossCoreSetFlag<MIX_BLOCK_SYNC, srcPipe>(SRC_READY_FLAG);
    }

    //dst等待src写入
    __aicore__ inline LocalTensor<T> DeQue()
    {
        CrossCoreWaitFlag<MIX_BLOCK_SYNC, dstPipe>(SRC_READY_FLAG);
        return GetBuf();
    }

    //dst读取完毕,释放buf
    __aicore__ inline void FreeTensor()
    {
        CrossCoreSetFlag<MIX_BLOCK_SYNC, dstPipe>(DST_READ_FLAG);
    }

private:
    __aicore__ inline LocalTensor<T> GetBuf()
    {
        if constexpr (enablePingPong) {
            pingFlag_ = !pingFlag_;
            return buf_.template GetWithOffset<T>(bufSize_, pingFlag_ * sizeof(T) * bufSize_);
        } else {
            return buf_.template Get<T>();
        }
    }

    TBuf<pos> buf_;
    uint32_t bufSize_ = 0;
    uint8_t freeCount_;
    bool pingFlag_ = true;
};

template <typename T>
class ConvBackpropFilterWinograd {
public:
    __aicore__ inline ConvBackpropFilterWinograd(
        const WinoFmapTransformer<T>& fmap,
        const WinoDyTransformer<T>& dy,
        uint32_t tilesH,
        uint32_t tilesW,
        uint32_t cin,
        uint32_t cout,
        uint16_t singleShapeCin,
        uint16_t singleShapeCout,
        uint16_t singleShapeTilesH,
        uint16_t singleShapeTilesW)
        : fmap_(fmap),
          dy_(dy),
          tilesH_(tilesH),
          tilesW_(tilesW),
          cin_(cin),
          cout_(cout),
          singleShapeCin_(singleShapeCin),
          singleShapeCout_(singleShapeCout),
          singleShapeTilesH_(singleShapeTilesH),
          singleShapeTilesW_(singleShapeTilesW),
          singleShapeFmapBufSize_(
              SingleShapeTileBufSize(
                  singleShapeTilesH,
                  singleShapeTilesW,
                  singleShapeCin)),
          singleShapeDyBufSize_(
              SingleShapeTileBufSize(
                  singleShapeTilesH,
                  singleShapeTilesW,
                  singleShapeCout))
    {
    }

    inline void __aicore__ Init(TPipe* pipe)
    {
        if ASCEND_IS_AIV {
            uint32_t tiles = singleShapeTilesH_ * singleShapeTilesW_;
            uint32_t fmapTSize =
                fmap_.CalculateTransposeBufC0Length(tiles);
            uint32_t dyTSize =
                dy_.CalculateTransposeBufC0Length(tiles);

            pipe->InitBuffer(transposeBuf_, dyTSize + fmapTSize);
            dyTransposeBuf_ = transposeBuf_.Get<T>(dyTSize);
            fmapTransposeBuf_ = transposeBuf_.GetWithOffset<T>(fmapTSize, dyTSize * sizeof(T));

            pipe->InitBuffer(a1TransformBuf_, singleShapeDyBufSize_ * 2);
            pipe->InitBuffer(b1TransformBuf_, singleShapeFmapBufSize_ * 2);
        }

        a1Que_.InitBuffer(pipe);
        b1Que_.InitBuffer(pipe);
    }

    inline void __aicore__ IterateAll(
        uint32_t fmapBatchOffset,
        uint32_t dyBatchOffset,
        uint32_t cinIdx,
        uint32_t coutIdx)
    {
        uint32_t cinLength = Std::min(static_cast<uint32_t>(singleShapeCin_), cin_ - cinIdx);
        uint32_t coutLength = Std::min(static_cast<uint32_t>(singleShapeCout_), cout_ - cinIdx);

        TileKIter iter;
        while (!iter.end) {
            HWBox tile = {};
            tile.hIdx = tilesH_;
            tile.wIdx = tilesW_;
            tile.hLength = Std::min(static_cast<uint32_t>(singleShapeTilesH_), tilesH_ - iter.tileHIdx);
            tile.wLength = Std::min(static_cast<uint32_t>(singleShapeTilesW_), tilesW_ - iter.tileWIdx);
            tile.elements = tile.hLength * tile.wLength;

            if ASCEND_IS_AIV {
                auto buf = getPingPongBuffer();
                LocalTensor<T> dyVBuf = Std::get<0>(buf);
                LocalTensor<T> fmapVBuf = Std::get<1>(buf);

                TileBox dyBox = dy_.CalculateSrcBox(tile, coutIdx, coutLength);
                TileBox fmapBox = fmap_.CalculateSrcBox(tile, cinIdx, cinLength);

                dy_.CopyIn(fmapVBuf, dyBox, dyBatchOffset);
                fmap_.CopyIn(dyVBuf, fmapBox, fmapBatchOffset);

                PipeBarrier<PIPE_ALL>();

                dy_.Compute(fmapVBuf, dyTransposeBuf_, dyBox);
                fmap_.Compute(dyVBuf, fmapTransposeBuf_, fmapBox);

                PipeBarrier<PIPE_ALL>();

                LocalTensor<T> dyA1Buf = a1Que_.AllocTensor();
                dy_.CopyOut(dyVBuf, dyA1Buf, dyBox);
                a1Que_.EneQue();

                LocalTensor<T> fmapB1Buf = b1Que_.AllocTensor();
                fmap_.CopyOut(fmapVBuf, fmapB1Buf, fmapBox);
                b1Que_.EneQue();
            }

            TileKIter::next(
                iter,
                tilesH_, tilesW_,
                singleShapeTilesH_, singleShapeTilesW_);
        }
    }

private:
    Std::tuple<LocalTensor<T>, LocalTensor<T> > getPingPongBuffer()
    {
        uint32_t dyOffset = pingFlag_ * singleShapeDyBufSize_ * sizeof(T);
        uint32_t fmapOffset = pingFlag_ * singleShapeFmapBufSize_ * sizeof(T);

        LocalTensor<T> dyVBuf = a1TransformBuf_.GetWithOffset<T>(singleShapeDyBufSize_, dyOffset);
        LocalTensor<T> fmapVBuf = b1TransformBuf_.GetWithOffset<T>(singleShapeFmapBufSize_, fmapOffset);

        LocalTensor<T> dyA1Buf = a1Buf_.GetWithOffset<T>(singleShapeDyBufSize_, dyOffset);
        LocalTensor<T> fmapB1Buf = b1Buf_.GetWithOffset<T>(singleShapeFmapBufSize_, fmapOffset);

        pingFlag_ = !pingFlag_;
        return Std::make_tuple(dyVBuf, fmapVBuf, dyA1Buf, fmapB1Buf);
    }

    static inline uint32_t __aicore__ SingleShapeTileBufSize(uint32_t c, uint32_t th, uint32_t tw)
    {
        return TileUnfoldElements(th * tw) * Ops::Base::CeilAlign(c, C0<T>());
    }


    CVSyncQueue<T, TPosition::A1, PIPE_MTE3, PIPE_MTE1, 0, 1> a1Que_;
    CVSyncQueue<T, TPosition::B1, PIPE_MTE3, PIPE_MTE1, 2, 3> b1Que_;

    TBuf<TPosition::VECIN> a1TransformBuf_;
    TBuf<TPosition::VECIN> b1TransformBuf_;

    TBuf<TPosition::VECIN> transposeBuf_;
    LocalTensor<T> fmapTransposeBuf_;
    LocalTensor<T> dyTransposeBuf_;

    const WinoFmapTransformer<T>& fmap_;
    const WinoDyTransformer<T>& dy_;
    const uint32_t tilesH_;
    const uint32_t tilesW_;
    const uint32_t cin_;
    const uint32_t cout_;
    const uint16_t singleShapeCin_;
    const uint16_t singleShapeCout_;
    const uint16_t singleShapeTilesH_;
    const uint16_t singleShapeTilesW_;
    const uint32_t singleShapeFmapBufSize_;
    const uint32_t singleShapeDyBufSize_;
    bool pingFlag_ = true;
};


#endif //CONV_BP_WINO_H