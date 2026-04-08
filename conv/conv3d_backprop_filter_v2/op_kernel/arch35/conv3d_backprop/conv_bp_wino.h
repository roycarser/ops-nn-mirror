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

using namespace AscendC;

template <typename T,
    TPosition dstPos,
    pipe_t dstPipe,
    uint8_t SRC_PUSH_QUE_FLAG,
    uint8_t DST_POP_QUE_FLAG,
    bool enablePingPong = true>
class MTE3CVSyncQueue {
    static constexpr uint8_t MIX_BLOCK_SYNC = 2;

public:
    __aicore__ inline MTE3CVSyncQueue()
    {
        freeCount_ = enablePingPong ? 2 : 1;
    }

    __aicore__ inline void Init(TPipe* pipe, uint32_t bufSize)
    {
        bufSize_ = bufSize;
        pipe->InitBuffer(buf_, sizeof(T) * (enablePingPong ? bufSize_ * 2 : bufSize_));
    }

    //申请dst buf写入
    __aicore__ inline LocalTensor<T> AllocTensor()
    {
        if (freeCount_ > 0) {
            //初始阶段可以直接写入
            freeCount_--;
        } else {
            CrossCoreWaitFlag<MIX_BLOCK_SYNC, PIPE_MTE3>(DST_POP_QUE_FLAG);
        }
        return GetBuf();
    }

    //src完成写入
    __aicore__ inline void EnQue()
    {
        CrossCoreSetFlag<MIX_BLOCK_SYNC, PIPE_MTE3>(SRC_PUSH_QUE_FLAG);
    }

    //dst等待src写入
    __aicore__ inline LocalTensor<T> DeQue()
    {
        CrossCoreWaitFlag<MIX_BLOCK_SYNC, dstPipe>(SRC_PUSH_QUE_FLAG);
        return GetBuf();
    }

    //dst读取完毕,释放buf
    __aicore__ inline void FreeTensor()
    {
        CrossCoreSetFlag<MIX_BLOCK_SYNC, dstPipe>(DST_POP_QUE_FLAG);
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

    TBuf<dstPos> buf_;
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
        uint32_t cin,
        uint32_t cout,
        uint32_t tilesH,
        uint32_t tilesW,
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
                  singleShapeCin,
                  singleShapeTilesH,
                  singleShapeTilesW)),
          singleShapeDyBufSize_(
              SingleShapeTileBufSize(
                  singleShapeCout,
                  singleShapeTilesH,
                  singleShapeTilesW))
    {
    }

    inline void __aicore__ Init()
    {
        TPipe* pipe = GetTPipePtr();

        if ASCEND_IS_AIV {
            uint32_t fmapTSize =
                fmap_.CalculateTransposeBufC0Length(singleShapeTilesH_, singleShapeTilesW_);
            uint32_t dyTSize =
                dy_.CalculateTransposeBufC0Length(singleShapeTilesH_, singleShapeTilesW_);

            pipe->InitBuffer(transposeBuf_, (dyTSize + fmapTSize) * sizeof(T));
            dyTransposeBuf_ = transposeBuf_.Get<T>(dyTSize);
            fmapTransposeBuf_ = transposeBuf_.GetWithOffset<T>(fmapTSize, dyTSize * sizeof(T));

            pipe->InitBuffer(dyVBuf_, singleShapeDyBufSize_ * 2 * sizeof(T));
            pipe->InitBuffer(fmapVBuf_, singleShapeFmapBufSize_ * 2 * sizeof(T));

            TransformVFlag::AllocEventId(pipe, fmapEventFlags_[0]);
            TransformVFlag::AllocEventId(pipe, fmapEventFlags_[1]);
            TransformVFlag::AllocEventId(pipe, dyEventFlags_[0]);
            TransformVFlag::AllocEventId(pipe, dyEventFlags_[1]);

            //头几次的mte2操作不需要等待mte3结束,预先置1
            SetFlag<HardEvent::MTE3_MTE2>(fmapEventFlags_[0].mte32mte2);
            SetFlag<HardEvent::MTE3_MTE2>(fmapEventFlags_[1].mte32mte2);
            SetFlag<HardEvent::MTE3_MTE2>(dyEventFlags_[0].mte32mte2);
            SetFlag<HardEvent::MTE3_MTE2>(dyEventFlags_[1].mte32mte2);
        }

        a1Mte3Que_.Init(pipe, singleShapeDyBufSize_);
        b1Mte3Que_.Init(pipe, singleShapeFmapBufSize_);
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
                auto flags = getPingPongFlags();
                TransformVFlag& fmapFlags = Std::get<0>(flags);
                TransformVFlag& dyFlags = Std::get<1>(flags);

                auto buf = getPingPongBuffer();
                LocalTensor<T>& fmapVBuf = Std::get<0>(buf);
                LocalTensor<T>& dyVBuf = Std::get<1>(buf);

                TileBox dyBox = dy_.CalculateSrcBox(tile, coutIdx, coutLength);
                TileBox fmapBox = fmap_.CalculateSrcBox(tile, cinIdx, cinLength);

                WaitFlag<HardEvent::MTE3_MTE2>(fmapFlags.mte32mte2);
                fmap_.CopyIn(fmapVBuf, fmapBox, fmapBatchOffset);
                SetFlag<HardEvent::MTE2_V>(fmapFlags.mte2v);

                WaitFlag<HardEvent::MTE3_MTE2>(dyFlags.mte32mte2);
                dy_.CopyIn(dyVBuf, dyBox, dyBatchOffset);
                SetFlag<HardEvent::MTE2_V>(dyFlags.mte2v);

                WaitFlag<HardEvent::MTE2_V>(fmapFlags.mte2v);
                fmap_.Compute(fmapVBuf, fmapTransposeBuf_, fmapBox);
                SetFlag<HardEvent::V_MTE3>(fmapFlags.v2mte3);

                WaitFlag<HardEvent::MTE2_V>(dyFlags.mte2v);
                dy_.Compute(dyVBuf, dyTransposeBuf_, dyBox);
                SetFlag<HardEvent::V_MTE3>(dyFlags.v2mte3);

                LocalTensor<T> fmapB1Buf = b1Mte3Que_.AllocTensor();
                WaitFlag<HardEvent::V_MTE3>(fmapFlags.v2mte3);
                fmap_.CopyOut(fmapVBuf, fmapB1Buf, fmapBox);
                SetFlag<HardEvent::MTE3_MTE2>(fmapFlags.mte32mte2);
                b1Mte3Que_.EnQue();

                LocalTensor<T> dyA1Buf = a1Mte3Que_.AllocTensor();
                WaitFlag<HardEvent::V_MTE3>(dyFlags.v2mte3);
                dy_.CopyOut(dyVBuf, dyA1Buf, dyBox);
                SetFlag<HardEvent::MTE3_MTE2>(dyFlags.mte32mte2);
                a1Mte3Que_.EnQue();
            }

            pingFlag_ = !pingFlag_;

            TileKIter::next(
                iter,
                tilesH_, tilesW_,
                singleShapeTilesH_, singleShapeTilesW_);
        }
    }

private:
    struct TransformVFlag {
        TEventID mte2v;
        TEventID v2mte3;
        TEventID mte32mte2;

        static __aicore__ inline void AllocEventId(TPipe* pipe, TransformVFlag& flags)
        {
            flags.mte2v = pipe->AllocEventID<HardEvent::MTE2_V>();
            flags.v2mte3 = pipe->AllocEventID<HardEvent::V_MTE3>();
            flags.mte32mte2 = pipe->AllocEventID<HardEvent::MTE3_MTE2>();
        }
    };

    __aicore__ inline Std::tuple<TransformVFlag, TransformVFlag> getPingPongFlags()
    {
        return Std::make_tuple(fmapEventFlags_[pingFlag_], dyEventFlags_[pingFlag_]);
    }

    __aicore__ inline Std::tuple<LocalTensor<T>, LocalTensor<T> > getPingPongBuffer()
    {
        uint32_t dyOffset = pingFlag_ * singleShapeDyBufSize_ * sizeof(T);
        uint32_t fmapOffset = pingFlag_ * singleShapeFmapBufSize_ * sizeof(T);

        LocalTensor<T> dyVBuf = dyVBuf_.GetWithOffset<T>(singleShapeDyBufSize_, dyOffset);
        LocalTensor<T> fmapVBuf = fmapVBuf_.GetWithOffset<T>(singleShapeFmapBufSize_, fmapOffset);

        return Std::make_tuple(fmapVBuf, dyVBuf);
    }

    static inline uint32_t __aicore__ SingleShapeTileBufSize(uint32_t c, uint32_t th, uint32_t tw)
    {
        uint32_t hw = TileUnfoldSize(th) * (TileUnfoldSize(tw) + TILE_BUF_BANK_CONFLICT_PADDING);
        return hw * Ops::Base::CeilAlign(c, C0<T>());
    }

    MTE3CVSyncQueue<T, TPosition::A1, PIPE_MTE1, 0, 1> a1Mte3Que_;
    MTE3CVSyncQueue<T, TPosition::B1, PIPE_MTE1, 2, 3> b1Mte3Que_;

    TBuf<TPosition::VECIN> fmapVBuf_;
    TBuf<TPosition::VECIN> dyVBuf_;

    TBuf<TPosition::VECIN> transposeBuf_;
    LocalTensor<T> fmapTransposeBuf_;
    LocalTensor<T> dyTransposeBuf_;

    TransformVFlag fmapEventFlags_[2];
    TransformVFlag dyEventFlags_[2];

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