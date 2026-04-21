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

#include "conv_bp_wino_mmad.h"
#include "conv_bp_wino_transform.h"

using namespace AscendC;


template <typename T>
class ConvBackpropFilterWinograd {
public:
    __aicore__ inline ConvBackpropFilterWinograd(
        const WinoFmapTransformer<T>& fmap,
        const WinoDyTransformer<T>& dy,
        const NK1C1K0C0<T>& nk1c1k0c0Fmap,
        const NK1C1K0C0<T>& nk1c1k0c0Dy,
        WinoMMAD<T>& winoMmad,
        uint32_t cin,
        uint32_t cout,
        uint32_t tilesH,
        uint32_t tilesW,
        uint16_t singleShapeCin,
        uint16_t singleShapeCout,
        uint16_t singleShapeTransformC,
        uint16_t blockNumCin,
        uint16_t blockNumCout,
        uint16_t singleShapeTilesH,
        uint16_t singleShapeTilesW)
        : fmap_(fmap),
          dy_(dy),
          nk1c1k0t16c0Fmap_(nk1c1k0c0Fmap),
          nk1c1k0t16c0Dy_(nk1c1k0c0Dy),
          winoMmad_(winoMmad),
          tilesH_(tilesH),
          tilesW_(tilesW),
          cin_(cin),
          cout_(cout),
          singleShapeCin_(singleShapeCin),
          singleShapeCout_(singleShapeCout),
          singleShapeTransformC_(singleShapeTransformC),
          singleShapeTilesH_(singleShapeTilesH),
          singleShapeTilesW_(singleShapeTilesW),
          blockNumCin_(blockNumCin),
          blockNumCout_(blockNumCout)
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
            uint32_t transformBufSize = SingleShapeTileBufSize(
                singleShapeTransformC_,
                singleShapeTilesH_,
                singleShapeTilesW_);

            pipe->InitBuffer(transposeBuf_, Std::max(fmapTSize, dyTSize) * sizeof(T));
            pipe->InitBuffer(transformBuf_[0], transformBufSize * sizeof(T));
            pipe->InitBuffer(transformBuf_[1], transformBufSize * sizeof(T));

            TransformVFlag::AllocEventId(pipe, transformEventFlags_[0]);
            TransformVFlag::AllocEventId(pipe, transformEventFlags_[1]);

            //初始的mte2操作不需要等待mte3结束,预先置1
            SetFlag<HardEvent::MTE3_MTE2>(transformEventFlags_[0].mte32mte2);
            SetFlag<HardEvent::MTE3_MTE2>(transformEventFlags_[1].mte32mte2);
        }

        if ASCEND_IS_AIC {
            winoMmad_.Init(singleShapeCout_, singleShapeCin_, singleShapeTilesH_ * singleShapeTilesW_);
        }
    }

    inline void __aicore__ End()
    {
        if ASCEND_IS_AIV {
            //不wait看文档说状态会残留?
            WaitFlag<HardEvent::MTE3_MTE2>(transformEventFlags_[0].mte32mte2);
            WaitFlag<HardEvent::MTE3_MTE2>(transformEventFlags_[1].mte32mte2);
        }

        if ASCEND_IS_AIC {
            winoMmad_.End();
        }

        aivMTE3ToAicMTE2SyncQue_.PipeBarrierAllEnd();
    }

    inline void __aicore__ IterateAll(
        uint32_t batchIdx)
    {
        uint32_t cinCnt = Ops::Base::CeilDiv(cin_, static_cast<uint32_t>(singleShapeCin_));
        uint32_t coutCnt = Ops::Base::CeilDiv(cout_, static_cast<uint32_t>(singleShapeCout_));

        //cout轴基本块数量必须为blockNumCout_的整数倍,不然cout方向尾块存在cube空转
        //cin轴不需要整数倍,超出后直接换行
        ascendc_assert((coutCnt % blockNumCout_) == 0);
        ascendc_assert((blockNumCout_ * blockNumCin_) == GetBlockNum());

        // 所有cube核按照[blockNumCout,blockNumCin]组织,向cin方向滑行处理基本块，通过这种方式均匀vector正变换的压力
        // 一轮下来，vector正变换需要blockNumCout+blockNumCin次
        // 假定核数为32,blockNumCout为4,blockNumCin为8，则需要12次正变换，若cube核完全按1维方式滑动,即blockNumCout为1
        // 则vector需要正变换1+32=33次
        //                                        loop order
        //                                      -------------->
        //                         blockNumCin
        //                   |-----------------------|
        //                -  +-----+-----+-----+-----+-----+-----+-----+
        //                |  |core0|core1|core2|core3|     |     |     |
        //  blockNumCout -|  +-----+-----+-----+-----+-----+-----+-----+
        //                |  |core4|core5|core6|core7|     |     |     |
        //                -  +-----+-----+-----+-----+-----+-----+-----+  CoutCnt
        //                   |     |     |     |     |     |     |     |
        //                   +-----+-----+-----+-----+-----+-----+-----+
        //                   |     |     |     |     |     |     |     |
        //                   +-----+-----+-----+-----+-----+-----+-----+
        //                                      CinCnt
        //实际计算时将整体视为一个[blockNumCout,CinCnt*CoutCnt/blockNumCout]的矩阵，然后[blockNumCout,blockNumCin]在其上滑窗
        //即将CoutCnt按blockNumCout为最小单位展平,输入确保CoutCnt能被blockNumCout整除
        //                         blockNumCin
        //                   |-----------------------|
        //                -  +-----+-----+-----+-----+-----+-----+-----+
        //                |  |core0|core1|core2|core3|     |     |     |
        //  blockNumCout -|  +-----+-----+-----+-----+-----+-----+-----+
        //                |  |core4|core5|core6|core7|     |     |     |
        //                -  +-----+-----+-----+-----+-----+-----+-----+
        //                          CinCnt*(CoutCnt/blockNumCout)

        uint32_t transformWatermarkCin = 0;
        uint32_t transformWatermarkCout = 0;

        BlockIterator blockIterator(coutCnt, cinCnt, blockNumCout_, blockNumCin_);
        while (blockIterator.More()) {
            if ASCEND_IS_AIC {
                uint32_t coutBlockIdx;
                uint32_t cinBlockIdx;

                if (blockIterator.GetCurrentAicHWIdx(coutBlockIdx, cinBlockIdx)) {
                    Mmad<true>(batchIdx, coutBlockIdx * singleShapeCout_, cinBlockIdx * singleShapeCin_);
                } else {
                    Mmad<false>(0, 0, 0);
                }
            }

            if ASCEND_IS_AIV {
                uint32_t buttonRightCoutBlockIdx = 0;
                uint32_t buttonRightCinBlockIdx = 0;

                blockIterator.GetHWIdx(
                    blockNumCout_ - 1, blockNumCin_ - 1,
                    buttonRightCoutBlockIdx, buttonRightCinBlockIdx);

                uint32_t topLeftCoutBlockIdx = 0;
                uint32_t topLeftCinBlockIdx = 0;

                blockIterator.GetHWIdx(
                    0, 0,
                    topLeftCoutBlockIdx, topLeftCinBlockIdx);

                uint32_t transformCoutEndIdx = Std::min(cout_, (buttonRightCoutBlockIdx + 1) * singleShapeCout_);
                uint32_t transformCinEndIdx;
                if (buttonRightCoutBlockIdx - topLeftCoutBlockIdx + 1 > blockNumCout_) {
                    // 滑窗产生换行,说明整个cin被滑完,必须转换完整个cin轴
                    transformCinEndIdx = cin_;
                } else {
                    transformCinEndIdx = Std::min(cin_, (buttonRightCinBlockIdx + 1) * singleShapeCin_);
                }

                //这一轮迭代cube需要的正变换cout/cin范围
                uint32_t transformCoutEnd = Std::max(transformCoutEndIdx, transformWatermarkCout);
                uint32_t transformCinEnd = Std::max(transformCinEndIdx, transformWatermarkCin);

                Transform(
                    batchIdx,
                    transformWatermarkCout,
                    transformWatermarkCin,
                    transformCoutEnd - transformWatermarkCout,
                    transformCinEnd - transformWatermarkCin);
                transformWatermarkCin = transformCinEnd;
                transformWatermarkCout = transformCoutEnd;
            }

            //滑窗向cin方向滑动blockNumCin块
            blockIterator.Next();
        }
    }

private:
    template <bool NotIdle>
    inline __aicore__ void Mmad(uint32_t batchIdx, uint32_t coutIdx, uint32_t cinIdx)
    {
        uint32_t coutLength = Std::min(cout_ - coutIdx, singleShapeCout_);
        uint32_t coutC1Idx = coutIdx / C0<T>();
        uint32_t coutC1Length = Ops::Base::CeilDiv(coutLength, C0<T>());

        uint32_t cinLength = Std::min(cin_ - cinIdx, singleShapeCin_);
        uint32_t cinC1Idx = cinIdx / C0<T>();
        uint32_t cinC1Length = Ops::Base::CeilDiv(cinLength, C0<T>());

        TileKIterator iter(*this);
        while (iter.More()) {
            HWBox tiles = iter.TileBox();

            aivMTE3ToAicMTE2SyncQue_.WaitData();
            //尾轮时虽然空跑但是队列信号还得照发
            if constexpr (NotIdle) {
                winoMmad_.LoadL1(
                    tiles, batchIdx, iter.kIdx(),
                    nk1c1k0t16c0Dy_, coutC1Idx, coutC1Length,
                    nk1c1k0t16c0Fmap_, cinC1Idx, cinC1Length);
            }

            aivMTE3ToAicMTE2SyncQue_.Pop();

            if constexpr (NotIdle) {
                winoMmad_.Compute(tiles, coutC1Length, cinC1Length, iter.kIdx() == 0);
            }
            iter.Next();
        }
    }

    inline __aicore__ void Transform(
        uint32_t batchIdx,
        uint32_t coutIdx, uint32_t cinIdx,
        uint32_t coutLength, uint32_t cinLength)
    {
        uint32_t aivTaskOffset = 0;
        uint32_t dyTaskCnt = Ops::Base::CeilDiv(coutLength, static_cast<uint32_t>(singleShapeTransformC_));
        uint32_t fmapTaskCnt = Ops::Base::CeilDiv(cinLength, static_cast<uint32_t>(singleShapeTransformC_));

        uint32_t totalTaskCnt = dyTaskCnt + fmapTaskCnt;
        uint32_t blockNumAiv = GetBlockNum() * GetSubBlockNum();

        //only NCHW
        uint64_t srcBatchOffsetDy = batchIdx * cout_ * dy_.SrcH() * dy_.SrcW();
        uint64_t srcBatchOffsetFmap = batchIdx * cin_ * fmap_.SrcH() * fmap_.SrcW();

        LocalTensor<T> transformVBuf[2] = {transformBuf_[0].Get<T>(), transformBuf_[1].Get<T>()};
        LocalTensor<T> transposeVBuf = transposeBuf_.Get<T>();

        TileKIterator iter(*this);
        while (iter.More()) {
            HWBox tile = iter.TileBox();

            aivMTE3ToAicMTE2SyncQue_.WaitSlot();

            //变换任务每轮从前一批空闲的v核开始拉取任务
            //假设每轮只有18个任务,那么第一轮0-17核拉取到任务,第二轮就是18-35核拉取到任务
            //这样可以尽量并行
            for (uint32_t taskId = (GetBlockIdx() + blockNumAiv - aivTaskOffset) % blockNumAiv;
                 taskId < totalTaskCnt;
                 taskId += blockNumAiv) {
                if (taskId < dyTaskCnt) {
                    uint32_t dyTaskId = taskId;
                    uint32_t coutStart = coutIdx + dyTaskId * singleShapeTransformC_;

                    ExecuteTransform(
                        dy_,
                        tile,
                        nk1c1k0t16c0Dy_,
                        batchIdx,
                        srcBatchOffsetDy,
                        coutStart,
                        Std::min(singleShapeTransformC_, coutIdx + coutLength - coutStart),
                        iter.kIdx(),
                        transformVBuf[transformPingPongFlag_],
                        transposeVBuf,
                        transformEventFlags_[transformPingPongFlag_]);
                } else {
                    uint32_t fmapTaskId = taskId - dyTaskCnt;
                    uint32_t cinStart = cinIdx + fmapTaskId * singleShapeTransformC_;

                    ExecuteTransform(
                        fmap_,
                        tile,
                        nk1c1k0t16c0Fmap_,
                        batchIdx,
                        srcBatchOffsetFmap,
                        cinStart,
                        Std::min(singleShapeTransformC_, cinIdx + cinLength - cinStart),
                        iter.kIdx(),
                        transformVBuf[transformPingPongFlag_],
                        transposeVBuf,
                        transformEventFlags_[transformPingPongFlag_]);
                }

                transformPingPongFlag_ = !transformPingPongFlag_;
            }

            aivTaskOffset = (aivTaskOffset + totalTaskCnt) % blockNumAiv;
            //同步等待所有aiv的mte3完成
            CrossCoreSetFlag<0, PIPE_MTE3>(kCROSS_CORE_AIV_SYNC_FLAG);
            CrossCoreWaitFlag<0, PIPE_MTE3>(kCROSS_CORE_AIV_SYNC_FLAG);

            //所有v核mte结束后通知cube消费正变换数据
            aivMTE3ToAicMTE2SyncQue_.Push();

            iter.Next();
        }
    }

    struct TransformVFlag;

    template <typename t0, auto t1, auto t2, typename t3, auto t4>
    static inline __aicore__ void ExecuteTransform(
        const WinoTransformer<t0, t1, t2, t3, t4>& transformer,
        const HWBox& tile,
        const NK1C1K0C0<T>& nk1c1k0c0,
        uint32_t batchIdx,
        uint64_t srcBatchOffset,
        uint32_t cIdx,
        uint32_t cLength,
        uint32_t k1Idx,
        LocalTensor<T>& transformVBuf,
        LocalTensor<T>& transposeVBuf,
        TransformVFlag& eventFlags)
    {
        TileBox box = transformer.CalculateSrcBox(tile, cIdx, cLength);

        WaitFlag<HardEvent::MTE3_MTE2>(eventFlags.mte32mte2);
        transformer.CopyIn(transformVBuf, box, srcBatchOffset);
        SetFlag<HardEvent::MTE2_V>(eventFlags.mte2v);

        WaitFlag<HardEvent::MTE2_V>(eventFlags.mte2v);
        transformer.Compute(transformVBuf, transposeVBuf, box);
        SetFlag<HardEvent::V_MTE3>(eventFlags.v2mte3);

        WaitFlag<HardEvent::V_MTE3>(eventFlags.v2mte3);
        transformer.CopyOut(transformVBuf, nk1c1k0c0, box, batchIdx, k1Idx);
        SetFlag<HardEvent::MTE3_MTE2>(eventFlags.mte32mte2);
    }

    class TileKIterator {
    public:
        __aicore__ inline explicit TileKIterator(ConvBackpropFilterWinograd& winograd): winograd_(winograd)
        {
        }

        __aicore__ inline HWBox TileBox()
        {
            HWBox tile = {};
            tile.hIdx = tileHIdx_;
            tile.wIdx = tileWIdx_;
            tile.hLength = Std::min(static_cast<uint32_t>(winograd_.singleShapeTilesH_), winograd_.tilesH_ - tileHIdx_);
            tile.wLength = Std::min(static_cast<uint32_t>(winograd_.singleShapeTilesW_), winograd_.tilesW_ - tileWIdx_);
            tile.elements = tile.hLength * tile.wLength;
            return tile;
        }

        __aicore__ inline void Next()
        {
            tileWIdx_ += winograd_.singleShapeTilesW_;
            if (tileWIdx_ >= winograd_.tilesW_) {
                tileWIdx_ = 0;
                tileHIdx_ += winograd_.singleShapeTilesH_;
                end_ = tileHIdx_ >= winograd_.tilesH_;
            }
            kIdx_++;
        }

        __aicore__ inline bool More()
        {
            return !end_;
        }

        __aicore__ inline uint32_t kIdx()
        {
            return kIdx_;
        }

    private:
        ConvBackpropFilterWinograd& winograd_;
        uint32_t tileHIdx_ = 0;
        uint32_t tileWIdx_ = 0;
        uint32_t kIdx_ = 0;
        bool end_ = false;
    };

    class BlockIterator {
    public:
        inline __aicore__ explicit BlockIterator(
            uint32_t hCnt,
            uint32_t wCnt,
            uint32_t blockH,
            uint32_t blockW)
            : hCnt_(hCnt),
              wCnt_(wCnt),
              blockH_(blockH),
              blockW_(blockW)
        {
            //沿着 W 轴(Cin)滑窗，所以 H 轴(Cout)必须能被整除以保证阵型不乱
            ascendc_assert((hCnt_ % blockH_) == 0);
            flattenWLength_ = wCnt_ * (hCnt_ / blockH_);

            if ASCEND_IS_AIC {
                currentAicBlockHOffset_ = GetBlockIdx() / blockW_;
                currentAicBlockWOffset_ = GetBlockIdx() % blockW_;
            } else {
                currentAicBlockHOffset_ = 0;
                currentAicBlockWOffset_ = 0;
            }
        }

        inline __aicore__ bool More() const
        {
            return loopIdx_ * blockW_ < flattenWLength_;
        }

        inline __aicore__ bool GetCurrentAicHWIdx(uint32_t& hIdx, uint32_t& wIdx)
        {
            if ASCEND_IS_AIC {
                return GetHWIdx(
                    currentAicBlockHOffset_,
                    currentAicBlockWOffset_,
                    hIdx,
                    wIdx);
            }
            return false;
        }

        inline __aicore__ bool GetHWIdx(
            uint32_t blockHOffset,
            uint32_t blockWOffset,
            uint32_t& outputHIdx,
            uint32_t& outputWIdx)
        {
            ascendc_assert(blockHOffset < blockH_);
            ascendc_assert(blockWOffset < blockW_);

            uint32_t flattenWIdx = loopIdx_ * blockW_ + blockWOffset;

            //不在hCnt和wCnt有效范围内也要设置
            uint32_t q = flattenWIdx / wCnt_;
            outputWIdx = flattenWIdx - q * wCnt_;
            outputHIdx = q * blockH_ + blockHOffset;

            return flattenWIdx < flattenWLength_;
        }

        inline __aicore__ void Next()
        {
            loopIdx_++;
        }

    private:
        const uint32_t hCnt_;
        const uint32_t wCnt_;
        const uint32_t blockH_;
        const uint32_t blockW_;
        uint32_t currentAicBlockHOffset_;
        uint32_t currentAicBlockWOffset_;
        uint32_t flattenWLength_;
        uint32_t loopIdx_ = 0;
    };

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

    template <pipe_t SRC_PIPE, pipe_t DST_PIPE, uint8_t PUSH_FLAG, uint8_t POP_FLAG>
    class CVSyncQue {
        //CrossCoreSetFlag内计数器上限不能超过15
        //这里设置连续push12次就要等待pop通知，防止计数器超限
        static constexpr uint8_t FREE_SLOTS = 12;

    public:
        __aicore__ inline void WaitSlot()
        {
            if (freeSlot == 0) {
                CrossCoreWaitFlag<2, SRC_PIPE>(POP_FLAG);
            }
        }

        __aicore__ inline void Push()
        {
            CrossCoreSetFlag<2, SRC_PIPE>(PUSH_FLAG);
            if (freeSlot > 0) {
                freeSlot--;
            }
        }

        __aicore__ inline void WaitData()
        {
            CrossCoreWaitFlag<2, DST_PIPE>(PUSH_FLAG);
        }

        __aicore__ inline void Pop()
        {
            CrossCoreSetFlag<2, DST_PIPE>(POP_FLAG);
        }

        __aicore__ inline void PipeBarrierAllEnd()
        {
            //如果CrossCoreSetFlag是最后的指令可能因为一执行就核就退出导致没能成功set,整个核结束前加个全量等待
            PipeBarrier<PIPE_ALL>();
        }

    private:
        uint8_t freeSlot = FREE_SLOTS;
    };

    static inline uint32_t __aicore__ SingleShapeTileBufSize(uint32_t c, uint32_t th, uint32_t tw)
    {
        uint32_t hw = TileUnfoldSize(th) * (TileUnfoldSize(tw) + TILE_BUF_BANK_CONFLICT_PADDING);
        return hw * Ops::Base::CeilAlign(c, C0<T>());
    }

    static constexpr uint8_t kCROSS_CORE_AIV_SYNC_FLAG = 0;
    static constexpr uint8_t kCROSS_CORE_AIV2AIC_SEND_FLAG = 1;
    static constexpr uint8_t kCROSS_CORE_AIC2AIV_RECV_FLAG = 2;


    TBuf<TPosition::VECIN> transposeBuf_;
    TBuf<TPosition::VECIN> transformBuf_[2];
    TransformVFlag transformEventFlags_[2];
    bool transformPingPongFlag_ = true;

    CVSyncQue<PIPE_MTE3, PIPE_MTE2,
        kCROSS_CORE_AIV2AIC_SEND_FLAG,
        kCROSS_CORE_AIC2AIV_RECV_FLAG> aivMTE3ToAicMTE2SyncQue_;

    const WinoFmapTransformer<T>& fmap_;
    const WinoDyTransformer<T>& dy_;
    const NK1C1K0C0<T>& nk1c1k0t16c0Fmap_;
    const NK1C1K0C0<T>& nk1c1k0t16c0Dy_;
    WinoMMAD<T>& winoMmad_;

    const uint32_t tilesH_;
    const uint32_t tilesW_;
    const uint32_t cin_;
    const uint32_t cout_;
    const uint16_t singleShapeCin_;
    const uint16_t singleShapeCout_;
    const uint16_t singleShapeTransformC_;
    const uint16_t singleShapeTilesH_;
    const uint16_t singleShapeTilesW_;
    const uint16_t blockNumCin_;
    const uint16_t blockNumCout_;
};


#endif //CONV_BP_WINO_H