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
        __gm__ T* nk1c1k0c0FmapGm,
        __gm__ T* nk1c1k0c0DyGm,
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
          nk1c1k0c0Fmap_(cin, tilesH, tilesW, singleShapeTilesH, singleShapeTilesW),
          nk1c1k0c0Dy_(cout, tilesH, tilesW, singleShapeTilesH, singleShapeTilesW),
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
        nk1c1k0c0FmapGm_.SetGlobalBuffer(nk1c1k0c0FmapGm);
        nk1c1k0c0DyGm_.SetGlobalBuffer(nk1c1k0c0DyGm);
    }

    inline void __aicore__ Init()
    {
        TPipe* pipe = GetTPipePtr();

        if ASCEND_IS_AIV {
            uint32_t fmapTmpSize = fmap_.CalculateTmpBufLength(singleShapeTilesH_, singleShapeTilesW_);
            uint32_t dyTmpSize = dy_.CalculateTmpBufLength(singleShapeTilesH_, singleShapeTilesW_);
            uint32_t transformBufSize = SingleShapeTileBufSize<T>(
                singleShapeTransformC_,
                singleShapeTilesH_ * singleShapeTilesW_);

            TBuf<TPosition::VECIN> transformTmpBuf;
            TBuf<TPosition::VECIN> transformBuf[2];

            pipe->InitBuffer(transformTmpBuf, Std::max(fmapTmpSize, dyTmpSize) * sizeof(T));
            pipe->InitBuffer(transformBuf[0], transformBufSize * sizeof(T));
            pipe->InitBuffer(transformBuf[1], transformBufSize * sizeof(T));

            tmpVBuf_ = transformTmpBuf.Get<T>();
            transformVBuf_[0] = transformBuf[0].Get<T>();
            transformVBuf_[1] = transformBuf[1].Get<T>();

            TransformVFlag::AllocEventId(pipe, transformEventFlags_[0]);
            TransformVFlag::AllocEventId(pipe, transformEventFlags_[1]);

            //初始的mte2操作不需要等待mte3结束,预先置1
            SetFlag<HardEvent::MTE3_MTE2>(transformEventFlags_[0].mte32mte2);
            SetFlag<HardEvent::MTE3_MTE2>(transformEventFlags_[1].mte32mte2);
        }

        winoMmad_.Init(singleShapeCout_, singleShapeCin_, singleShapeTilesH_ * singleShapeTilesW_);


        auto l1BufPing =  winoMmad_.GetL1Buf(true);
        auto l1BufPong =  winoMmad_.GetL1Buf(false);

        LocalTensor<T> l1DyBuf[2] = {Std::get<0>(l1BufPing), Std::get<0>(l1BufPong)};
        dyUB2L1_.Init(l1DyBuf);
    }

    inline void __aicore__ End()
    {
        if ASCEND_IS_AIV {
            //不wait看文档说状态会残留?
            WaitFlag<HardEvent::MTE3_MTE2>(transformEventFlags_[0].mte32mte2);
            WaitFlag<HardEvent::MTE3_MTE2>(transformEventFlags_[1].mte32mte2);
        }

        fmapGM2L1_.End();
        dyUB2L1_.End();
        winoMmad_.End();
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
                uint32_t transformCoutEnd;
                uint32_t transformCinEnd;

                GetTransformRange(
                    blockIterator,
                    transformWatermarkCout,
                    transformWatermarkCin,
                    transformCoutEnd,
                    transformCinEnd);

                uint32_t currentAicCoutBlockIdx;
                uint32_t currentAicCinBlockIdx;
                bool valid = blockIterator.GetCurrentAicHWIdx(currentAicCoutBlockIdx, currentAicCinBlockIdx);

                //dy只处理当前aiv对应的aic基本块
                uint32_t coutIdx = valid ? currentAicCoutBlockIdx * singleShapeCout_ : 0;
                uint32_t coutLength = valid ? Std::min(cout_ - coutIdx, singleShapeCout_) : 0;

                Transform(
                    batchIdx,
                    coutIdx,
                    transformWatermarkCin,
                    coutLength,
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
        uint32_t coutC1Length = Ops::Base::CeilDiv(coutLength, C0<T>());

        uint32_t cinLength = Std::min(cin_ - cinIdx, singleShapeCin_);
        uint32_t cinC1Idx = cinIdx / C0<T>();
        uint32_t cinC1Length = Ops::Base::CeilDiv(cinLength, C0<T>());

        TileKIterator iter(*this);

        if (iter.More()) {
            //winograd每个点位需要执行16次独立的mad计算
            //由于dav上cube的issue queue大小为16,算上wait flag
            //如果一次最多塞入8条mad指令后就会阻塞,进而block住整个scalar
            //即便按批一次处理4个点，那么加上一个wait flag,也最多处理12个点就block住
            //让下一轮的mte2无法执行,导致整体串行化
            //所以这里用预取下一轮的数据的方式来解决
            //
            // 首次Compute前直接下发PingPong两块L1的搬运指令:
            //  LoadL1 Ping
            //  LoadL1 Pong
            //
            // 然后在L1Ping上做计算,此时scalar单元会由于issue queue满而被阻塞
            //  Compute Ping (block scalar)
            //
            // ComputePing的scalar执行完后在下发L1Ping的搬运指令,即便scalar被卡主也没关系,因为
            // L1Ping搬入时为了下下轮计算,下一轮所需要的L1Pong已经被预载了
            //  LoadL1 Ping
            //
            // 下发L1Pong的计算指令,由于L1Pong的搬运指令已经提前下发,所以在vector正变换更得上的情况下L1Pong应该搬运的差不多了
            // ComputePong可以立刻执行
            //  Compute Pong

            bool loadPingPong = false;
            bool computePingPong = loadPingPong;

            HWBox tiles = iter.TileBox();
            uint32_t kIdx = iter.kIdx();

            MmadLoadGMFmap<NotIdle>(
                tiles, batchIdx, kIdx, cinC1Idx, cinC1Length,
                loadPingPong);

            iter.Next();

            while (iter.More()) {
                HWBox nextTiles = iter.TileBox();
                uint32_t nextKIdx = iter.kIdx();

                MmadLoadGMFmap<NotIdle>(
                    nextTiles, batchIdx, nextKIdx,
                    cinC1Idx, cinC1Length,
                    loadPingPong);

                MmadCompute<NotIdle>(tiles, coutC1Length, cinC1Length, kIdx, computePingPong);

                tiles = nextTiles;
                kIdx = nextKIdx;

                iter.Next();
            }
            MmadCompute<NotIdle>(tiles, coutC1Length, cinC1Length, kIdx, computePingPong);
        }
    }

    template <bool NotIdle>
    inline __aicore__ void MmadLoadGMFmap(
        const HWBox& tiles, uint32_t batchIdx, uint32_t k1Idx,
        uint32_t cinC1Idx, uint32_t cinC1Length, bool& l1PingPongFlag)
    {
        fmapGM2L1_.WaitData();

        //尾轮时虽然空跑但是队列信号还得照发
        if constexpr (NotIdle) {
            NK1C1K0C0::CopyK0Params<T> copyFmapParams;
            copyFmapParams.tiles = tiles.elements;
            copyFmapParams.batchIdx = batchIdx;
            copyFmapParams.k1Idx = k1Idx;
            copyFmapParams.c1Idx = cinC1Idx;
            copyFmapParams.c1Length = cinC1Length;
            copyFmapParams.gm = nk1c1k0c0FmapGm_;

            winoMmad_.LoadL1Fmap(nk1c1k0c0Fmap_, copyFmapParams, l1PingPongFlag);
            l1PingPongFlag = !l1PingPongFlag;
        }

        fmapGM2L1_.DeQue();
    }

    template <bool NotIdle>
    inline __aicore__ void MmadCompute(
        const HWBox& tiles, uint32_t coutC1, uint32_t cinC1, uint32_t kIdx, bool& l1PingPongFlag)
    {
        dyUB2L1_.WaitData();
        if constexpr (NotIdle) {
            winoMmad_.Compute(tiles, coutC1, cinC1, kIdx == 0, l1PingPongFlag);
            l1PingPongFlag = !l1PingPongFlag;
        }

        dyUB2L1_.DeQue();
    }

    class BlockIterator;

    inline void __aicore__ GetTransformRange(
        BlockIterator& blockIterator,
        uint32_t transformWatermarkCout,
        uint32_t transformWatermarkCin,
        uint32_t& transformCoutEndOut, uint32_t& transformCinEndOut)
    {
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
        transformCoutEndOut = Std::max(transformCoutEndIdx, transformWatermarkCout);
        transformCinEndOut = Std::max(transformCinEndIdx, transformWatermarkCin);
    }

    inline __aicore__ void Transform(
        uint32_t batchIdx,
        uint32_t coutIdx, uint32_t cinIdx,
        uint32_t coutLength, uint32_t cinLength)
    {
        uint32_t dyTaskCnt = Ops::Base::CeilDiv(coutLength, static_cast<uint32_t>(singleShapeTransformC_));
        uint32_t fmapTaskCnt = Ops::Base::CeilDiv(cinLength, static_cast<uint32_t>(singleShapeTransformC_));

        //dy每次由单个Block内的aiv处理
        uint32_t dyStride = GetSubBlockNum();
        uint32_t dyCoreId = GetSubBlockIdx();

        //fmap分发到所有核上
        uint32_t fmapStride = GetSubBlockNum() * GetBlockNum();
        uint32_t fmapCoreId = GetBlockIdx() * GetSubBlockNum() + GetSubBlockIdx();

        //only NCHW
        uint64_t srcBatchOffsetDy = batchIdx * cout_ * dy_.SrcH() * dy_.SrcW();
        uint64_t srcBatchOffsetFmap = batchIdx * cin_ * fmap_.SrcH() * fmap_.SrcW();

        uint32_t dyTaskOffset = 0;
        uint32_t fmapTaskOffset = 0;

        TileKIterator iter(*this);
        while (iter.More()) {
            HWBox tile = iter.TileBox();

            // aivMTE3ToAicMTE2SyncQue_.WaitSlot();

            //变换任务每轮从前一批空闲的v核开始拉取任务
            //假设每轮只有18个任务,那么第一轮0-17核拉取到任务,第二轮就是18-35核拉取到任务
            //这样可以尽量并行
            NK1C1K0C0::CopyK0Params<T> copyK0Params;
            copyK0Params.batchIdx = batchIdx;
            copyK0Params.k1Idx = iter.kIdx();
            copyK0Params.gm = nk1c1k0c0FmapGm_;

            fmapGM2L1_.WaitSlot();
            for (uint32_t fmapTaskId = (fmapCoreId + fmapStride - fmapTaskOffset) % fmapStride;
                 fmapTaskId < fmapTaskCnt;
                 fmapTaskId += fmapStride) {
                uint32_t cinStart = cinIdx + fmapTaskId * singleShapeTransformC_;

                ExecuteTransform<false>(
                    fmap_,
                    tile,
                    nk1c1k0c0Fmap_,
                    copyK0Params,
                    srcBatchOffsetFmap,
                    cinStart,
                    Std::min(singleShapeTransformC_, cinIdx + cinLength - cinStart),
                    0);
            }
            //TODO fmap的任务控制block尽量小
            fmapGM2L1_.EnQue();

            copyK0Params.gm = nk1c1k0c0DyGm_;
            dyUB2L1_.WaitSlot();
            uint32_t dyUB2L1Offset = tile.elements * F23_TRANSFORM_TILE_ELEMENTS_16 * singleShapeTransformC_;

            for (uint32_t dyTaskId = (dyCoreId + dyStride - dyTaskOffset) % dyStride;
                 dyTaskId < dyTaskCnt;
                 dyTaskId += dyStride) {
                uint32_t coutStart = coutIdx + dyTaskId * singleShapeTransformC_;
                uint32_t l1Offset = dyTaskId * dyUB2L1Offset;

                ExecuteTransform<true>(
                    dy_,
                    tile,
                    nk1c1k0c0Dy_,
                    copyK0Params,
                    srcBatchOffsetDy,
                    coutStart,
                    Std::min(singleShapeTransformC_, coutIdx + coutLength - coutStart),
                    l1Offset);
            }
            dyUB2L1_.EnQue();

            dyTaskOffset = (dyTaskOffset + dyTaskCnt) % dyStride;
            fmapTaskOffset = (fmapTaskOffset + fmapTaskCnt) % fmapStride;

            iter.Next();
        }
    }


    struct TransformVFlag;

    template <bool dy, typename t0, auto t1, auto t2, typename t3, auto t4>
    inline __aicore__ void ExecuteTransform(
        const WinoTransformer<t0, t1, t2, t3, t4>& transformer,
        const HWBox& tile,
        const NK1C1K0C0::Shape<T>& nk1c1k0c0,
        NK1C1K0C0::CopyK0Params<T>& copyK0Params,
        uint64_t srcBatchOffset,
        uint32_t cIdx,
        uint32_t cLength,
        uint32_t ub2l1Offset)
    {
        TileBox box = transformer.CalculateSrcBox(tile, cIdx, cLength);
        LocalTensor<T>& transformVBuf = transformVBuf_[transformPingPongFlag_];
        TransformVFlag& eventFlags = transformEventFlags_[transformPingPongFlag_];

        WaitFlag<HardEvent::MTE3_MTE2>(eventFlags.mte32mte2);
        transformer.CopyIn(transformVBuf, box, srcBatchOffset);
        SetFlag<HardEvent::MTE2_V>(eventFlags.mte2v);

        WaitFlag<HardEvent::MTE2_V>(eventFlags.mte2v);
        transformer.Compute(transformVBuf, tmpVBuf_, box);
        SetFlag<HardEvent::V_MTE3>(eventFlags.v2mte3);

        WaitFlag<HardEvent::V_MTE3>(eventFlags.v2mte3);
        transformer.SetNK1C1K0C0CopyParams(
            copyK0Params,
            transformVBuf,
            box);

        if constexpr (dy) {
            dyUB2L1_.Write(copyK0Params, ub2l1Offset);
        } else {
            fmapGM2L1_.Write(copyK0Params, nk1c1k0c0);
        }

        SetFlag<HardEvent::MTE3_MTE2>(eventFlags.mte32mte2);
        transformPingPongFlag_ = !transformPingPongFlag_;
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
                currentAicBlockHOffset_ = GetBlockIdx() / GetSubBlockNum() / blockW_;
                currentAicBlockWOffset_ = GetBlockIdx() / GetSubBlockNum() % blockW_;
            }
        }

        inline __aicore__ bool More() const
        {
            return loopIdx_ * blockW_ < flattenWLength_;
        }

        inline __aicore__ bool GetCurrentAicHWIdx(uint32_t& hIdx, uint32_t& wIdx)
        {
            return GetHWIdx(
                currentAicBlockHOffset_,
                currentAicBlockWOffset_,
                hIdx,
                wIdx);
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

            //不在hCnt和wCnt有效范围内也设置
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


    static constexpr uint8_t CROSS_CORE_AIC_SYNC_FLAG = 0;
    static constexpr uint8_t CROSS_CORE_AIV2AIC_SEND_UB2GM_FLAG = 1;
    static constexpr uint8_t CROSS_CORE_AIC2AIV_RECV_GM2L1_FLAG = 2;
    static constexpr uint8_t CROSS_CORE_AIV2AIC_SEND_UB2L1_FLAG = 3;
    static constexpr uint8_t CROSS_CORE_AIC2AIV_RECV_UB2L1_FLAG = 4;

    LocalTensor<T> tmpVBuf_;
    LocalTensor<T> transformVBuf_[2];
    TransformVFlag transformEventFlags_[2];
    bool transformPingPongFlag_ = true;

    GM2L1Queue<T,
        CROSS_CORE_AIV2AIC_SEND_UB2GM_FLAG,
        CROSS_CORE_AIC2AIV_RECV_GM2L1_FLAG,
        CROSS_CORE_AIC_SYNC_FLAG> fmapGM2L1_;

    UB2L1Queue<T,
        CROSS_CORE_AIV2AIC_SEND_UB2L1_FLAG,
        CROSS_CORE_AIC2AIV_RECV_UB2L1_FLAG> dyUB2L1_;

    const WinoFmapTransformer<T>& fmap_;
    const WinoDyTransformer<T>& dy_;
    const NK1C1K0C0::Shape<T> nk1c1k0c0Fmap_;
    const NK1C1K0C0::Shape<T> nk1c1k0c0Dy_;
    GlobalTensor<T> nk1c1k0c0FmapGm_;
    GlobalTensor<T> nk1c1k0c0DyGm_;
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