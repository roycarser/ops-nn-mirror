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
#include "conv_bp_wino_inv_transform.h"
#include "conv_bp_wino_transform.h"

using namespace AscendC;


namespace WinoDetail {
static constexpr uint8_t CROSS_CORE_AIC_SYNC_FLAG = 0;
static constexpr uint8_t CROSS_CORE_AIV2AIC_SEND_UB2GM_FLAG = 1;
static constexpr uint8_t CROSS_CORE_AIC2AIV_RECV_GM2L1_FLAG = 2;
static constexpr uint8_t CROSS_CORE_AIV2AIC_SEND_UB2L1_FLAG = 3;
static constexpr uint8_t CROSS_CORE_AIC2AIV_RECV_UB2L1_FLAG = 4;
static constexpr uint8_t CROSS_CORE_AIC2AIV_SEND_MMAD_DATA_FLAG = 5;
static constexpr uint8_t CROSS_CORE_AIC2AIV_RECV_MMAD_DATA_FLAG = 6;

template <typename T>
using FwdTransformGM2L1Queue = GM2L1Queue<T,
    CROSS_CORE_AIV2AIC_SEND_UB2GM_FLAG,
    CROSS_CORE_AIC2AIV_RECV_GM2L1_FLAG,
    CROSS_CORE_AIC_SYNC_FLAG>;

template <typename T>
using FwdTransformUB2L1Queue = UB2L1Queue<T,
    CROSS_CORE_AIV2AIC_SEND_UB2L1_FLAG,
    CROSS_CORE_AIC2AIV_RECV_UB2L1_FLAG>;

using InvTransformL0C2UBSyncQueue = CVSyncQue<PIPE_FIX, PIPE_V, PIPE_MTE3,
    CROSS_CORE_AIC2AIV_SEND_MMAD_DATA_FLAG,
    CROSS_CORE_AIC2AIV_RECV_MMAD_DATA_FLAG,
    SINGLE_FREE_SLOTS, true>;

template <typename TilingT>
class TileKIterator {
public:
    __aicore__ inline explicit TileKIterator(
        uint32_t tilesH, uint32_t tilesW)
        : tilesH_(tilesH),
          tilesW_(tilesW)
    {
    }

    __aicore__ inline HWBox TileBox() const
    {
        HWBox tile = {};
        tile.hIdx = tileHIdx_;
        tile.wIdx = tileWIdx_;
        tile.hLength = Std::min(static_cast<uint32_t>(SingleShapeTileH), tilesH_ - tileHIdx_);
        tile.wLength = Std::min(static_cast<uint32_t>(SingleShapeTileW), tilesW_ - tileWIdx_);
        tile.elements = tile.hLength * tile.wLength;
        return tile;
    }

    // __aicore__ inline HWBox TileBox(uint32_t kIdx) const
    // {
    //     uint32_t hStepIdx = kIdx / wStep_;
    //     uint32_t wStepIdx = kIdx - hStepIdx * wStep_;
    //
    //     HWBox tile = {};
    //     tile.hIdx = hStepIdx * SingleShapeTileH;
    //     tile.wIdx = wStepIdx * singleShapeTilesW_;
    //     tile.hLength = Std::min(static_cast<uint32_t>(singleShapeTilesH_), tilesH_ - tile.hIdx);
    //     tile.wLength = Std::min(static_cast<uint32_t>(singleShapeTilesW_), tilesW_ - tile.wIdx);
    //     tile.elements = tile.hLength * tile.wLength;
    //     return tile;
    // }

    __aicore__ inline void Next()
    {
        tileWIdx_ += SingleShapeTileW;
        if (tileWIdx_ >= tilesW_) {
            tileWIdx_ = 0;
            tileHIdx_ += SingleShapeTileH;
            end_ = tileHIdx_ >= tilesH_;
        }
        kIdx_++;
    }

    __aicore__ inline bool More() const
    {
        return !end_;
    }

    __aicore__ inline uint32_t kIdx() const
    {
        return kIdx_;
    }

private:
    constexpr static uint16_t SingleShapeTileH = BlockConfig::SingleShapeTileH<TilingT>();
    constexpr static uint16_t SingleShapeTileW = BlockConfig::SingleShapeTileW<TilingT>();
    const uint32_t tilesH_;
    const uint32_t tilesW_;
    uint32_t tileHIdx_ = 0;
    uint32_t tileWIdx_ = 0;
    uint32_t kIdx_ = 0;
    bool end_ = false;
};

enum BlockIterDirection {
    COUT,
    CIN,
};

struct CoutCinRange {
    uint32_t coutIdx = 0;
    uint32_t cinIdx = 0;
    uint32_t coutLength = 0;
    uint32_t cinLength = 0;

    template <BlockConfig::InputTensor t>
    __aicore__ inline uint32_t GetIdx() const
    {
        if constexpr (t == BlockConfig::InputTensor::FMAP) {
            return cinIdx;
        } else if constexpr (t == BlockConfig::InputTensor::DY) {
            return coutIdx;
        }
    }

    template <BlockConfig::InputTensor t>
    __aicore__ inline uint32_t GetLen() const
    {
        if constexpr (t == BlockConfig::InputTensor::FMAP) {
            return cinLength;
        } else if constexpr (t == BlockConfig::InputTensor::DY) {
            return coutLength;
        }
    }
};

template <BlockIterDirection IterDir, typename TilingT>
class BlockIterator {
public:
    static constexpr uint16_t SingleShapeCout = BlockConfig::SingleShapeCout<TilingT>();
    static constexpr uint16_t SingleShapeCin = BlockConfig::SingleShapeCout<TilingT>();

    inline __aicore__ explicit BlockIterator(
        uint32_t cout,
        uint32_t cin)
        : cout_(cout),
          cin_(cin),
          coutCnt_(Ops::Base::CeilDiv(cout, static_cast<uint32_t>(SingleShapeCout))),
          cinCnt_(Ops::Base::CeilDiv(cin, static_cast<uint32_t>(SingleShapeCin))),
          totalCnt_(coutCnt_ * cinCnt_)
    {
    }

    inline __aicore__ bool More() const
    {
        return loopIdx_ * GetBlockNum() < totalCnt_;
    }

    //获取当前aic计算的基本块范围
    inline __aicore__ bool GetLocalBlock(CoutCinRange& cRange) const
    {
        uint16_t coreId;

        if ASCEND_IS_AIC {
            coreId = GetBlockIdx();
        } else {
            coreId = GetBlockIdx() / GetSubBlockNum();
        }

        return GetBlock(coreId, cRange);
    }

    inline __aicore__ bool GetBlock(uint16_t coreId, CoutCinRange& cRange) const
    {
        uint32_t coutBlockIdx;
        uint32_t cinBlockIdx;
        bool valid = GetCBlockOfCore(coreId, coutBlockIdx, cinBlockIdx);

        cRange.coutIdx = coutBlockIdx * SingleShapeCout;
        cRange.cinIdx = cinBlockIdx * SingleShapeCin;
        cRange.coutLength = valid ? Std::min(SingleShapeCout, cout_ - cRange.coutIdx) : 0;
        cRange.cinLength = valid ? Std::min(SingleShapeCin, cin_ - cRange.cinIdx) : 0;

        return valid;
    }

    //获取本轮全核计算涉及基本块的cout/cin范围最大值
    inline __aicore__ void GetClusterBlockUpperBound(uint32_t& outCoutBound, uint32_t& outCinBound) const
    {
        uint32_t minCoutBlockIdx, minCinBlockIdx;
        GetCBlockOfCore(0, minCoutBlockIdx, minCinBlockIdx);

        uint32_t maxCoutBlockIdx, maxCinBlockIdx;
        GetCBlockOfCore(GetBlockNum() - 1, maxCoutBlockIdx, maxCinBlockIdx);

        uint32_t maxCoutIdx = maxCoutBlockIdx * SingleShapeCout;
        uint32_t maxCinIdx = maxCinBlockIdx * SingleShapeCin;

        if constexpr (IterDir == CIN) {
            outCoutBound = Std::min(maxCoutIdx + SingleShapeCout, cout_);
            outCinBound = maxCoutBlockIdx > minCoutBlockIdx ?
                              cin_ :
                              Std::min(maxCinIdx + SingleShapeCin, cin_);
        } else {
            outCinBound = Std::min(maxCinIdx + SingleShapeCin, cin_);
            outCoutBound = maxCinBlockIdx > minCinBlockIdx ?
                               cout_ :
                               Std::min(maxCoutIdx + SingleShapeCout, cout_);
        }
    }

    inline __aicore__ void Next()
    {
        loopIdx_++;
    }

private:
    inline __aicore__ bool GetCBlockOfCore(
        uint16_t coreId,
        uint32_t& outputCoutIdx,
        uint32_t& outputCinIdx) const
    {
        uint32_t flattenIdx = loopIdx_ * GetBlockNum() + coreId;
        if constexpr (IterDir == CIN) {
            //沿着cin方向递进
            outputCoutIdx = flattenIdx / cinCnt_;
            outputCinIdx = flattenIdx - outputCoutIdx * cinCnt_;
        } else {
            //沿着cout方向递进
            outputCinIdx = flattenIdx / coutCnt_;
            outputCoutIdx = flattenIdx - outputCinIdx * coutCnt_;
        }
        return flattenIdx < totalCnt_;
    }

    const uint32_t cout_;
    const uint32_t cin_;
    const uint32_t coutCnt_;
    const uint32_t cinCnt_;
    const uint32_t totalCnt_;
    uint32_t loopIdx_ = 0;
};

template <typename T, typename TilingT>
class AivFwdTransformer {
public:
    static constexpr uint8_t BUF_CNT = BlockConfig::SingleTransformBufCnt<TilingT>();

    __aicore__ inline AivFwdTransformer(
        const WinoFmapFwdTransformer<T, TilingT>& fmapFwd,
        const WinoDyFwdTransformer<T, TilingT>& dyFwd)
        : fmapFwd_(fmapFwd),
          dyFwd_(dyFwd)
    {
    }

    __aicore__ inline void Init()
    {
        constexpr uint32_t fwdTmpBufSize = GetFwdTmpBufSize();
        constexpr uint32_t fwdSrcBufSize = GetFwdSrcBufSize();
        constexpr uint32_t fwdOutBufSize = GetFwdOutBufSize();
        constexpr uint32_t totalSize = fwdTmpBufSize + fwdSrcBufSize * BUF_CNT + fwdOutBufSize * BUF_CNT;
        static_assert(totalSize * sizeof(T) < TOTAL_UB_SIZE, "exceed ub size limit");

        TBuf<TPosition::VECIN> transformFwdTmpBuf;
        TBuf<TPosition::VECIN> transformFwdSrcBuf;
        TBuf<TPosition::VECIN> transformFwdOutBuf;

        TPipe* pipe = GetTPipePtr();
        pipe->InitBuffer(transformFwdTmpBuf, fwdTmpBufSize * sizeof(T));
        pipe->InitBuffer(transformFwdSrcBuf, fwdSrcBufSize * sizeof(T) * BUF_CNT);
        pipe->InitBuffer(transformFwdOutBuf, fwdOutBufSize * sizeof(T) * BUF_CNT);
        transformFwdTmpVBuf_ = transformFwdTmpBuf.Get<T>();
        transformFwdSrcVBuf_ = transformFwdSrcBuf.Get<T>();
        transformFwdOutVBuf_ = transformFwdOutBuf.Get<T>();

        for (uint8_t i = 0; i < BUF_CNT; i++) {
            transformFwdEventFlags_[i] = TransformVFlag::AllocEventId(pipe);
            SetFlag<HardEvent::V_MTE2>(transformFwdEventFlags_[i].v2mte2);
            SetFlag<HardEvent::MTE3_V>(transformFwdEventFlags_[i].mte32v);
        }
    }

    template <auto D>
    __aicore__ inline void IterateK(
        const BlockIterator<D, TilingT>& blockIter,
        TileKIterator<TilingT>& kIter,
        FwdTransformGM2L1Queue<T>& gm2l1Que,
        FwdTransformUB2L1Queue<T>& ub2l1Que,
        uint32_t batchIdx)
    {
        CoutCinRange localBlock;
        blockIter.GetLocalBlock(localBlock);

        uint32_t clusterCoutBound, clusterCinBound;
        blockIter.GetClusterBlockUpperBound(clusterCoutBound, clusterCinBound);

        using BlockConfig::InputTensor;
        constexpr InputTensor ResidentTarget = BlockConfig::ResidentTarget<TilingT>();
        constexpr InputTensor TensorT0 = ResidentTarget != InputTensor::FMAP ? InputTensor::FMAP : InputTensor::DY;
        constexpr InputTensor TensorT1 = ResidentTarget == InputTensor::FMAP ? InputTensor::FMAP : InputTensor::DY;

        StreamTaskInfo streamT0;
        ComputeT0TaskInfo(localBlock.GetIdx<TensorT0>(), localBlock.GetLen<TensorT0>(), streamT0);

        const uint32_t cBoundT1 = TensorT1 == InputTensor::FMAP ? clusterCinBound : clusterCoutBound;

        StreamTaskInfo streamT1;
        ResidentTaskInfo residentT1;
        ComputeT1TaskInfo(
            localBlock.GetIdx<TensorT1>(), localBlock.GetLen<TensorT1>(),
            cBoundT1,
            BlockConfig::SingleShapeC<TilingT, ResidentTarget>(),
            streamT1, residentT1);

        while (kIter.More()) {
            HWBox tile = kIter.TileBox();

            if (cBoundT1 > watermarkResidentC_) {
                typename TransformFunctions::GM2L1Ctx gm2l1Ctx = {batchIdx, kIter.kIdx(), {gm2l1Que}};
                gm2l1Que.WaitSlot();

                //TODO 全核轮询执行，而非一直从0核开始
                ProcessResidentTransform<TensorT1>(
                    tile,
                    gm2l1Ctx,
                    residentT1);

                gm2l1Que.EnQue();
            }

            typename TransformFunctions::UB2L1Ctx ub2l1Ctx = {batchIdx, kIter.kIdx(), {ub2l1Que, 0}};
            ub2l1Que.WaitSlot();

            ProcessStreamingTransform<TensorT0>(
                tile, ub2l1Ctx, streamT0);

            ProcessStreamingTransform<TensorT1>(
                tile, ub2l1Ctx, streamT1);

            ub2l1Que.EnQue();

            kIter.Next();
        }

        watermarkResidentC_ = Std::max(watermarkResidentC_, cBoundT1);
    }

    __aicore__ inline void End()
    {
        //不wait看文档说状态会残留?
        for (uint8_t i = 0; i < BUF_CNT; i++) {
            WaitFlag<HardEvent::V_MTE2>(transformFwdEventFlags_[i].v2mte2);
            WaitFlag<HardEvent::MTE3_V>(transformFwdEventFlags_[i].mte32v);
        }
    }

private:
    struct StreamTaskInfo;
    struct ResidentTaskInfo;

    __aicore__ inline void ComputeT0TaskInfo(
        uint32_t localCIdx,
        uint16_t localCLen,
        StreamTaskInfo& stream) const
    {
        //当前非驻留矩阵区域
        stream.cIdx = localCIdx;
        stream.cLen = localCLen;
        const uint32_t aivNumInBlock = GetSubBlockNum();
        stream.singleCoreCLen = Ops::Base::CeilDiv(
                                    Ops::Base::CeilDiv(stream.cLen, C0<T>()),
                                    aivNumInBlock) * C0<T>();
    }

    __aicore__ inline void ComputeT1TaskInfo(
        uint32_t localCIdx,
        uint16_t localCLen,
        uint32_t clusterCBound,
        uint16_t singleShapeC,
        StreamTaskInfo& stream,
        ResidentTaskInfo& resident) const
    {
        stream.cIdx = localCIdx + SingleShapeResidentC;
        stream.cLen = Std::max(localCLen, SingleShapeResidentC) - SingleShapeResidentC;
        const uint32_t aivNumInBlock = GetSubBlockNum();
        stream.singleCoreCLen = Ops::Base::CeilDiv(
                                    Ops::Base::CeilDiv(stream.cLen, C0<T>()),
                                    aivNumInBlock) * C0<T>();

        if (clusterCBound > watermarkResidentC_) {
            uint32_t t1FullCLen = clusterCBound - watermarkResidentC_;
            uint32_t t1MainCBlk = t1FullCLen / singleShapeC;
            uint16_t t1TailCLen = t1FullCLen % singleShapeC;

            resident.cIdx = watermarkResidentC_;
            resident.singleShapeTailC = t1TailCLen;
            resident.tailCTaskCnt = Ops::Base::CeilDiv(
                Std::min(t1TailCLen, SingleShapeResidentC),
                SingleShapeTransformC);
            resident.cTaskCnt = resident.tailCTaskCnt + t1MainCBlk * TaskPerSingleResidentC;
        }
    }


    struct TransformFunctions {
        struct GM2L1 {
            FwdTransformGM2L1Queue<T>& queue;
        };

        struct UB2L1 {
            FwdTransformUB2L1Queue<T>& queue;
            uint32_t ub2l1Offset = 0;
        };

        template <typename L1Method>
        struct Context {
            uint32_t batchIdx = 0;
            uint32_t kIdx = 0;
            L1Method l1method;

            __aicore__ inline auto& GetL1Queue()
            {
                return l1method.queue;
            }
        };

        using GM2L1Ctx = Context<GM2L1>;
        using UB2L1Ctx = Context<UB2L1>;

        template <typename TransformConfig, typename L1Method>
        __aicore__ inline static void CopyIn(
            const WinoTransformer<TransformConfig>& transformer,
            const TileBox& box,
            Context<L1Method>& ctx,
            LocalTensor<T>& transformFwdSrcVBuf)
        {
            transformer.CopyIn(
                transformFwdSrcVBuf,
                box,
                ctx.batchIdx);
        }

        template <typename TransformConfig, typename L1Method>
        __aicore__ inline static void Compute(
            const WinoTransformer<TransformConfig>& transformer,
            const TileBox& box,
            Context<L1Method>& dummy,
            LocalTensor<T>& transformFwdSrcVBuf,
            LocalTensor<T>& transformFwdOutVBuf,
            LocalTensor<T>& transformFwdTmpVBuf)
        {
            transformer.Compute(
                transformFwdSrcVBuf,
                transformFwdOutVBuf,
                transformFwdTmpVBuf, box);
        }

        template <typename TransformConfig, typename L1Method>
        __aicore__ inline static void CopyOut(
            const WinoTransformer<TransformConfig>& transformer,
            const TileBox& box,
            Context<L1Method>& ctx,
            LocalTensor<T>& transformFwdOutVBuf)
        {
            NK1C1K0C0::CopyK0Params ckp;
            ckp.batchIdx = ctx.batchIdx;
            ckp.k1Idx = ctx.kIdx;
            transformer.SetNK1C1K0C0CopyParams(ckp, box);

            if constexpr (Std::is_same_v<L1Method, GM2L1>) {
                GM2L1& gm2l1 = ctx.l1method;
                gm2l1.queue.Write(ckp, transformFwdOutVBuf);
            } else {
                UB2L1& ub2l1 = ctx.l1method;
                if constexpr (Std::is_same_v<TransformConfig, WinoTransformDetail::DyConfig<T, TilingT> >) {
                    ub2l1.queue.WriteDy(ckp, transformFwdOutVBuf, ub2l1.ub2l1Offset);
                } else {
                    ub2l1.queue.WriteFmap(ckp, transformFwdOutVBuf, ub2l1.ub2l1Offset);
                }
            }
        }
    };

    struct ResidentTaskInfo {
        uint32_t cIdx;
        uint32_t cTaskCnt;
        uint16_t singleShapeTailC;
        uint16_t tailCTaskCnt;
    };

    template <BlockConfig::InputTensor TransformType>
    __aicore__ inline void ProcessResidentTransform(
        const HWBox& tile,
        typename TransformFunctions::GM2L1Ctx& ctx,
        const ResidentTaskInfo& task)
    {
        using TransformConfig = Std::conditional_t<
            TransformType == BlockConfig::InputTensor::FMAP,
            WinoTransformDetail::FmapConfig<T, TilingT>,
            WinoTransformDetail::DyConfig<T, TilingT> >;

        const uint16_t coreId = GetBlockIdx() * GetSubBlockNum() + GetSubBlockIdx();
        const uint16_t stride = GetSubBlockNum() * GetBlockNum();

        for (uint32_t taskId = coreId;
             taskId < task.cTaskCnt;
             taskId += stride) {
            uint32_t cBlockIdx = taskId / TaskPerSingleResidentC;
            uint32_t taskIdxInCBlock = taskId % TaskPerSingleResidentC;

            uint32_t cBlockOffset = cBlockIdx * BlockConfig::SingleShapeC<TilingT, TransformType>();
            uint32_t offsetInCBlock = taskIdxInCBlock * SingleShapeTransformC;
            bool isTailTask = taskId >= task.cTaskCnt - task.tailCTaskCnt;
            uint32_t cLengthInBlock = SingleShapeResidentC;
            if (isTailTask) {
                cLengthInBlock = Std::min(SingleShapeResidentC, task.singleShapeTailC);
            }
            Execute(
                GetTransformer<TransformType>(),
                ctx,
                TransformFunctions::template CopyIn<TransformConfig, typename TransformFunctions::GM2L1>,
                TransformFunctions::template Compute<TransformConfig, typename TransformFunctions::GM2L1>,
                TransformFunctions::template CopyOut<TransformConfig, typename TransformFunctions::GM2L1>,
                tile, task.cIdx + cBlockOffset, offsetInCBlock, cLengthInBlock);
        }
    }

    struct StreamTaskInfo {
        uint32_t cIdx;
        uint32_t cLen;
        uint16_t singleCoreCLen;
    };

    template <BlockConfig::InputTensor TransformType>
    __aicore__ inline void ProcessStreamingTransform(
        const HWBox& tile,
        typename TransformFunctions::UB2L1Ctx& ctx,
        const StreamTaskInfo& tasks)
    {
        using TransformConfig = Std::conditional_t<
            TransformType == BlockConfig::InputTensor::FMAP,
            WinoTransformDetail::FmapConfig<T, TilingT>,
            WinoTransformDetail::DyConfig<T, TilingT> >;

        const uint32_t cOffset = GetSubBlockIdx() * tasks.singleCoreCLen;
        uint32_t cIdx = tasks.cIdx + cOffset;
        uint32_t cLength = Std::min(tasks.singleCoreCLen, tasks.cLen - cOffset);

        for (uint32_t c = 0;
             c < cLength;
             c += SingleShapeTransformC) {
            //c一定是C0对齐，所以tile元素直接乘上c值就行
            ctx.l1method.ub2l1Offset = tile.elements * F23_TRANSFORM_TILE_ELEMENTS_16 * (cIdx + c);

            Execute(
                GetTransformer<TransformType>(),
                ctx,
                TransformFunctions::template CopyIn<TransformConfig, typename TransformFunctions::UB2L1>,
                TransformFunctions::template Compute<TransformConfig, typename TransformFunctions::UB2L1>,
                TransformFunctions::template CopyOut<TransformConfig, typename TransformFunctions::UB2L1>,
                tile, cIdx, c, cLength);
        }
    }


    template <typename TransformConfig,
        typename Ctx,
        typename CopyIn,
        typename Compute,
        typename CopyOut>
    __aicore__ inline void Execute(
        const WinoTransformer<TransformConfig>& transformer,
        Ctx& ctx, CopyIn copyIn, Compute compute, CopyOut copyOut,
        const HWBox& tile, uint32_t cIdx, uint32_t cStartOffset, uint32_t cLength)
    {
        constexpr uint32_t srcBufLen = GetFwdSrcBufSize();
        constexpr uint32_t outBufLen = GetFwdOutBufSize();
        LocalTensor<T> transformFwdSrcVBuf = transformFwdSrcVBuf_[bufIndex * srcBufLen];
        LocalTensor<T> transformFwdOutVBuf = transformFwdOutVBuf_[bufIndex * outBufLen];
        const TransformVFlag& eventFlag = transformFwdEventFlags_[bufIndex];

        constexpr uint32_t singleShapeTransformC = BlockConfig::SingleTransformC1<TilingT>() * C0<T>();
        uint32_t cStartIdx = cIdx + cStartOffset;
        uint32_t cExeLength = Std::min(singleShapeTransformC, cIdx + cLength - cStartIdx);
        const TileBox box = transformer.CalculateSrcBox(tile, cStartIdx, cExeLength);

        WaitFlag<HardEvent::V_MTE2>(eventFlag.v2mte2);

        copyIn(transformer, box, ctx, transformFwdSrcVBuf);

        SetFlag<HardEvent::MTE2_V>(eventFlag.mte22v);

        WaitFlag<HardEvent::MTE2_V>(eventFlag.mte22v);
        WaitFlag<HardEvent::MTE3_V>(eventFlag.mte32v);

        compute(
            transformer, box, ctx,
            transformFwdSrcVBuf,
            transformFwdOutVBuf,
            transformFwdTmpVBuf_);

        SetFlag<HardEvent::V_MTE2>(eventFlag.v2mte2);
        SetFlag<HardEvent::V_MTE3>(eventFlag.v2mte3);

        WaitFlag<HardEvent::V_MTE3>(eventFlag.v2mte3);

        copyOut(transformer, box, ctx, transformFwdOutVBuf);

        SetFlag<HardEvent::MTE3_V>(eventFlag.mte32v);
        bufIndex = (bufIndex + 1) % BUF_CNT;
    }

    static constexpr __aicore__ inline uint32_t GetFwdTmpBufSize()
    {
        constexpr uint32_t t0 = WinoTransformDetail::GetTmpBufLength<WinoTransformDetail::FmapConfig<T, TilingT> >();
        constexpr uint32_t t1 = WinoTransformDetail::GetTmpBufLength<WinoTransformDetail::DyConfig<T, TilingT> >();
        return ConstexprMaths::Max(t0, t1);
    }

    static constexpr __aicore__ inline uint32_t GetFwdSrcBufSize()
    {
        constexpr uint32_t t0 = WinoTransformDetail::GetInputBufSize<WinoTransformDetail::FmapConfig<T, TilingT> >();
        constexpr uint32_t t1 = WinoTransformDetail::GetInputBufSize<WinoTransformDetail::DyConfig<T, TilingT> >();
        return ConstexprMaths::Max(t0, t1);
    }

    static constexpr __aicore__ inline uint32_t GetFwdOutBufSize()
    {
        constexpr uint32_t t0 = WinoTransformDetail::GetTransformBufSize<WinoTransformDetail::FmapConfig<T,
            TilingT> >();
        constexpr uint32_t t1 = WinoTransformDetail::GetTransformBufSize<WinoTransformDetail::DyConfig<T,
            TilingT> >();
        return ConstexprMaths::Max(t0, t1);
    }

    template <BlockConfig::InputTensor t>
    __aicore__ inline auto& GetTransformer() const
    {
        if constexpr (t == BlockConfig::InputTensor::FMAP) {
            return fmapFwd_;
        } else if (t == BlockConfig::InputTensor::DY) {
            return dyFwd_;
        }
    }

    struct TransformVFlag {
        TEventID mte22v;
        TEventID v2mte2;
        TEventID mte32v;
        TEventID v2mte3;

        static __aicore__ inline TransformVFlag AllocEventId(TPipe* pipe)
        {
            return {
                pipe->AllocEventID<HardEvent::MTE2_V>(),
                pipe->AllocEventID<HardEvent::V_MTE2>(),
                pipe->AllocEventID<HardEvent::MTE3_V>(),
                pipe->AllocEventID<HardEvent::V_MTE3>()
            };
        }
    };

    static constexpr uint16_t SingleShapeResidentC = BlockConfig::SingleShapeResidentC<TilingT>();
    static constexpr uint16_t SingleShapeTransformC = BlockConfig::SingleTransformC1<TilingT>() * C0<T>();
    static constexpr uint16_t TaskPerSingleResidentC = ConstexprMaths::CeilDiv(
        SingleShapeResidentC,
        SingleShapeTransformC);

    const WinoFmapFwdTransformer<T, TilingT>& fmapFwd_;
    const WinoDyFwdTransformer<T, TilingT>& dyFwd_;

    LocalTensor<T> transformFwdTmpVBuf_;
    LocalTensor<T> transformFwdSrcVBuf_;
    LocalTensor<T> transformFwdOutVBuf_;
    TransformVFlag transformFwdEventFlags_[BUF_CNT];

    uint32_t watermarkResidentC_ = 0;
    uint8_t bufIndex = 0;
};


template <typename T, typename TilingT>
class AicMmadComputer {
public:
    static constexpr BlockConfig::InputTensor ResidentTarget = BlockConfig::ResidentTarget<TilingT>();

    __aicore__ inline explicit AicMmadComputer(
        WinoMMAD<T, TilingT>& winoMmad)
        : winoMmad_(winoMmad)
    {
    }

    inline void __aicore__ Init(FwdTransformUB2L1Queue<T>& ub2l1)
    {
        winoMmad_.Init();
        auto l1BufPing = winoMmad_.GetL1Buf(false);
        auto l1BufPong = winoMmad_.GetL1Buf(true);

        constexpr uint8_t FMAP_BUF_IDX = 1;
        constexpr uint8_t DY_BUF_IDX = 0;
        LocalTensor<T> l1FmapBuf[2] = {Std::get<FMAP_BUF_IDX>(l1BufPing), Std::get<FMAP_BUF_IDX>(l1BufPong)};
        LocalTensor<T> l1DyBuf[2] = {Std::get<DY_BUF_IDX>(l1BufPing), Std::get<DY_BUF_IDX>(l1BufPong)};

        ub2l1.Init(l1FmapBuf, l1DyBuf);
    }

    inline void __aicore__ End()
    {
        winoMmad_.End();
    }

    template <auto D>
    __aicore__ inline void IterateK(
        const BlockIterator<D, TilingT>& blockIter,
        TileKIterator<TilingT>& kIter,
        uint32_t batchIdx,
        FwdTransformGM2L1Queue<T>& gm2l1,
        FwdTransformUB2L1Queue<T>& ub2l1,
        InvTransformL0C2UBSyncQueue& l0c2ubSync,
        const LocalTensor<float>& transformInvVBuf)
    {
        uint32_t coutBound, cinBound;
        blockIter.GetClusterBlockUpperBound(coutBound, cinBound);
        uint32_t residentCBound = ResidentTarget == BlockConfig::InputTensor::FMAP ? cinBound : coutBound;

        if (CoutCinRange blockRange; likely(blockIter.GetLocalBlock(blockRange))) {
            RunMmad<true>(
                batchIdx, blockRange, kIter,
                gm2l1, ub2l1, l0c2ubSync,
                transformInvVBuf,
                residentCBound > watermarkResidentC_);
        } else {
            // 闲置核仅参与 Queue 信号同步，维持集群流水线运转，不进行实际 Compute
            RunMmad<false>(
                batchIdx, blockRange, kIter,
                gm2l1, ub2l1, l0c2ubSync,
                transformInvVBuf,
                residentCBound > watermarkResidentC_);
        }
        watermarkResidentC_ = Std::max(watermarkResidentC_, residentCBound);
    }

private:
    template <bool NotIdle>
    __aicore__ inline void RunMmad(
        const uint32_t batchIdx,
        const CoutCinRange& cRange,
        TileKIterator<TilingT>& iter,
        FwdTransformGM2L1Queue<T>& gm2l1,
        FwdTransformUB2L1Queue<T>& ub2l1,
        InvTransformL0C2UBSyncQueue& l0c2ubSync,
        const LocalTensor<float>& transformInvVBuf,
        bool waitResidentTransform)
    {
        static constexpr uint16_t SingleShapeResidentC = BlockConfig::SingleShapeResidentC<TilingT>();
        uint32_t coutC1Length;
        uint32_t cinC1Length;
        uint32_t residentC1Idx;
        uint32_t residentC1Length;

        if constexpr (NotIdle) {
            coutC1Length = Ops::Base::CeilDiv(cRange.coutLength, C0<T>());
            cinC1Length = Ops::Base::CeilDiv(cRange.cinLength, C0<T>());
            if constexpr (ResidentTarget == BlockConfig::InputTensor::FMAP) {
                residentC1Idx = cRange.cinIdx / C0<T>();
                residentC1Length = Ops::Base::CeilDiv(
                    Std::min(cRange.cinLength, SingleShapeResidentC),
                    C0<T>());
            } else {
                residentC1Idx = cRange.coutIdx / C0<T>();
                residentC1Length = Ops::Base::CeilDiv(
                    Std::min(cRange.coutLength, SingleShapeResidentC),
                    C0<T>());
            }
        } else {
            coutC1Length = 0;
            cinC1Length = 0;
            residentC1Idx = 0;
            residentC1Length = 0;
        }

        if (likely(iter.More())) {
            bool loadPingPong = false;
            bool computePingPong = false;

            HWBox tiles = iter.TileBox();
            uint32_t kIdx = iter.kIdx();

            // ================= 阶段 1: Prologue (预载入第一轮数据) =================
            MmadLoadResident<NotIdle>(
                tiles, gm2l1, batchIdx, kIdx,
                residentC1Idx, residentC1Length,
                waitResidentTransform, loadPingPong);

            iter.Next();

            // ================= 阶段 2: Steady State (计算当前轮 + 预载入下一轮) =================
            while (iter.More()) {
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
                HWBox nextTiles = iter.TileBox();
                uint32_t nextKIdx = iter.kIdx();

                MmadLoadResident<NotIdle>(
                    nextTiles, gm2l1, batchIdx, nextKIdx,
                    residentC1Idx, residentC1Length,
                    waitResidentTransform, loadPingPong);

                MmadCompute<NotIdle, false>(
                    tiles, ub2l1, l0c2ubSync,
                    cRange.coutLength, coutC1Length,
                    cRange.cinLength, cinC1Length,
                    kIdx,
                    transformInvVBuf,
                    computePingPong);

                tiles = nextTiles;
                kIdx = nextKIdx;

                iter.Next();
            }

            // ================= 阶段 3: Epilogue (计算最后一轮数据并触发 Fixpipe 落盘) =================
            MmadCompute<NotIdle, true>(
                tiles, ub2l1, l0c2ubSync,
                cRange.coutLength, coutC1Length,
                cRange.cinLength, cinC1Length,
                kIdx,
                transformInvVBuf,
                computePingPong);
        }
    }

    template <bool NotIdle>
    __aicore__ inline void MmadLoadResident(
        const HWBox& tiles,
        FwdTransformGM2L1Queue<T>& gm2l1,
        uint32_t batchIdx,
        uint32_t k1Idx,
        uint32_t c1Idx,
        uint32_t c1Length,
        bool waitResidentFinished,
        bool& l1PingPongFlag)
    {
        if (waitResidentFinished) {
            gm2l1.WaitData();
        }

        if constexpr (NotIdle) {
            NK1C1K0C0::CopyK0Params params;
            params.tiles = tiles.elements;
            params.batchIdx = batchIdx;
            params.k1Idx = k1Idx;
            params.c1Idx = c1Idx;
            params.c1Length = c1Length;

            winoMmad_.template LoadL1<ResidentTarget>(
                gm2l1.GetGlobalTensor(),
                gm2l1.GetGMShape(),
                params,
                l1PingPongFlag);

            l1PingPongFlag = !l1PingPongFlag;
        }

        if (waitResidentFinished) {
            gm2l1.DeQue();
        }
    }

    template <bool NotIdle, bool FixpipeInLastK>
    __aicore__ inline void MmadCompute(
        const HWBox& tiles,
        FwdTransformUB2L1Queue<T>& ub2l1,
        InvTransformL0C2UBSyncQueue& l0c2ubSync,
        uint32_t cout,
        uint32_t coutC1,
        uint32_t cin,
        uint32_t cinC1,
        uint32_t kIdx,
        const LocalTensor<float>& transformInvVBuf,
        bool& l1PingPongFlag)
    {
        // 阻塞等待 AIV 的 DY 生产信号
        ub2l1.WaitData();

        if constexpr (NotIdle) {
            if constexpr (FixpipeInLastK) {
                l0c2ubSync.WaitSlot();
            }

            winoMmad_.template Compute<FixpipeInLastK>(
                tiles,
                cout,
                coutC1,
                cin,
                cinC1,
                kIdx == 0,
                l1PingPongFlag,
                transformInvVBuf);

            if constexpr (FixpipeInLastK) {
                l0c2ubSync.EnQue();
            }
            l1PingPongFlag = !l1PingPongFlag;
        }

        ub2l1.DeQue();
    }

    WinoMMAD<T, TilingT>& winoMmad_;
    uint32_t watermarkResidentC_ = 0;
};
}

template <typename T, typename TilingT>
class ConvBackpropFilterWinograd {
public:
    static constexpr bool ResidentFmap =
        BlockConfig::ResidentTarget<TilingT>() == BlockConfig::InputTensor::FMAP;

    __aicore__ inline ConvBackpropFilterWinograd(
        const WinoFmapFwdTransformer<T, TilingT>& fmap,
        const WinoDyFwdTransformer<T, TilingT>& dy,
        __gm__ T* nk1c1k0c0FmapGm,
        __gm__ T* nk1c1k0c0DyGm,
        __gm__ float* yGm,
        WinoMMAD<T, TilingT>& winoMmad,
        uint32_t tilesH,
        uint32_t tilesW)
        : tilesH_(tilesH),
          tilesW_(tilesW),
          cin_(fmap.SrcC()),
          cout_(dy.SrcC()),
          gm2l1_(
              ResidentFmap ? nk1c1k0c0FmapGm : nk1c1k0c0DyGm,
              NK1C1K0C0::Shape<T>::template Create<TilingT>(
                  ResidentFmap ? cin_ : cout_, tilesH, tilesW)),
          dwFwd_(fmap, dy),
          dwMmad_(winoMmad)
    {
        yGm_.SetGlobalBuffer(yGm);
    }

    inline void __aicore__ Init()
    {
        if ASCEND_IS_AIV {
            dwFwd_.Init();
            //逆变换输出时数据按M轴均分到每个V核上
            dwInv_.Init();
            // uint32_t transformInvOutBufSize = AivPartitioner::Get2DAlignBufLength<float>(
            //                                       singleShapeCout_,
            //                                       singleShapeCin_)
            //                                   * WinoInvTransformer::COUT_CIN_BUF_CNT;

            // TBuf<TPosition::VECIN> transformInvOutBuf;
            // pipe->InitBuffer(transformInvOutBuf, transformInvOutBufSize * sizeof(float));
            // transformInvVBuf_ = transformInvOutBuf.Get<float>();
        }

        // uint32_t singleShapeTileHW = singleShapeTilesH_ * singleShapeTilesW_;
        dwMmad_.Init(ub2l1_);
    }

    inline void __aicore__ End()
    {
        if ASCEND_IS_AIV {
            dwFwd_.End();
        }

        gm2l1_.End();
        ub2l1_.End();
        l0c2ubSync_.End();
        dwMmad_.End();
    }

    inline void __aicore__ IterateAll(
        uint32_t batchIdx)
    {
        using namespace WinoDetail;

        constexpr BlockIterDirection BasicBlockDir = ResidentFmap ? CIN : COUT;
        BlockIterator<BasicBlockDir, TilingT> blockIter(
            cout_,
            cin_);

        while (blockIter.More()) {
            TileKIterator<TilingT> kIter(
                tilesH_,
                tilesW_);

            if ASCEND_IS_AIC {
                dwMmad_.IterateK(
                    blockIter,
                    kIter,
                    batchIdx,
                    gm2l1_,
                    ub2l1_,
                    l0c2ubSync_,
                    transformInvVBuf_);
            }

            if ASCEND_IS_AIV {
                dwFwd_.IterateK(
                    blockIter,
                    kIter,
                    gm2l1_,
                    ub2l1_,
                    batchIdx);

                TransformOutput(blockIter);
            }
            blockIter.Next();
        }
    }

private:
    template <WinoDetail::BlockIterDirection D>
    inline __aicore__ void TransformOutput(
        const WinoDetail::BlockIterator<D, TilingT>& blockIter)
    {
        WinoDetail::CoutCinRange cRange;
        if (!blockIter.GetLocalBlock(cRange)) {
            return;
        }
        l0c2ubSync_.WaitData();

        // dwInv_.PartitionProcess(
        //     yGm_, transformInvVBuf_,
        //     cRange.coutIdx,
        //     cRange.cinIdx,
        //     cRange.coutLength,
        //     cRange.cinLength,
        //     cin_);

        l0c2ubSync_.DeQue();
    }


    const uint32_t tilesH_;
    const uint32_t tilesW_;
    const uint32_t cin_;
    const uint32_t cout_;

    LocalTensor<float> transformInvVBuf_;
    WinoDetail::FwdTransformGM2L1Queue<T> gm2l1_;
    WinoDetail::FwdTransformUB2L1Queue<T> ub2l1_;
    WinoDetail::InvTransformL0C2UBSyncQueue l0c2ubSync_;
    WinoDetail::AivFwdTransformer<T, TilingT> dwFwd_;
    WinoDetail::AicMmadComputer<T, TilingT> dwMmad_;
    WinoInvTransformer dwInv_;
    GlobalTensor<float> yGm_;
};


#endif //CONV_BP_WINO_H