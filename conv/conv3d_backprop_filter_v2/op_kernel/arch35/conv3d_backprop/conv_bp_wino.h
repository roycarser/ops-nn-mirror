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


enum FwdTransformGMResidentTarget {
    FMAP,
    DY,
};

namespace WinoDetail {
static constexpr uint8_t CROSS_CORE_AIC_SYNC_FLAG = 0;
static constexpr uint8_t CROSS_CORE_AIV2AIC_SEND_UB2GM_FLAG = 1;
static constexpr uint8_t CROSS_CORE_AIC2AIV_RECV_GM2L1_FLAG = 2;
static constexpr uint8_t CROSS_CORE_AIV2AIC_SEND_UB2L1_FLAG = 3;
static constexpr uint8_t CROSS_CORE_AIC2AIV_RECV_UB2L1_FLAG = 4;
static constexpr uint8_t CROSS_CORE_AIC2AIV_SEND_MMAD_DATA_FLAG = 5;
static constexpr uint8_t CROSS_CORE_AIC2AIV_RECV_MMAD_DATA_FLAG = 6;
static constexpr uint8_t CROSS_CORE_AIV_PRE_TRANSPOSE_SYNC_FLAG = 7;

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

class TileKIterator {
public:
    __aicore__ inline explicit TileKIterator(
        uint32_t tilesH, uint32_t tilesW,
        uint16_t singleShapeTileH,
        uint16_t singleShapeTileW)
        : tilesH_(tilesH),
          tilesW_(tilesW),
          singleShapeTilesH_(singleShapeTileH),
          singleShapeTilesW_(singleShapeTileW),
          wStep_(Ops::Base::CeilDiv(tilesW, static_cast<uint32_t>(singleShapeTileW))),
          kCnt_(wStep_ * Ops::Base::CeilDiv(tilesH, static_cast<uint32_t>(singleShapeTileH)))
    {
    }

    __aicore__ inline HWBox TileBox() const
    {
        HWBox tile = {};
        tile.hIdx = tileHIdx_;
        tile.wIdx = tileWIdx_;
        tile.hLength = Std::min(static_cast<uint32_t>(singleShapeTilesH_), tilesH_ - tileHIdx_);
        tile.wLength = Std::min(static_cast<uint32_t>(singleShapeTilesW_), tilesW_ - tileWIdx_);
        tile.elements = tile.hLength * tile.wLength;
        return tile;
    }

    __aicore__ inline HWBox TileBox(uint32_t kIdx) const
    {
        uint32_t hStepIdx = kIdx / wStep_;
        uint32_t wStepIdx = kIdx - hStepIdx * wStep_;

        HWBox tile = {};
        tile.hIdx = hStepIdx * singleShapeTilesH_;
        tile.wIdx = wStepIdx * singleShapeTilesW_;
        tile.hLength = Std::min(static_cast<uint32_t>(singleShapeTilesH_), tilesH_ - tile.hIdx);
        tile.wLength = Std::min(static_cast<uint32_t>(singleShapeTilesW_), tilesW_ - tile.wIdx);
        tile.elements = tile.hLength * tile.wLength;
        return tile;
    }

    __aicore__ inline void Next()
    {
        tileWIdx_ += singleShapeTilesW_;
        if (tileWIdx_ >= tilesW_) {
            tileWIdx_ = 0;
            tileHIdx_ += singleShapeTilesH_;
        }
        kIdx_++;
        end_ = kIdx_ >= kCnt_;
    }

    __aicore__ inline uint32_t TotalK() const
    {
        return kCnt_;
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
    const uint32_t tilesH_;
    const uint32_t tilesW_;
    const uint16_t singleShapeTilesH_;
    const uint16_t singleShapeTilesW_;
    const uint32_t wStep_;
    const uint32_t kCnt_;
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
};

template <BlockIterDirection IterDir>
class BlockIterator {
public:
    inline __aicore__ explicit BlockIterator(
        uint32_t cout,
        uint32_t cin,
        uint16_t singleShapeCout,
        uint16_t singleShapeCin)
        : cout_(cout),
          cin_(cin),
          coutCnt_(Ops::Base::CeilDiv(cout, static_cast<uint32_t>(singleShapeCout))),
          cinCnt_(Ops::Base::CeilDiv(cin, static_cast<uint32_t>(singleShapeCin))),
          totalCnt_(coutCnt_ * cinCnt_),
          singleShapeCout_(singleShapeCout),
          singleShapeCin_(singleShapeCin)
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

        cRange.coutIdx = coutBlockIdx * singleShapeCout_;
        cRange.cinIdx = cinBlockIdx * singleShapeCin_;
        cRange.coutLength = valid ? Std::min(singleShapeCout_, cout_ - cRange.coutIdx) : 0;
        cRange.cinLength = valid ? Std::min(singleShapeCin_, cin_ - cRange.cinIdx) : 0;

        return valid;
    }

    //获取本轮全核计算涉及基本块的cout/cin范围最大值
    inline __aicore__ void GetClusterBlockUpperBound(uint32_t& outCoutBound, uint32_t& outCinBound) const
    {
        uint32_t minCoutBlockIdx, minCinBlockIdx;
        GetCBlockOfCore(0, minCoutBlockIdx, minCinBlockIdx);

        uint32_t maxCoutBlockIdx, maxCinBlockIdx;
        GetCBlockOfCore(GetBlockNum() - 1, maxCoutBlockIdx, maxCinBlockIdx);

        uint32_t maxCoutIdx = maxCoutBlockIdx * singleShapeCout_;
        uint32_t maxCinIdx = maxCinBlockIdx * singleShapeCin_;

        if constexpr (IterDir == CIN) {
            outCoutBound = Std::min(maxCoutIdx + singleShapeCout_, cout_);
            outCinBound = maxCoutBlockIdx > minCoutBlockIdx ?
                              cin_ :
                              Std::min(maxCinIdx + singleShapeCin_, cin_);
        } else {
            outCinBound = Std::min(maxCinIdx + singleShapeCin_, cin_);
            outCoutBound = maxCinBlockIdx > minCinBlockIdx ?
                               cout_ :
                               Std::min(maxCoutIdx + singleShapeCout_, cout_);
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
    const uint16_t singleShapeCout_;
    const uint16_t singleShapeCin_;
    uint32_t loopIdx_ = 0;
};

template <FwdTransformGMResidentTarget ResidentTarget, typename T>
class AivFwdTransformer {
public:
    __aicore__ inline AivFwdTransformer(
        const WinoFmapFwdTransformer<T>& fmapFwd,
        const WinoDyFwdTransformer<T>& dyFwd,
        const uint16_t singleShapeTransformC)
        : fmapFwd_(fmapFwd),
          dyFwd_(dyFwd),
          singleShapeTransformC_(singleShapeTransformC)
    {
    }

    __aicore__ inline void Init(uint16_t singleShapeTilesH, uint16_t singleShapeTilesW)
    {
        uint32_t transformFwdTmpBufSize = Std::max(
            fmapFwd_.GetTmpBufLength(singleShapeTilesH, singleShapeTilesW),
            dyFwd_.GetTmpBufLength(singleShapeTilesH, singleShapeTilesW));

        uint32_t transformFwdSrcBufSize = Std::max(
            fmapFwd_.GetInputBufSize(singleShapeTransformC_, singleShapeTilesH, singleShapeTilesW),
            dyFwd_.GetInputBufSize(singleShapeTransformC_, singleShapeTilesH, singleShapeTilesW));

        uint32_t transformFwdOutBufSize = Std::max(
            fmapFwd_.GetOutputBufSize(singleShapeTransformC_, singleShapeTilesH, singleShapeTilesW),
            dyFwd_.GetOutputBufSize(singleShapeTransformC_, singleShapeTilesH, singleShapeTilesW));

        TBuf<TPosition::VECIN> transformFwdTmpBuf;
        TBuf<TPosition::VECIN> transformFwdSrcBuf[2];
        TBuf<TPosition::VECIN> transformFwdOutBuf[2];

        TPipe* pipe = GetTPipePtr();
        pipe->InitBuffer(transformFwdTmpBuf, transformFwdTmpBufSize * sizeof(T));
        pipe->InitBuffer(transformFwdSrcBuf[0], transformFwdSrcBufSize * sizeof(T));
        pipe->InitBuffer(transformFwdSrcBuf[1], transformFwdSrcBufSize * sizeof(T));
        pipe->InitBuffer(transformFwdOutBuf[0], transformFwdOutBufSize * sizeof(T));
        pipe->InitBuffer(transformFwdOutBuf[1], transformFwdOutBufSize * sizeof(T));

        transformFwdTmpVBuf_ = transformFwdTmpBuf.Get<T>();
        transformFwdSrcVBuf_[0] = transformFwdSrcBuf[0].Get<T>();
        transformFwdSrcVBuf_[1] = transformFwdSrcBuf[1].Get<T>();
        transformFwdOutVBuf_[0] = transformFwdOutBuf[0].Get<T>();
        transformFwdOutVBuf_[1] = transformFwdOutBuf[1].Get<T>();

        TransformVFlag::AllocEventId(pipe, transformFwdEventFlags_[0]);
        TransformVFlag::AllocEventId(pipe, transformFwdEventFlags_[1]);

        //初始的mte2和v不需要等v和mte3执行,预先置1
        SetFlag<HardEvent::V_MTE2>(transformFwdEventFlags_[0].v2mte2);
        SetFlag<HardEvent::V_MTE2>(transformFwdEventFlags_[1].v2mte2);
        SetFlag<HardEvent::MTE3_V>(transformFwdEventFlags_[0].mte32v);
        SetFlag<HardEvent::MTE3_V>(transformFwdEventFlags_[1].mte32v);
    }

    __aicore__ inline void IterateK(
        const BlockIterator<ResidentTarget == FMAP ? CIN : COUT>& blockIter,
        TileKIterator& kIter,
        FwdTransformGM2L1Queue<T>& gm2l1Que,
        FwdTransformUB2L1Queue<T>& ub2l1Que,
        uint32_t batchIdx)
    {
        CoutCinRange localBlock;
        blockIter.GetLocalBlock(localBlock);

        uint32_t clusterCoutBound, clusterCinBound;
        blockIter.GetClusterBlockUpperBound(clusterCoutBound, clusterCinBound);

        //变换当前单核基本块范围
        const uint32_t streamCIdx = ResidentTarget != FMAP ? localBlock.cinIdx : localBlock.coutIdx;
        const uint32_t streamCLength = ResidentTarget != FMAP ? localBlock.cinLength : localBlock.coutLength;

        TaskInfo<TaskType::LOCAL_BLOCK> streamTaskInfo = {
            Ops::Base::CeilDiv(streamCLength, static_cast<uint32_t>(singleShapeTransformC_)),
            0};

        //驻留处理，全核全局处理，提取全局基本块范围
        const uint32_t residentClusterBound = ResidentTarget == FMAP ? clusterCinBound : clusterCoutBound;
        const uint32_t residentCIdx = watermarkResidentC_;
        const uint32_t residentCLength = residentClusterBound > watermarkResidentC_ ?
                                             residentClusterBound - watermarkResidentC_ :
                                             0;

        TaskInfo<TaskType::CLUSTER> residentTaskInfo = {
            Ops::Base::CeilDiv(residentCLength, static_cast<uint32_t>(singleShapeTransformC_)),
            0};

        //PreTranspose：和GM驻留处理一样提取全局基本块范围全核一起计算
        const uint32_t streamClusterBound = ResidentTarget != FMAP ? clusterCinBound : clusterCoutBound;
        const uint32_t preTransCIdx = watermarkPreTransposeC_;
        const uint32_t preTransCLength = streamClusterBound > watermarkPreTransposeC_ ?
                                             streamClusterBound - watermarkPreTransposeC_ :
                                             0;

        const uint32_t preTransCTaskCnt = Ops::Base::CeilDiv(
            preTransCLength,
            static_cast<uint32_t>(singleShapeTransformC_));
        const uint16_t totalAivNum = GetBlockNum() * GetSubBlockNum();
        const uint32_t preTransKLookAhead = preTransCTaskCnt > 0 ?
                                                Std::max(1, totalAivNum / preTransCTaskCnt) :
                                                0;

        while (kIter.More()) {
            HWBox tile = kIter.TileBox();

            typename PreTransposeFunctions::Context preTransCtx = {batchIdx, kIter.kIdx()};
            const bool transposed = ProcessPreTranspose(
                kIter, preTransCtx, preTransKLookAhead,
                preTransCIdx,
                preTransCLength,
                preTransCTaskCnt);

            if (transposed) {
                CrossCoreSetFlag<0, PIPE_MTE3>(CROSS_CORE_AIV_PRE_TRANSPOSE_SYNC_FLAG);
            }

            typename TransformFunctions::GM2L1Ctx gm2l1Ctx = {batchIdx, kIter.kIdx(), {gm2l1Que}};
            ProcessResidentTransform(tile, gm2l1Ctx, residentCIdx, residentCLength, residentTaskInfo);
            residentTaskInfo.UpdateOffset();

            if (transposed) {
                CrossCoreWaitFlag<0, PIPE_MTE2>(CROSS_CORE_AIV_PRE_TRANSPOSE_SYNC_FLAG);
            }

            typename TransformFunctions::UB2L1Ctx ub2l1Ctx = {batchIdx, kIter.kIdx(), {ub2l1Que, 0}};
            ProcessStreamingTransform(tile, ub2l1Ctx, streamCIdx, streamCLength, streamTaskInfo);
            streamTaskInfo.UpdateOffset();

            kIter.Next();
        }

        watermarkResidentC_ = Std::max(watermarkResidentC_, residentClusterBound);
        watermarkPreTransposeC_ = Std::max(watermarkPreTransposeC_, streamClusterBound);
    }

    __aicore__ inline void End()
    {
        //不wait看文档说状态会残留?
        WaitFlag<HardEvent::V_MTE2>(transformFwdEventFlags_[0].v2mte2);
        WaitFlag<HardEvent::V_MTE2>(transformFwdEventFlags_[1].v2mte2);
        WaitFlag<HardEvent::MTE3_V>(transformFwdEventFlags_[0].mte32v);
        WaitFlag<HardEvent::MTE3_V>(transformFwdEventFlags_[1].mte32v);
    }

private:
    struct TransformFunctions {
        struct GM2L1 {
            FwdTransformGM2L1Queue<T>& queue;
        };

        struct UB2L1 {
            FwdTransformUB2L1Queue<T>& queue;
            uint32_t ub2l1Offset;
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

        template <typename TransformConfig, bool PreTranspose, typename L1Method>
        __aicore__ inline static void CopyIn(
            const WinoTransformer<TransformConfig>& transformer,
            const TileBox& box,
            Context<L1Method>& ctx,
            LocalTensor<T>& transformFwdSrcVBuf)
        {
            transformer.template CopyIn<PreTranspose>(
                transformFwdSrcVBuf,
                box,
                ctx.batchIdx,
                ctx.kIdx);
        }

        template <typename TransformConfig, bool PreTranspose, typename L1Method>
        __aicore__ inline static void Compute(
            const WinoTransformer<TransformConfig>& transformer,
            const TileBox& box,
            Context<L1Method>& dummy,
            LocalTensor<T>& transformFwdSrcVBuf,
            LocalTensor<T>& transformFwdOutVBuf,
            LocalTensor<T>& transformFwdTmpVBuf)
        {
            transformer.template Compute<PreTranspose>(
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
                ub2l1.queue.Write(ckp, transformFwdOutVBuf, ub2l1.ub2l1Offset);
            }
        }
    };

    enum TaskType {
        LOCAL_BLOCK,
        CLUSTER
    };

    template <TaskType Type>
    struct TaskInfo {
        const uint32_t count;
        uint32_t offset;

        __aicore__ inline uint16_t GetCoreId() const
        {
            if constexpr (Type == LOCAL_BLOCK) {
                return GetSubBlockIdx();
            } else {
                return GetBlockIdx() * GetSubBlockNum() + GetSubBlockIdx();
            }
        }

        __aicore__ inline uint16_t GetStride() const
        {
            if constexpr (Type == LOCAL_BLOCK) {
                return GetSubBlockNum();
            } else {
                return GetSubBlockNum() * GetBlockNum();
            }
        }

        __aicore__ inline uint32_t StartTaskId() const
        {
            const uint16_t coreId = GetCoreId();
            const uint16_t stride = GetStride();
            return (coreId + stride - offset) % stride;
        }

        __aicore__ inline void UpdateOffset()
        {
            offset = (offset + count) % GetStride();
        }
    };

    __aicore__ inline void ProcessResidentTransform(
        const HWBox& tile,
        typename TransformFunctions::GM2L1Ctx& ctx,
        uint32_t cIdx, uint32_t cLength,
        const TaskInfo<TaskType::CLUSTER>& taskInfo)
    {
        FwdTransformGM2L1Queue<T>& gm2l1 = ctx.GetL1Queue();
        gm2l1.WaitSlot();

        using TransformConfig = Std::conditional_t<
            ResidentTarget == FMAP,
            WinoTransformDetail::FmapConfig<T>,
            WinoTransformDetail::DyConfig<T> >;

        for (uint32_t taskId = taskInfo.StartTaskId();
             taskId < taskInfo.count;
             taskId += taskInfo.GetStride()) {
            Execute(
                GetTransformer<ResidentTarget == FMAP>(),
                ctx,
                TransformFunctions::template CopyIn<TransformConfig, false, typename TransformFunctions::GM2L1>,
                TransformFunctions::template Compute<TransformConfig, false, typename TransformFunctions::GM2L1>,
                TransformFunctions::template CopyOut<TransformConfig, typename TransformFunctions::GM2L1>,
                tile, cIdx, taskId * singleShapeTransformC_, cLength);
        }

        gm2l1.EnQue();
    }

    __aicore__ inline void ProcessStreamingTransform(
        const HWBox& tile,
        typename TransformFunctions::UB2L1Ctx& ctx,
        uint32_t cIdx, uint32_t cLength,
        const TaskInfo<TaskType::LOCAL_BLOCK>& taskInfo)
    {
        FwdTransformUB2L1Queue<T>& ub2l1 = ctx.GetL1Queue();
        ub2l1.WaitSlot();

        const uint32_t ub2L1Offset = tile.elements * F23_TRANSFORM_TILE_ELEMENTS_16 * singleShapeTransformC_;

        using TransformConfig = Std::conditional_t<
            ResidentTarget != FMAP,
            WinoTransformDetail::FmapConfig<T>,
            WinoTransformDetail::DyConfig<T> >;

        for (uint32_t taskId = taskInfo.StartTaskId();
             taskId < taskInfo.count;
             taskId += taskInfo.GetStride()) {
            ctx.l1method.ub2l1Offset = taskId * ub2L1Offset;

            Execute(
                GetTransformer<ResidentTarget != FMAP>(),
                ctx,
                TransformFunctions::template CopyIn<TransformConfig, true, typename TransformFunctions::UB2L1>,
                TransformFunctions::template Compute<TransformConfig, true, typename TransformFunctions::UB2L1>,
                TransformFunctions::template CopyOut<TransformConfig, typename TransformFunctions::UB2L1>,
                tile, cIdx, taskId * singleShapeTransformC_, cLength);
        }

        ub2l1.EnQue();
    }


    struct PreTransposeFunctions {
        struct Context {
            uint32_t batchIdx;
            uint32_t kIdx;
        };

        template <typename TransformConfig>
        __aicore__ inline static void CopyIn(
            const WinoTransformer<TransformConfig>& transformer,
            const TileBox& box,
            Context& ctx,
            LocalTensor<T>& transformFwdSrcVBuf)
        {
            transformer.template CopyIn<false>(
                transformFwdSrcVBuf,
                box,
                ctx.batchIdx,
                ctx.kIdx);
        }

        template <typename TransformConfig>
        __aicore__ inline static void Compute(
            const WinoTransformer<TransformConfig>& transformer,
            const TileBox& box,
            Context& dummy0,
            LocalTensor<T>& transformFwdSrcVBuf,
            LocalTensor<T>& transformFwdOutVBuf,
            LocalTensor<T>& dummy1)
        {
            transformer.PreTranspose(
                transformFwdSrcVBuf,
                transformFwdOutVBuf,
                box);
        }

        template <typename TransformConfig>
        __aicore__ inline static void CopyOut(
            const WinoTransformer<TransformConfig>& transformer,
            const TileBox& box,
            Context& ctx,
            LocalTensor<T>& transformFwdOutVBuf)
        {
            transformer.CopyPreTransposeOut(
                transformFwdOutVBuf,
                box,
                ctx.batchIdx,
                ctx.kIdx);
        }
    };

    __aicore__ inline bool ProcessPreTranspose(
        const TileKIterator& tileKIter,
        typename PreTransposeFunctions::Context& ctx,
        uint32_t kLookAhead,
        uint32_t cIdx, uint32_t cLength, uint32_t cTaskCnt)
    {
        using TransformConfig = Std::conditional_t<
            ResidentTarget == FMAP,
            WinoTransformDetail::DyConfig<T>,
            WinoTransformDetail::FmapConfig<T> >;

        //每隔kLookAhead个k计算一次
        if (kLookAhead == 0 || tileKIter.kIdx() % kLookAhead != 0 || cTaskCnt == 0) {
            return false;
        }

        uint32_t transposeK = Std::min(kLookAhead, tileKIter.TotalK() - tileKIter.kIdx());
        TaskInfo<TaskType::CLUSTER> taskInfo = {cTaskCnt * transposeK, 0};

        for (uint32_t taskId = taskInfo.StartTaskId();
             taskId < taskInfo.count;
             taskId += taskInfo.GetStride()) {
            uint32_t kOffset = taskId / cTaskCnt;
            uint32_t cTaskId = taskId % cTaskCnt;

            ctx.kIdx = tileKIter.kIdx() + kOffset;
            HWBox tile = tileKIter.TileBox(ctx.kIdx);

            Execute(
                GetTransformer<ResidentTarget != FMAP>(),
                ctx,
                PreTransposeFunctions::template CopyIn<TransformConfig>,
                PreTransposeFunctions::template Compute<TransformConfig>,
                PreTransposeFunctions::template CopyOut<TransformConfig>,
                tile, cIdx, cTaskId * singleShapeTransformC_, cLength);
        }

        return true;
    }

    template <typename TransformConfig, typename Ctx, typename CopyIn, typename Compute, typename CopyOut>
    __aicore__ inline void Execute(
        const WinoTransformer<TransformConfig>& transformer,
        Ctx& ctx, CopyIn copyIn, Compute compute, CopyOut copyOut,
        const HWBox& tile, uint32_t cIdx, uint32_t cStartOffset, uint32_t cLength)
    {
        LocalTensor<T>& transformFwdSrcVBuf = transformFwdSrcVBuf_[transformFwdPingPongFlag_];
        LocalTensor<T>& transformFwdOutVBuf = transformFwdOutVBuf_[transformFwdPingPongFlag_];
        TransformVFlag& eventFlags = transformFwdEventFlags_[transformFwdPingPongFlag_];

        uint32_t cStartIdx = cIdx + cStartOffset;
        uint32_t cExeLength = Std::min(singleShapeTransformC_, cIdx + cLength - cStartIdx);
        const TileBox box = transformer.CalculateSrcBox(tile, cStartIdx, cExeLength);

        WaitFlag<HardEvent::V_MTE2>(eventFlags.v2mte2);

        copyIn(transformer, box, ctx, transformFwdSrcVBuf);

        SetFlag<HardEvent::MTE2_V>(eventFlags.mte22v);

        WaitFlag<HardEvent::MTE2_V>(eventFlags.mte22v);
        WaitFlag<HardEvent::MTE3_V>(eventFlags.mte32v);

        compute(
            transformer, box, ctx,
            transformFwdSrcVBuf,
            transformFwdOutVBuf,
            transformFwdTmpVBuf_);

        SetFlag<HardEvent::V_MTE2>(eventFlags.v2mte2);
        SetFlag<HardEvent::V_MTE3>(eventFlags.v2mte3);

        WaitFlag<HardEvent::V_MTE3>(eventFlags.v2mte3);

        copyOut(transformer, box, ctx, transformFwdOutVBuf);

        SetFlag<HardEvent::MTE3_V>(eventFlags.mte32v);
        transformFwdPingPongFlag_ = !transformFwdPingPongFlag_;
    }

    template <bool GetFmapFwd>
    __aicore__ inline auto& GetTransformer() const
    {
        if constexpr (GetFmapFwd) {
            return fmapFwd_;
        } else {
            return dyFwd_;
        }
    }

    struct TransformVFlag {
        TEventID mte22v;
        TEventID v2mte2;
        TEventID mte32v;
        TEventID v2mte3;

        static __aicore__ inline void AllocEventId(TPipe* pipe, TransformVFlag& flags)
        {
            flags.mte22v = pipe->AllocEventID<HardEvent::MTE2_V>();
            flags.v2mte2 = pipe->AllocEventID<HardEvent::V_MTE2>();
            flags.mte32v = pipe->AllocEventID<HardEvent::MTE3_V>();
            flags.v2mte3 = pipe->AllocEventID<HardEvent::V_MTE3>();
        }
    };

    const WinoFmapFwdTransformer<T>& fmapFwd_;
    const WinoDyFwdTransformer<T>& dyFwd_;

    LocalTensor<T> transformFwdTmpVBuf_;
    LocalTensor<T> transformFwdSrcVBuf_[2];
    LocalTensor<T> transformFwdOutVBuf_[2];
    TransformVFlag transformFwdEventFlags_[2];

    uint32_t watermarkPreTransposeC_ = 0;
    uint32_t watermarkResidentC_ = 0;
    const uint16_t singleShapeTransformC_;
    bool transformFwdPingPongFlag_ = false;
};


template <FwdTransformGMResidentTarget ResidentTarget, typename T>
class AicMmadComputer {
public:
    __aicore__ inline explicit AicMmadComputer(
        WinoMMAD<T>& winoMmad)
        : winoMmad_(winoMmad)
    {
    }

    inline void __aicore__ Init(
        uint16_t singleShapeCout, uint16_t singleShapeCin, uint32_t singleShapeTileHW,
        FwdTransformUB2L1Queue<T>& ub2l1)
    {
        winoMmad_.Init(singleShapeCout, singleShapeCin, singleShapeTileHW);
        auto l1BufPing = winoMmad_.GetL1Buf(false);
        auto l1BufPong = winoMmad_.GetL1Buf(true);

        constexpr uint8_t DY_BUF_IDX = 0;
        constexpr uint8_t FMAP_BUF_IDX = 1;
        constexpr uint8_t UB2L1BUF_IDX = ResidentTarget == FMAP ? DY_BUF_IDX : FMAP_BUF_IDX;
        LocalTensor<T> l1Buf[2] = {Std::get<UB2L1BUF_IDX>(l1BufPing), Std::get<UB2L1BUF_IDX>(l1BufPong)};
        ub2l1.Init(l1Buf);
    }

    inline void __aicore__ End()
    {
        winoMmad_.End();
    }

    __aicore__ inline void IterateK(
        const BlockIterator<ResidentTarget == FMAP ? CIN : COUT>& blockIter,
        TileKIterator& kIter,
        uint32_t batchIdx,
        FwdTransformGM2L1Queue<T>& gm2l1,
        FwdTransformUB2L1Queue<T>& ub2l1,
        InvTransformL0C2UBSyncQueue& l0c2ubSync,
        const LocalTensor<float>& transformInvVBuf)
    {
        CoutCinRange block0Range;
        blockIter.GetBlock(0, block0Range);
        bool residentFinished = ResidentTarget == FMAP ? block0Range.coutIdx > 0 : block0Range.cinIdx > 0;

        if (CoutCinRange blockRange; likely(blockIter.GetLocalBlock(blockRange))) {
            RunMmad<true>(
                batchIdx, blockRange, kIter,
                gm2l1, ub2l1, l0c2ubSync,
                transformInvVBuf,
                residentFinished);
        } else {
            // 闲置核仅参与 Queue 信号同步，维持集群流水线运转，不进行实际 Compute
            RunMmad<false>(
                batchIdx, blockRange, kIter,
                gm2l1, ub2l1, l0c2ubSync,
                transformInvVBuf,
                residentFinished);
        }
    }

private:
    template <bool NotIdle>
    __aicore__ inline void RunMmad(
        const uint32_t batchIdx,
        const CoutCinRange& cRange,
        TileKIterator& iter,
        FwdTransformGM2L1Queue<T>& gm2l1,
        FwdTransformUB2L1Queue<T>& ub2l1,
        InvTransformL0C2UBSyncQueue& l0c2ubSync,
        const LocalTensor<float>& transformInvVBuf,
        bool residentFinished)
    {
        uint32_t coutC1Length;
        uint32_t cinC1Length;
        uint32_t residentC1Idx;
        uint32_t residentC1Length;

        if constexpr (NotIdle) {
            coutC1Length = Ops::Base::CeilDiv(cRange.coutLength, C0<T>());
            cinC1Length = Ops::Base::CeilDiv(cRange.cinLength, C0<T>());
            if constexpr (ResidentTarget == FMAP) {
                residentC1Idx = cRange.cinIdx / C0<T>();
                residentC1Length = cinC1Length;
            } else {
                residentC1Idx = cRange.coutIdx / C0<T>();
                residentC1Length = coutC1Length;
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
                residentFinished, loadPingPong);

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
                    residentFinished, loadPingPong);

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
        bool residentFinished,
        bool& l1PingPongFlag)
    {
        gm2l1.WaitData(residentFinished);

        if constexpr (NotIdle) {
            NK1C1K0C0::CopyK0Params copyFmapParams;
            copyFmapParams.tiles = tiles.elements;
            copyFmapParams.batchIdx = batchIdx;
            copyFmapParams.k1Idx = k1Idx;
            copyFmapParams.c1Idx = c1Idx;
            copyFmapParams.c1Length = c1Length;

            winoMmad_.template LoadL1<ResidentTarget == FMAP>(
                gm2l1.GetGlobalTensor(),
                gm2l1.GetGMShape(),
                copyFmapParams,
                l1PingPongFlag);

            l1PingPongFlag = !l1PingPongFlag;
        }

        gm2l1.DeQue();
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

    WinoMMAD<T>& winoMmad_;
};
}

template <FwdTransformGMResidentTarget ResidentTarget, typename T>
class ConvBackpropFilterWinograd {
public:
    __aicore__ inline ConvBackpropFilterWinograd(
        const WinoFmapFwdTransformer<T>& fmap,
        const WinoDyFwdTransformer<T>& dy,
        __gm__ T* nk1c1k0c0FmapGm,
        __gm__ T* nk1c1k0c0DyGm,
        __gm__ float* yGm,
        WinoMMAD<T>& winoMmad,
        uint32_t tilesH,
        uint32_t tilesW,
        uint16_t singleShapeCin,
        uint16_t singleShapeCout,
        uint16_t singleShapeTransformC,
        uint16_t singleShapeTilesH,
        uint16_t singleShapeTilesW)
        : tilesH_(tilesH),
          tilesW_(tilesW),
          cin_(fmap.SrcC()),
          cout_(dy.SrcC()),
          singleShapeCin_(singleShapeCin),
          singleShapeCout_(singleShapeCout),
          singleShapeTilesH_(singleShapeTilesH),
          singleShapeTilesW_(singleShapeTilesW),
          gm2l1_(
              ResidentTarget == FMAP ? nk1c1k0c0FmapGm : nk1c1k0c0DyGm,
              NK1C1K0C0::Shape<T>(
                  ResidentTarget == FMAP ? cin_ : cout_, tilesH, tilesW,
                  singleShapeTilesH, singleShapeTilesW)),
          dwFwd_(fmap, dy, singleShapeTransformC),
          dwMmad_(winoMmad)
    {
        yGm_.SetGlobalBuffer(yGm);
    }

    inline void __aicore__ Init()
    {
        if ASCEND_IS_AIV {
            dwFwd_.Init(singleShapeTilesH_, singleShapeTilesW_);
            //逆变换输出时数据按M轴均分到每个V核上
            dwInv_.Init();
            uint32_t transformInvOutBufSize = AivPartitioner::Get2DAlignBufLength<float>(
                                                  singleShapeCout_,
                                                  singleShapeCin_)
                                              * WinoInvTransformer::COUT_CIN_BUF_CNT;

            // TBuf<TPosition::VECIN> transformInvOutBuf;
            // pipe->InitBuffer(transformInvOutBuf, transformInvOutBufSize * sizeof(float));
            // transformInvVBuf_ = transformInvOutBuf.Get<float>();
        }

        uint32_t singleShapeTileHW = singleShapeTilesH_ * singleShapeTilesW_;
        dwMmad_.Init(singleShapeCout_, singleShapeCin_, singleShapeTileHW, ub2l1_);
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

        constexpr BlockIterDirection BasicBlockDir = ResidentTarget == FMAP ? CIN : COUT;
        BlockIterator<BasicBlockDir> blockIter(
            cout_,
            cin_,
            singleShapeCout_,
            singleShapeCin_);

        while (blockIter.More()) {
            TileKIterator kIter(
                tilesH_,
                tilesW_,
                singleShapeTilesH_,
                singleShapeTilesW_);

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
        const WinoDetail::BlockIterator<D>& blockIter)
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
    const uint16_t singleShapeCin_;
    const uint16_t singleShapeCout_;
    const uint16_t singleShapeTilesH_;
    const uint16_t singleShapeTilesW_;

    LocalTensor<float> transformInvVBuf_;
    WinoDetail::FwdTransformGM2L1Queue<T> gm2l1_;
    WinoDetail::FwdTransformUB2L1Queue<T> ub2l1_;
    WinoDetail::InvTransformL0C2UBSyncQueue l0c2ubSync_;
    WinoDetail::AivFwdTransformer<ResidentTarget, T> dwFwd_;
    WinoDetail::AicMmadComputer<ResidentTarget, T> dwMmad_;
    WinoInvTransformer dwInv_;
    GlobalTensor<float> yGm_;
};


#endif //CONV_BP_WINO_H