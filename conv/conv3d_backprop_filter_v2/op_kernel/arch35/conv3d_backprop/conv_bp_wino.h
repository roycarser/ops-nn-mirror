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

template <typename TilingT>
using InvTransformL0C2UBSyncQueue = CVSyncQue<CVSyncQueConfig<PIPE_FIX, PIPE_V, PIPE_MTE3,
    CROSS_CORE_AIC2AIV_SEND_MMAD_DATA_FLAG,
    CROSS_CORE_AIC2AIV_RECV_MMAD_DATA_FLAG,
    BlockConfig::InvTransformBufCnt<TilingT>(), true> >;

template <typename TilingT>
class BatchTileKIterator {
public:
    //支持切K,K的维度为[batch,CeilDiv(tileH,SingleShapeTileH)]
    //当前简单点不切tileW
    __aicore__ inline explicit BatchTileKIterator(
        uint32_t batch, uint32_t tilesH, uint32_t tilesW,
        uint32_t kBegin, uint32_t kLength)
        : batch_(batch),
          tilesH_(tilesH),
          tilesW_(tilesW),
          hSteps_(Ops::Base::CeilDiv(tilesH, SingleShapeTileH)),
          wSteps_(Ops::Base::CeilDiv(tilesW, SingleShapeTileW)),
          kBegin_(kBegin),
          kLength_(kLength)
    {
        Update(0, 0);
    }

    __aicore__ inline explicit BatchTileKIterator(
        uint32_t batch, uint32_t tilesH, uint32_t tilesW)
        : batch_(batch),
          tilesH_(tilesH),
          tilesW_(tilesW),
          hSteps_(Ops::Base::CeilDiv(tilesH, SingleShapeTileH)),
          wSteps_(Ops::Base::CeilDiv(tilesW, SingleShapeTileW)),
          kBegin_(0),
          kLength_(batch * hSteps_)
    {
        Update(0, 0);
    }

    __aicore__ inline HWBox TileBox() const
    {
        HWBox tile = {};
        tile.hIdx = tileHIdx_;
        tile.wIdx = tileWIdx_;
        tile.hLength = Std::min(SingleShapeTileH, tilesH_ - tileHIdx_);
        tile.wLength = Std::min(SingleShapeTileW, tilesW_ - tileWIdx_);
        tile.elements = tile.hLength * tile.wLength;
        return tile;
    }


    __aicore__ inline void Next()
    {
        tileWIdx_ += SingleShapeTileW;
        if (tileWIdx_ >= tilesW_) {
            processedKStep_++;
            if (processedKStep_ >= kLength_) {
                end_ = true;
            } else {
                Update(processedKStep_, 0);
            }
        } else {
            Update(processedKStep_, tileWIdx_);
        }
    }

    __aicore__ inline bool More() const
    {
        return !end_;
    }

    __aicore__ inline uint32_t TileKIdx() const
    {
        return tileKIdx_;
    }

    __aicore__ inline uint32_t BatchIdx() const
    {
        return batchIdx_;
    }

private:
    __aicore__ inline void Update(uint32_t processedSteps, uint32_t tileWIdx)
    {
        uint32_t kStep = processedSteps + kBegin_;
        batchIdx_ = kStep / hSteps_;
        uint32_t singleShapeTileHIdx = kStep - batchIdx_ * hSteps_;
        tileHIdx_ = singleShapeTileHIdx * SingleShapeTileH;
        tileWIdx_ = tileWIdx;
        uint32_t singleShapeTileWIdx = tileWIdx_ / SingleShapeTileW;
        tileKIdx_ = singleShapeTileHIdx * wSteps_ + singleShapeTileWIdx;
    }

    constexpr static uint32_t SingleShapeTileH = BlockConfig::SingleShapeTileH<TilingT>();
    constexpr static uint32_t SingleShapeTileW = BlockConfig::SingleShapeTileW<TilingT>();
    const uint32_t tilesH_;
    const uint32_t tilesW_;
    const uint32_t batch_;
    const uint32_t hSteps_;
    const uint32_t wSteps_;
    const uint32_t kBegin_;
    const uint32_t kLength_;
    uint32_t tileHIdx_ = 0;
    uint32_t tileWIdx_ = 0;
    uint32_t batchIdx_ = 0;
    uint32_t tileKIdx_ = 0;
    uint32_t processedKStep_ = 0;
    bool end_ = false;
};


class SwizzleTopology2D {
public:
    //实现简单的Tile和蛇形走位，所有核构成一个blockHW块进行递进，提升L2cache的命中率
    //尾轮自适应，仅最后一轮才会产生空转
    //
    //                         blockW(4)
    //                   |-----------------------|
    //                -  +-----+-----+-----+-----+-----+-----+-----+
    //                |  |core0|core2|core4|core6|core0|core2|core4|
    //      blockH(2)-|  +-----+-----+-----+-----+-----+-----+-----+
    //                |  |core1|core3|core5|core7|core1|core3|core5|
    //                -  +-----+-----+-----+-----+-----+-----+-----+  HCnt
    //                   |core4|core3|core2|core1|core0|core7|core6|
    //                   +-----+-----+-----+-----+-----+-----+-----+
    //                                      WCnt

    __aicore__ static inline void CalBlockGrid(uint32_t h, uint32_t w, uint16_t& outBlockH, uint16_t& outBlockW)
    {
        uint16_t coreNum = GetBlockNum();
        uint16_t bestH = 1;
        uint16_t bestW = coreNum;

        //常用核数配置直接写死，不用再去跑一遍循环
        if (coreNum == 32) {
            bestH = 4;
            bestW = 8;
        } else if (coreNum == 28) {
            bestH = 4;
            bestW = 7;
        } else if (coreNum == 36) {
            bestH = 6;
            bestW = 6;
        } else {
            for (uint16_t i = 1; i * i <= coreNum; i++) {
                if (coreNum % i == 0) {
                    bestH = i;
                    bestW = coreNum / i;
                }
            }
        }

        // 形状匹配：将较大的维度分配给张量中较大的那个轴，进一步减少跨行/跨列跳跃
        if (h >= w) {
            outBlockH = Std::max(bestH, bestW);
            outBlockW = Std::min(bestH, bestW);
        } else {
            outBlockH = Std::min(bestH, bestW);
            outBlockW = Std::max(bestH, bestW);
        }
    }

    __aicore__ inline SwizzleTopology2D(
        uint32_t h,
        uint32_t w,
        uint16_t blockH,
        uint16_t blockW)
        : h_(h), w_(w), blockH_(blockH), blockW_(blockW),
          fullSuperRows_(h / blockH),
          totalCnt_(h * w)
    {
    }

    __aicore__ inline bool GetHW(uint32_t loopIdx, uint16_t coreId, uint32_t& outH, uint32_t& outW) const
    {
        uint32_t flattenIdx = loopIdx * GetBlockNum() + coreId;

        // 拦截越界
        if (unlikely(flattenIdx >= totalCnt_)) {
            outH = h_;
            outW = w_;
            return false;
        }

        uint32_t dummy;
        ComputeHW(flattenIdx, outH, outW, dummy);
        return true;
    }

    //包围盒计算：计算当前轮次在 H 和 W 方向触达的最远逻辑边界
    __aicore__ inline void GetBoundHW(uint32_t loopIdx, uint32_t& boundH, uint32_t& boundW) const
    {
        uint32_t startIdx = loopIdx * GetBlockNum();
        if (unlikely(startIdx >= totalCnt_)) {
            boundH = 0;
            boundW = 0;
            return;
        }

        uint32_t endIdx = Std::min(startIdx + GetBlockNum(), totalCnt_) - 1;

        uint32_t dummyH1, startW, startSuperIdx;
        ComputeHW(startIdx, dummyH1, startW, startSuperIdx);

        uint32_t dummyH2, endW, endSuperIdx;
        ComputeHW(endIdx, dummyH2, endW, endSuperIdx);

        // ================= W 轴边界检测,由于存在蛇形走位，需要按照奇偶额外判断 =================
        if (startSuperIdx == endSuperIdx) {
            // 1. 未换行：直接取最大值
            boundW = Std::max(startW, endW);
        } else if (endSuperIdx - startSuperIdx >= 2) {
            // 2. 跨越多行：中间必然包含一个完整的偶数行，绝对会撞击右侧墙壁
            boundW = w_ - 1;
        } else {
            // 3. 恰好相邻跨越 1 行
            if (startSuperIdx % 2 == 0) {
                // 偶切奇：在右侧墙壁折返，必然触碰 w_ - 1
                boundW = w_ - 1;
            } else {
                // 奇切偶：在左侧墙壁(W=0)折返，极值由起点或终点决定
                boundW = Std::max(startW, endW);
            }
        }
        // =========================================================

        boundH = endSuperIdx * blockH_ + Std::min(blockH_, h_ - endSuperIdx * blockH_) - 1;
    }

    __aicore__ inline uint32_t TotalCnt() const
    {
        return totalCnt_;
    }

private:
    __aicore__ inline void ComputeHW(
        uint32_t flattenIdx,
        uint32_t& outH, uint32_t& outW,
        uint32_t& outSuperIdx) const
    {
        const uint32_t superRowElements = blockH_ * w_;
        const uint32_t fullSuperRowElements = fullSuperRows_ * superRowElements;
        const uint32_t superIdx = flattenIdx < fullSuperRowElements ? flattenIdx / superRowElements : fullSuperRows_;
        const uint32_t localIdx = flattenIdx - superIdx * superRowElements;
        const uint32_t superRowH = superIdx * blockH_;
        const uint32_t localBlockH = Std::min(blockH_, h_ - superRowH);
        //每个BlockHW里面按H方向优先递进,也就是连续核的范围为(H0,W0),(H1,W0),(H2,W0)
        //列H方向优先按当前实现起来较为简单
        outH = superRowH + localIdx % localBlockH;
        const uint32_t forwardW = localIdx / localBlockH;
        //蛇形走位，先从头走到尾，在从尾走到头
        outW = (superIdx % 2 == 0) ? forwardW : (w_ - 1 - forwardW);
        outSuperIdx = superIdx;
    }

    const uint32_t h_;
    const uint32_t w_;
    const uint16_t blockH_;
    const uint16_t blockW_;
    const uint32_t fullSuperRows_;
    const uint32_t totalCnt_;
};

enum BlockIterDirection {
    COUT,
    CIN,
};

template <BlockIterDirection IterDir>
static __aicore__ inline bool GetBlockFromSwizzle2D(
    const SwizzleTopology2D& topology, uint32_t loopIdx,
    uint16_t coreId,
    uint32_t singleShapeCout, uint32_t singleShapeCin,
    uint32_t cout, uint32_t cin,
    CoutCinRange& cRange)
{
    uint32_t topoH, topoW;
    bool valid = topology.GetHW(loopIdx, coreId, topoH, topoW);

    uint32_t coutBlockIdx = (IterDir == CIN) ? topoH : topoW;
    uint32_t cinBlockIdx = (IterDir == CIN) ? topoW : topoH;

    cRange.coutIdx = coutBlockIdx * singleShapeCout;
    cRange.cinIdx = cinBlockIdx * singleShapeCin;
    cRange.coutLength = valid ? Std::min(singleShapeCout, cout - cRange.coutIdx) : 0;
    cRange.cinLength = valid ? Std::min(singleShapeCin, cin - cRange.cinIdx) : 0;

    return valid;
}

template <BlockIterDirection IterDir, typename TilingT, bool OnlyIterMainBlocks = false>
class BlockIterator {
public:
    static constexpr uint32_t SingleShapeCout = BlockConfig::SingleShapeCout<TilingT>();
    static constexpr uint32_t SingleShapeCin = BlockConfig::SingleShapeCin<TilingT>();

    inline __aicore__ bool More() const
    {
        return loopIdx_ < blocksIterCnt_;
    }

    //获取当前aic计算的基本块范围,若当前核无基本块计算则返回false并且将length设置为0
    inline __aicore__ bool GetLocalBlock(CoutCinRange& cRange) const
    {
        return GetBlock(AicCoreId(), cRange);
    }

    inline __aicore__ bool GetBlock(uint16_t coreId, CoutCinRange& cRange) const
    {
        return GetBlockFromSwizzle2D<IterDir>(
            topology_, loopIdx_, coreId,
            SingleShapeCout, SingleShapeCin,
            cout_, cin_,
            cRange);
    }

    //获取本轮全核计算涉及基本块的cout/cin范围最大值
    inline __aicore__ void GetClusterBlockUpperBound(uint32_t& outCoutBound, uint32_t& outCinBound) const
    {
        uint32_t boundH, boundW;
        topology_.GetBoundHW(loopIdx_, boundH, boundW);

        // 边界反向映射
        uint32_t boundCoutBlockIdx = (IterDir == CIN) ? boundH : boundW;
        uint32_t boundCinBlockIdx = (IterDir == CIN) ? boundW : boundH;

        // 转化为实际的空间维度绝对边界
        outCoutBound = Std::min((boundCoutBlockIdx + 1) * SingleShapeCout, cout_);
        outCinBound = Std::min((boundCinBlockIdx + 1) * SingleShapeCin, cin_);
    }

    inline __aicore__ void Next()
    {
        loopIdx_++;
    }

    inline __aicore__ const SwizzleTopology2D& GetSwizzleTopology() const
    {
        return topology_;
    }

    inline __aicore__ uint32_t GetTailBlockCnt() const
    {
        const uint32_t mainBlockNum = blocksIterCnt_ * GetBlockNum();
        return topology_.TotalCnt() > mainBlockNum ? topology_.TotalCnt() - mainBlockNum : 0;
    }

    static inline __aicore__ BlockIterator Create(uint32_t cout, uint32_t cin)
    {
        uint32_t coutCnt = Ops::Base::CeilDiv(cout, SingleShapeCout);
        uint32_t cinCnt = Ops::Base::CeilDiv(cin, SingleShapeCin);
        uint32_t topologyH = (IterDir == CIN) ? coutCnt : cinCnt;
        uint32_t topologyW = (IterDir == CIN) ? cinCnt : coutCnt;
        uint16_t blockH, blockW;
        SwizzleTopology2D::CalBlockGrid(topologyH, topologyW, blockH, blockW);
        return BlockIterator(cout, cin, topologyH, topologyW, blockH, blockW);
    }

private:
    inline __aicore__ explicit BlockIterator(
        uint32_t cout,
        uint32_t cin,
        uint32_t topologyH,
        uint32_t topologyW,
        uint16_t topologyBlockH,
        uint16_t topologyBlockW)
        : cout_(cout),
          cin_(cin),
          topology_(topologyH, topologyW, topologyBlockH, topologyBlockW),
          blocksIterCnt_(GetBlockIterCnt(topology_.TotalCnt()))
    {
    }

    inline __aicore__ static uint32_t GetBlockIterCnt(uint32_t totalBlocks)
    {
        uint16_t blockNum = GetBlockNum();
        uint32_t mainIterCnt = totalBlocks / blockNum;
        uint32_t tailBlocks = totalBlocks - mainIterCnt * blockNum;

        if constexpr (OnlyIterMainBlocks) {
            //尾轮空闲核超过一半时，将这些block留到后续切k处理
            return tailBlocks > (blockNum / 2) ? mainIterCnt + 1 : mainIterCnt;
        } else {
            return mainIterCnt + (tailBlocks > 0 ? 1 : 0);
        }
    }

    const uint32_t cout_;
    const uint32_t cin_;
    const SwizzleTopology2D topology_;
    const uint32_t blocksIterCnt_;
    uint32_t loopIdx_ = 0;
};

struct SplitKState {
    uint32_t kBegin = 0;
    uint32_t kLength = 0;
    //单个kGroup最多会被分到几个k
    uint32_t kMaxLength = 0;
    //k轴被切分成几个组
    uint16_t kGroups = 0;
    //当前核所在的kGroup的索引
    uint16_t kGroupIdx = 0;
    //当前kGroup的核数
    uint16_t kGroupStartCoreId = 0;
    uint16_t kGroupCoreNum = 0;
};

template <BlockIterDirection MainBlockIterDir, typename TilingT>
class TailBlockSplitKIterator {
public:
    // 由于主轮的走位不是按照固定的矩形方式走的，TailBlocks在整个基本块里的形状不一定能用一个矩形表示，所以构造函数里需要
    // 传入主轮使用的SwizzleTopology2D解算实际坐标
    //
    inline __aicore__ TailBlockSplitKIterator(
        uint32_t tailBlockCnt, const SwizzleTopology2D& topology, uint32_t totalK,
        uint32_t cout, uint32_t cin)
        : topology_(topology),
          tailBlockCnt_(tailBlockCnt),
          topologyTailIter_((topology.TotalCnt() - tailBlockCnt) / GetBlockNum()),
          totalK_(totalK),
          cout_(cout),
          cin_(cin)
    {
    }

    //将尾块按k轴平分到核上，要求尾轮基本块不超过核数一半
    //所有核会被拆分成kGroup个组，每个组处理所有尾块的相同k段
    //这样可以保证kGroup内的逻辑应当和主轮相同，本质是处理TailBlocks个基本块，只是k轴缩短
    //分不进kGroup的尾核统一塞进最后一个kGroup，这样子外层不用单独适配尾轮的空跑代码，所以最后一个kGroup的核可能会多一些
    //
    //  kGroup = Std::min(totalCoreNum / tailBlockCnt_, totalK_)
    //  kCores = CoreNums/kGroup
    //
    //                                KGroup(2)
    //                -  +--------------+---------------+
    //                |  |core0         |kCore          |
    //   TailBlocks  -|  +--------------+---------------+
    //                |  |core1         |kCore1         |
    //                -  +--------------+---------------+
    //                   |core2(idle)   |kCore2(idle)   |
    //                   +--------------+---------------+
    //                   |..............|...............|
    //                   +--------------+---------------+
    //                   |kCore-1(idle) |kCore*2-1(idle)|
    //                   +--------------+---------------+
    //                                  |...............|
    //                                  +---------------+
    //                                  |LastCore(idle) |
    //                                  +---------------+
    //
    inline __aicore__ void GetLocalBlock(CoutCinRange& cRange, SplitKState& k) const
    {
        uint16_t totalCoreNum = GetBlockNum();

        //最多多少个core去切分同一个基本块的k
        k.kGroups = Std::min(totalCoreNum / tailBlockCnt_, totalK_);
        //每个kGroup里有几个core
        uint32_t kCores = totalCoreNum / k.kGroups;

        uint32_t coreId = AicCoreId();
        //不在任意一个kGroup里的尾核统统塞进最后一个kGroup里面，不能完全不管，得参与全核同步
        k.kGroupIdx = Std::min(coreId / kCores, k.kGroups - 1);
        k.kGroupCoreNum = kCores;
        k.kGroupStartCoreId = kCores * k.kGroupIdx;
        if (k.kGroupIdx == k.kGroups - 1) {
            k.kGroupCoreNum = totalCoreNum - k.kGroupStartCoreId;
        }

        RemainderDistributionSpliter splitter(totalK_, k.kGroups);
        splitter.GetSplit(k.kGroupIdx, k.kBegin, k.kLength);
        k.kMaxLength = splitter.GetMaxLength();

        uint16_t tailBlockIdx = coreId - k.kGroupStartCoreId;
        GetBlockFromSwizzle2D<MainBlockIterDir>(
            topology_, topologyTailIter_, tailBlockIdx,
            SingleShapeCout, SingleShapeCin,
            cout_, cin_,
            cRange);
    }

private:
    static constexpr uint32_t SingleShapeCout = BlockConfig::SingleShapeCout<TilingT>();
    static constexpr uint32_t SingleShapeCin = BlockConfig::SingleShapeCin<TilingT>();
    const SwizzleTopology2D topology_;
    const uint16_t tailBlockCnt_;
    const uint32_t topologyTailIter_;
    const uint32_t totalK_;
    const uint32_t cout_;
    const uint32_t cin_;
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
        constexpr uint32_t fwdTmpBufSize = GetFwdTmpBufSize() * sizeof(T);
        constexpr uint32_t fwdSrcBufSize = GetFwdSrcBufSize() * sizeof(T) * BUF_CNT;
        constexpr uint32_t fwdOutBufSize = GetFwdOutBufSize() * sizeof(T) * BUF_CNT;

        TBuf<TPosition::VECIN> transformFwdTmpBuf;
        TBuf<TPosition::VECIN> transformFwdSrcBuf;
        TBuf<TPosition::VECIN> transformFwdOutBuf;

        static_assert((fwdTmpBufSize + fwdSrcBufSize + fwdOutBufSize) < TOTAL_UB_SIZE, "illegal buffer size");

        TPipe* pipe = GetTPipePtr();
        pipe->InitBuffer(transformFwdTmpBuf, fwdTmpBufSize);
        pipe->InitBuffer(transformFwdSrcBuf, fwdSrcBufSize);
        pipe->InitBuffer(transformFwdOutBuf, fwdOutBufSize);
        transformFwdTmpVBuf_ = transformFwdTmpBuf.Get<T>();
        transformFwdSrcVBuf_ = transformFwdSrcBuf.Get<T>();
        transformFwdOutVBuf_ = transformFwdOutBuf.Get<T>();

        for (uint8_t i = 0; i < BUF_CNT; i++) {
            transformFwdEventFlags_[i] = TransformVFlag::AllocEventId(pipe);
            SetFlag<HardEvent::V_MTE2>(transformFwdEventFlags_[i].v2mte2);
            SetFlag<HardEvent::MTE3_V>(transformFwdEventFlags_[i].mte32v);
        }
    }

    __aicore__ inline void IterateK(
        const CoutCinRange& localBlock,
        BatchTileKIterator<TilingT>& kIter,
        FwdTransformGM2L1Queue<T>& gm2l1Que,
        FwdTransformUB2L1Queue<T>& ub2l1Que,
        uint32_t watermarkResidentC,
        uint32_t residentCBound,
        uint16_t residentKGroupStartCoreIdx,
        uint16_t residentKGroupCoreNum)
    {
        using BlockConfig::InputTensor;
        constexpr InputTensor ResidentTarget = BlockConfig::ResidentTarget<TilingT>();
        constexpr InputTensor TensorT0 = ResidentTarget != InputTensor::FMAP ? InputTensor::FMAP : InputTensor::DY;
        constexpr InputTensor TensorT1 = ResidentTarget == InputTensor::FMAP ? InputTensor::FMAP : InputTensor::DY;

        StreamTaskInfo streamT0;
        ComputeT0TaskInfo(localBlock.GetIdx<TensorT0>(), localBlock.GetLen<TensorT0>(), streamT0);

        StreamTaskInfo streamT1;
        ResidentTaskInfo residentT1;
        ComputeT1TaskInfo(
            localBlock.GetIdx<TensorT1>(), localBlock.GetLen<TensorT1>(),
            residentCBound, watermarkResidentC,
            BlockConfig::SingleShapeC<TilingT, ResidentTarget>(),
            streamT1, residentT1);

        uint32_t residentTaskOffset = 0;
        while (kIter.More()) {
            HWBox tile = kIter.TileBox();

            if (residentCBound > watermarkResidentC) {
                typename TransformFunctions::GM2L1Ctx gm2l1Ctx = {kIter.BatchIdx(), kIter.TileKIdx(), {gm2l1Que}};
                gm2l1Que.WaitSlot();

                ProcessResidentTransform<TensorT1>(
                    tile,
                    gm2l1Ctx,
                    residentT1, residentTaskOffset,
                    residentKGroupStartCoreIdx, residentKGroupCoreNum);

                gm2l1Que.EnQue();
            }

            typename TransformFunctions::UB2L1Ctx ub2l1Ctx = {kIter.BatchIdx(), kIter.TileKIdx(), {ub2l1Que, 0}};
            ub2l1Que.WaitSlot();

            ProcessStreamingTransform<TensorT1>(
                tile, ub2l1Ctx, streamT1);

            ProcessStreamingTransform<TensorT0>(
                tile, ub2l1Ctx, streamT0);

            ub2l1Que.EnQue();

            kIter.Next();
        }
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
        stream.cLocalIdx = localCIdx;
        stream.cIdx = localCIdx;
        stream.cLen = localCLen;
        stream.singleCoreCLen = Ops::Base::CeilDiv(
                                    Ops::Base::CeilDiv(stream.cLen, C0<T>()),
                                    AivNumInBlock()) * C0<T>();
    }

    __aicore__ inline void ComputeT1TaskInfo(
        uint32_t localCIdx,
        uint16_t localCLen,
        uint32_t residentCBound,
        uint32_t watermarkResidentC,
        uint16_t singleShapeC,
        StreamTaskInfo& stream,
        ResidentTaskInfo& resident) const
    {
        //[resident,stream]
        stream.cLocalIdx = localCIdx;
        stream.cIdx = localCIdx + SingleShapeResidentC;
        stream.cLen = Std::max(localCLen, SingleShapeResidentC) - SingleShapeResidentC;
        stream.singleCoreCLen = Ops::Base::CeilDiv(
                                    Ops::Base::CeilDiv(stream.cLen, C0<T>()),
                                    AivNumInBlock()) * C0<T>();

        if (residentCBound > watermarkResidentC) {
            uint32_t t1FullCLen = residentCBound - watermarkResidentC;
            uint32_t t1MainCBlk = t1FullCLen / singleShapeC;
            uint16_t t1TailCLen = t1FullCLen % singleShapeC;

            resident.cIdx = watermarkResidentC;
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
        const ResidentTaskInfo& task, uint32_t& taskOffset,
        uint16_t residentKGroupStartCoreIdx, uint16_t residentKGroupCoreNum)
    {
        using TransformConfig = Std::conditional_t<
            TransformType == BlockConfig::InputTensor::FMAP,
            WinoTransformDetail::FmapConfig<T, TilingT>,
            WinoTransformDetail::DyConfig<T, TilingT> >;

        const uint32_t coreId = AivCoreId() - residentKGroupStartCoreIdx * AivNumInBlock();
        const uint32_t stride = residentKGroupCoreNum * AivNumInBlock();

        for (uint32_t taskId = (coreId + stride - taskOffset) % stride;
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

        taskOffset = (taskOffset + task.cTaskCnt) % stride;
    }

    struct StreamTaskInfo {
        uint32_t cLocalIdx;
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
            ctx.l1method.ub2l1Offset = tile.elements * F23_TRANSFORM_TILE_ELEMENTS_16 * (cIdx + c - tasks.cLocalIdx);

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

    __aicore__ inline void IterateK(
        const CoutCinRange& blockRange,
        BatchTileKIterator<TilingT>& kIter,
        FwdTransformGM2L1Queue<T>& gm2l1,
        FwdTransformUB2L1Queue<T>& ub2l1,
        bool waitResidentTransform)
    {
        if (blockRange.NotEmpty()) {
            RunMmad<true>(
                blockRange, kIter,
                gm2l1, ub2l1,
                waitResidentTransform);
        } else {
            // 闲置核仅参与 Queue 信号同步，维持集群流水线运转，不进行实际 Compute
            RunMmad<false>(
                blockRange, kIter,
                gm2l1, ub2l1,
                waitResidentTransform);
        }
    }

    template <typename SplitMImpl>
    __aicore__ inline void Fixpipe2UB(
        InvTransformL0C2UBSyncQueue<TilingT>& syncQue,
        const CoutCinRange& localBlock,
        const LocalTensor<float>& invBuf,
        const SplitMScheduler::Interface<SplitMImpl>& splitMScheduler)
    {
        winoMmad_.Fixpipe2UB(
            syncQue,
            localBlock,
            invBuf,
            splitMScheduler);
    }

private:
    template <bool NotIdle>
    __aicore__ inline void RunMmad(
        const CoutCinRange& cRange,
        BatchTileKIterator<TilingT>& iter,
        FwdTransformGM2L1Queue<T>& gm2l1,
        FwdTransformUB2L1Queue<T>& ub2l1,
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
            uint32_t kIdx = iter.TileKIdx();
            uint32_t batchIdx = iter.BatchIdx();

            // ================= 阶段 1: Prologue (预载入第一轮数据) =================
            MmadLoadResident<NotIdle>(
                tiles, gm2l1, batchIdx, kIdx,
                residentC1Idx, residentC1Length,
                waitResidentTransform, loadPingPong);

            iter.Next();
            bool firstK = true;
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
                uint32_t nextKIdx = iter.TileKIdx();
                uint32_t nextBatchIdx = iter.BatchIdx();

                MmadLoadResident<NotIdle>(
                    nextTiles, gm2l1, nextBatchIdx, nextKIdx,
                    residentC1Idx, residentC1Length,
                    waitResidentTransform, loadPingPong);

                MmadCompute<NotIdle>(
                    tiles, ub2l1,
                    cRange.coutLength, coutC1Length,
                    cRange.cinLength, cinC1Length,
                    firstK,
                    computePingPong);

                firstK = false;
                tiles = nextTiles;
                kIdx = nextKIdx;
                batchIdx = nextBatchIdx;

                iter.Next();
            }

            // ================= 阶段 3: Epilogue (计算最后一轮数据) =================
            MmadCompute<NotIdle>(
                tiles, ub2l1,
                cRange.coutLength, coutC1Length,
                cRange.cinLength, cinC1Length,
                kIdx == 0 && batchIdx == 0,
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

    template <bool NotIdle>
    __aicore__ inline void MmadCompute(
        const HWBox& tiles,
        FwdTransformUB2L1Queue<T>& ub2l1,
        uint32_t cout,
        uint32_t coutC1,
        uint32_t cin,
        uint32_t cinC1,
        bool firstK,
        bool& l1PingPongFlag)
    {
        // 阻塞等待 AIV 的 DY 生产信号
        ub2l1.WaitData();

        if constexpr (NotIdle) {
            winoMmad_.Compute(
                tiles,
                cout,
                coutC1,
                cin,
                cinC1,
                firstK,
                l1PingPongFlag);
            //TODO pingpong和ub2l1更新同步
            l1PingPongFlag = !l1PingPongFlag;
        }

        ub2l1.DeQue();
    }

    WinoMMAD<T, TilingT>& winoMmad_;
};
}

template <typename SrcT, typename DstT, typename TilingT>
class ConvBackpropFilterWinograd {
public:
    static constexpr bool ResidentFmap =
        BlockConfig::ResidentTarget<TilingT>() == BlockConfig::InputTensor::FMAP;

    __aicore__ inline ConvBackpropFilterWinograd(
        const WinoFmapFwdTransformer<SrcT, TilingT>& fmap,
        const WinoDyFwdTransformer<SrcT, TilingT>& dy,
        __gm__ SrcT* nk1c1k0c0Gm,
        __gm__ DstT* yGm,
        WinoMMAD<SrcT, TilingT>& winoMmad,
        uint32_t tilesH,
        uint32_t tilesW,
        uint32_t batch)
        : tilesH_(tilesH),
          tilesW_(tilesW),
          batch_(batch),
          cin_(fmap.SrcC()),
          cout_(dy.SrcC()),
          gm2l1_(
              nk1c1k0c0Gm,
              NK1C1K0C0::Shape<SrcT>::template Create<TilingT>(
                  ResidentFmap ? cin_ : cout_, tilesH, tilesW)),
          dwFwd_(fmap, dy),
          dwMmad_(winoMmad),
          dwInv_(yGm)
    {
    }

    inline void __aicore__ Init()
    {
        if ASCEND_IS_AIV {
            dwFwd_.Init();
            dwInv_.Init();
        }
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

    inline void __aicore__ IterateAll()
    {
        using namespace WinoDetail;
        //驻留fmap就往cout方向循环,减少执行驻留带来的全局同步影响
        constexpr BlockIterDirection BasicBlockIterDir = ResidentFmap ? COUT : CIN;
        auto blockIter = BlockIterator<BasicBlockIterDir, TilingT, true>::Create(cout_, cin_);

        uint32_t watermarkResidentC = 0;

        while (blockIter.More()) {
            CoutCinRange localBlock;
            blockIter.GetLocalBlock(localBlock);

            uint32_t clusterCoutBound, clusterCinBound;
            blockIter.GetClusterBlockUpperBound(clusterCoutBound, clusterCinBound);
            uint32_t residentCBound = ResidentFmap ? clusterCinBound : clusterCoutBound;

            BatchTileKIterator<TilingT> kIter(batch_, tilesH_, tilesW_);
            //主轮不切K，cout整个轴在搬出时不做交织切分
            auto scheduleM = SplitMScheduler::Interface(SplitMScheduler::Single(localBlock.coutLength));
            IterateK(
                localBlock, kIter,
                residentCBound, watermarkResidentC,
                0, GetBlockNum(), false,
                scheduleM);

            blockIter.Next();
            watermarkResidentC = Std::max(watermarkResidentC, residentCBound);
        }

        if (blockIter.GetTailBlockCnt() == 0) {
            return;
        }

        constexpr uint32_t singleShapeTileH = BlockConfig::SingleShapeTileH<TilingT>();
        auto tailIter = TailBlockSplitKIterator<BasicBlockIterDir, TilingT>(
            blockIter.GetTailBlockCnt(),
            blockIter.GetSwizzleTopology(),
            batch_ * Ops::Base::CeilDiv(tilesH_, singleShapeTileH),
            cout_, cin_);

        CoutCinRange localBlock;
        SplitKState splitKState;
        tailIter.GetLocalBlock(localBlock, splitKState);
        uint32_t residentCBound = ResidentFmap ? cin_ : cout_;

        BatchTileKIterator<TilingT> kIter(batch_, tilesH_, tilesW_, splitKState.kBegin, splitKState.kLength);
        auto scheduleM = SplitMScheduler::Interface(
            SplitMScheduler::Interleave(
                splitKState.kGroups,
                splitKState.kGroupIdx,
                localBlock.coutLength));

        IterateK(
            localBlock, kIter,
            residentCBound, watermarkResidentC,
            splitKState.kGroupStartCoreId, splitKState.kGroupCoreNum,
            splitKState.kLength < splitKState.kMaxLength, //切k不均衡时要补一轮同步
            scheduleM);
    }

private:
    template <typename SplitMImpl>
    inline __aicore__ void IterateK(
        const CoutCinRange& localBlock,
        WinoDetail::BatchTileKIterator<TilingT>& kIter,
        uint32_t residentCBound,
        uint32_t watermarkResidentC,
        uint16_t kGroupStartCore, uint16_t kCore,
        bool appendResidentCrossCoreSync,
        const SplitMScheduler::Interface<SplitMImpl>& scheduleM)
    {
        bool shouldResidentTransform = residentCBound > watermarkResidentC;
        constexpr bool IsTailSplitK = Std::is_same_v<SplitMImpl, SplitMScheduler::Interleave>;

        if ASCEND_IS_AIC {
            dwMmad_.IterateK(
                localBlock,
                kIter,
                gm2l1_,
                ub2l1_,
                shouldResidentTransform);

            if constexpr (IsTailSplitK) {
                if (appendResidentCrossCoreSync && shouldResidentTransform) {
                    //尾轮处理时切k不均衡需要额外补一次全核同步
                    //要是芯片跨核同步支持分组不强制全核一起来就好了
                    gm2l1_.WaitData();
                    gm2l1_.DeQue();
                }
            }
        }

        if ASCEND_IS_AIV {
            dwFwd_.IterateK(
                localBlock,
                kIter,
                gm2l1_,
                ub2l1_,
                watermarkResidentC,
                residentCBound,
                kGroupStartCore,
                kCore);

            if constexpr (IsTailSplitK) {
                if (appendResidentCrossCoreSync && shouldResidentTransform) {
                    gm2l1_.WaitSlot();
                    gm2l1_.EnQue();
                }
            }
        }

        //当前ub很难同时放下正变换和逆变换的速率，所以逆变换需要停掉整个正变换，并空出整个ub来逆变换，
        constexpr uint32_t invBufSize = WinoInvBufUtil::GetInvBufTotalSizeInBytes<TilingT>();
        static_assert(invBufSize < TOTAL_UB_SIZE, "illegal buffer size");
        auto invBuf = LocalTensor<float>(TPosition::VECIN, 0, invBufSize);

        if (localBlock.NotEmpty()) {
            if ASCEND_IS_AIC {
                dwMmad_.Fixpipe2UB(
                    l0c2ubSync_, localBlock, invBuf, scheduleM);
            }
            if ASCEND_IS_AIV {
                dwInv_.TransformOutput(l0c2ubSync_, localBlock, cin_, invBuf, scheduleM);
                //主轮下停掉正变换的mte2搬运直到逆变换mte3搬出结束
                //尾轮不用，因为已经全算完了，不会有mte2触发了
                if constexpr (!IsTailSplitK) {
                    dwInv_.BlockMTE2ByMTE3();
                }
            }
        } else {
            //主轮情况空跑核直接退出不做搬出就行
            //尾轮切k下还得参与交织写入的全局同步
            if constexpr (IsTailSplitK) {
                if ASCEND_IS_AIV {
                    dwInv_.TransformOutputJoinInterleaveSyncOnly(l0c2ubSync_, localBlock, cin_, invBuf, scheduleM);
                }
            }
        }
    }


    const uint32_t tilesH_;
    const uint32_t tilesW_;
    const uint32_t batch_;
    const uint32_t cin_;
    const uint32_t cout_;

    WinoDetail::FwdTransformGM2L1Queue<SrcT> gm2l1_;
    WinoDetail::FwdTransformUB2L1Queue<SrcT> ub2l1_;
    WinoDetail::InvTransformL0C2UBSyncQueue<TilingT> l0c2ubSync_;
    WinoDetail::AivFwdTransformer<SrcT, TilingT> dwFwd_;
    WinoDetail::AicMmadComputer<SrcT, TilingT> dwMmad_;
    WinoInvTransformer<DstT, TilingT> dwInv_;
};


#endif //CONV_BP_WINO_H