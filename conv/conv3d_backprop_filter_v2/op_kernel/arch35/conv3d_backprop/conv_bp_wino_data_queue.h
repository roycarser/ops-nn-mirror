/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CONV_BP_WINO_DATA_QUEUE_H
#define CONV_BP_WINO_DATA_QUEUE_H

#include "conv_bp_wino_util.h"

//正变换后在gm上的排布[N,k1(TileH/SingleShapeTileH * TileW/SingleShapeTileW),C1,k0(16,SingleShapeTileHW),C0]
namespace NK1C1K0C0 {
template <typename T>
struct Shape {
    __aicore__ inline Shape(
        uint32_t c,
        uint32_t tileH,
        uint32_t tileW,
        uint32_t singleShapeTileH,
        uint32_t singleShapeTileW)
        : k1(Ops::Base::CeilDiv(tileH, singleShapeTileH) * Ops::Base::CeilDiv(tileW, singleShapeTileW)),
          c1(Ops::Base::CeilDiv(c, C0<T>())),
          k0(singleShapeTileH * singleShapeTileW * F23_TRANSFORM_TILE_ELEMENTS_16)
    {
    }

    __aicore__ inline uint64_t GetOffset(
        uint32_t nIdx,
        uint32_t k1Idx,
        uint32_t c1Idx) const
    {
        uint64_t k0c0 = static_cast<uint64_t>(k0) * c0;
        uint64_t c1k0c0 = static_cast<uint64_t>(c1) * k0c0;
        uint64_t k1c1k0c0 = static_cast<uint64_t>(k1) * c1k0c0;

        return nIdx * k1c1k0c0 + k1Idx * c1k0c0 + c1Idx * k0c0;
    }

    const uint32_t k1;
    const uint32_t c1;
    const uint32_t k0;
    static constexpr uint8_t c0 = C0<T>();
};

template <typename T>
struct CopyK0Params {
    uint32_t tiles = 0;
    uint32_t srcBufWidthBlockStride = 0;
    uint32_t batchIdx = 0;
    uint32_t k1Idx = 0;
    uint32_t c1Idx = 0;
    uint32_t c1Length = 0;
    AscendC::GlobalTensor<T> gm;
    AscendC::LocalTensor<T> ub;
    AscendC::LocalTensor<T> l1;
};


template <typename T>
__aicore__ inline void CopyK0UB2GM(CopyK0Params<T>& p, const Shape<T>& shape)
{
    ascendc_assert(shape.k0>= F23_TRANSFORM_TILE_ELEMENTS_16 * p.tiles, "can only move one k0 out");

    uint64_t gmOffset = shape.GetOffset(p.batchIdx, p.k1Idx, p.c1Idx);

    AscendC::DataCopyParams params;
    params.blockCount = F23_TRANSFORM_TILE_ELEMENTS_16;
    params.blockLen = p.tiles;
    params.srcGap = p.srcBufWidthBlockStride - p.tiles;
    params.dstGap = 0;

    constexpr uint8_t c0Byte = Shape<T>::c0 * sizeof(T);
    AscendC::LoopModeParams loop;
    loop.loop1Size = p.c1Length;
    loop.loop1SrcStride = F23_TRANSFORM_TILE_ELEMENTS_16 * p.srcBufWidthBlockStride * c0Byte;
    loop.loop1DstStride = shape.k0 * c0Byte;
    loop.loop2Size = 1;

    AscendC::SetLoopModePara(loop, AscendC::DataCopyMVType::UB_TO_OUT);
    AscendC::DataCopy(p.gm[gmOffset], p.ub, params);
    AscendC::ResetLoopModePara(AscendC::DataCopyMVType::UB_TO_OUT);
}

template <typename T>
__aicore__ inline void CopyK0GM2L1(CopyK0Params<T>& p, const Shape<T>& shape)
{
    ascendc_assert(shape.k0>= F23_TRANSFORM_TILE_ELEMENTS_16 * p.tiles, "can only move one k0 out");
    uint64_t gmOffset = shape.GetOffset(p.batchIdx, p.k1Idx, p.c1Idx);

    AscendC::DataCopyParams params;
    params.blockCount = p.c1Length;
    params.blockLen = p.tiles * F23_TRANSFORM_TILE_ELEMENTS_16;
    params.srcGap = shape.k0 - params.blockLen;
    params.dstGap = 0;

    AscendC::DataCopy(p.l1, p.gm[gmOffset], params);
}

template <typename T>
__aicore__ inline void CopyK0UB2L1(CopyK0Params<T>& p)
{
    for (uint32_t c1 = 0; c1 < p.c1Length; c1++) {
        AscendC::DataCopyParams params;
        params.blockCount = F23_TRANSFORM_TILE_ELEMENTS_16;
        params.blockLen = p.tiles;
        params.srcGap = p.srcBufWidthBlockStride - p.tiles;
        params.dstGap = 0;

        uint32_t ubOffset = p.srcBufWidthBlockStride * F23_TRANSFORM_TILE_ELEMENTS_16 * C0<T>() * c1;
        uint32_t l1Offset = p.tiles * F23_TRANSFORM_TILE_ELEMENTS_16 * C0<T>() * c1;
        AscendC::DataCopy(p.l1[l1Offset], p.ub[ubOffset], params);
    }
}
}


//CrossCoreSetFlag内计数器上限不能超过15
//这里设置连续EnQue12次就要等待DeQue通知，防止计数器超限
static constexpr uint8_t DEFAULT_FREE_SLOTS = 12;
//纯PingPong写入,只允许连续EnQue2次就要等DeQue通知
static constexpr uint8_t PINGPONG_FREE_SLOTS = 2;

template <pipe_t SRC_PIPE, pipe_t DST_PIPE, uint8_t PUSH_FLAG, uint8_t POP_FLAG, uint8_t FREE_SLOTS>
class CVSyncQue {
public:
    __aicore__ inline void WaitSlot()
    {
        if (freeSlots_ == 0) {
            AscendC::CrossCoreWaitFlag<2, SRC_PIPE>(POP_FLAG);
        }
    }

    __aicore__ inline void EnQue()
    {
        AscendC::CrossCoreSetFlag<2, SRC_PIPE>(PUSH_FLAG);
        if (freeSlots_ > 0) {
            freeSlots_--;
        }
    }

    __aicore__ inline void WaitData()
    {
        AscendC::CrossCoreWaitFlag<2, DST_PIPE>(PUSH_FLAG);
    }

    __aicore__ inline void DeQue()
    {
        AscendC::CrossCoreSetFlag<2, DST_PIPE>(POP_FLAG);
    }

    __aicore__ inline void PipeBarrierAllEnd()
    {
        //如果CrossCoreSetFlag是最后的指令可能因为一执行就核就退出导致没能成功set,整个核结束前加个全量等待
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    uint8_t freeSlots_ = FREE_SLOTS;
};

//ub正变换到l1的队列,分成ub->l1和ub->gm->l1这2类
//使用接口:
//  Init()初始化
//  1.ub写出
//    q.WaitSlot() 等待队列空间
//    q.Write() 写入数据
//    q.EnQue() 完成写入,执行CrossCore通知可读
// 2. l1读出
//    q.WaitData() 等待队列数据
//    LoadL1 读取数据到L1
//    q.DeQue() 释放队列空间
// End()释放资源
//

template <pipe_t SRC_PIPE, pipe_t DST_PIPE, uint8_t PUSH_FLAG, uint8_t POP_FLAG, uint8_t FREE_SLOTS>
class BaseL1Queue {
public:
    using BaseT = BaseL1Queue;

    __aicore__ inline void End()
    {
        cvSyncQue_.PipeBarrierAllEnd();
    }

    __aicore__ inline void WaitSlot()
    {
        if ASCEND_IS_AIV {
            cvSyncQue_.WaitSlot();
        }
    }

    __aicore__ inline void WaitData()
    {
        if ASCEND_IS_AIC {
            cvSyncQue_.WaitData();
        }
    }

    __aicore__ inline void EnQue()
    {
        if ASCEND_IS_AIV {
            //同步等待所有aiv的mte3完成
            this->cvSyncQue_.EnQue();
        }
    }

    __aicore__ inline void DeQue()
    {
        if ASCEND_IS_AIC {
            this->cvSyncQue_.DeQue();
        }
    }

protected:
    CVSyncQue<SRC_PIPE, DST_PIPE, PUSH_FLAG, POP_FLAG, FREE_SLOTS> cvSyncQue_;
};

template <typename T, uint8_t PUSH_FLAG, uint8_t POP_FLAG>
class UB2L1Queue : public BaseL1Queue<PIPE_MTE3, PIPE_MTE1, PUSH_FLAG, POP_FLAG, PINGPONG_FREE_SLOTS> {
public:
    __aicore__ inline void Init(AscendC::LocalTensor<T> (&l1Buf)[2])
    {
        l1_[0] = l1Buf[0];
        l1_[1] = l1Buf[1];
    }

    __aicore__ inline void Write(NK1C1K0C0::CopyK0Params<T>& p, uint32_t l1Offset)
    {
        if ASCEND_IS_AIV {
            p.l1 = this->l1_[writePingPongFlag_][l1Offset];
            NK1C1K0C0::CopyK0UB2L1(p);
        }
    }

    __aicore__ inline void EnQue()
    {
        if ASCEND_IS_AIV {
            UB2L1Queue::BaseT::EnQue();
            writePingPongFlag_ = !writePingPongFlag_;
        }
    }

private:
    bool writePingPongFlag_ = false;
    AscendC::LocalTensor<T> l1_[2];
};


template <typename T, uint8_t PUSH_FLAG, uint8_t POP_FLAG, uint8_t AIC_MTE2_SYNC_FLAG>
class GM2L1Queue : public BaseL1Queue<PIPE_MTE3, PIPE_MTE2, PUSH_FLAG, POP_FLAG, DEFAULT_FREE_SLOTS> {
public:
    __aicore__ inline void Write(NK1C1K0C0::CopyK0Params<T>& p, const NK1C1K0C0::Shape<T>& shape)
    {
        if ASCEND_IS_AIV {
            NK1C1K0C0::CopyK0UB2GM(p, shape);
        }
    }

    __aicore__ inline void WaitData()
    {
        GM2L1Queue::BaseT::WaitData();
        //所有cube核接收到aiv发送的通知后才表示这一轮数据都准备好了
        AscendC::CrossCoreSetFlag<0, PIPE_MTE2>(AIC_MTE2_SYNC_FLAG);
        AscendC::CrossCoreWaitFlag<0, PIPE_MTE2>(AIC_MTE2_SYNC_FLAG);
    }

};


#endif //CONV_BP_WINO_DATA_QUEUE_H