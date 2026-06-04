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

#include "conv_bp_wino_data_queue.h"

using namespace AscendC;

template <typename T, typename TilingT>
class WinoMMAD {
public:
    __aicore__ WinoMMAD(bool hf32Flag)
        : hf32Flag_(hf32Flag)
    {
    }

    __aicore__ inline void Init()
    {
        //TODO L1 要留一个(16-tile.elements%16)的空间
        // 让load2d取最后一个点的最后一个分形时凑满512字节

        TPipe* pipe = GetTPipePtr();
        if ASCEND_IS_AIC {
            mte2mte1Flag_[0] = EventFlag::template Alloc<HardEvent::MTE2_MTE1, HardEvent::MTE1_MTE2>(pipe);
            mte2mte1Flag_[1] = EventFlag::template Alloc<HardEvent::MTE2_MTE1, HardEvent::MTE1_MTE2>(pipe);
            mte1madFlag_[0] = EventFlag::template Alloc<HardEvent::MTE1_M, HardEvent::M_MTE1>(pipe);
            mte1madFlag_[1] = EventFlag::template Alloc<HardEvent::MTE1_M, HardEvent::M_MTE1>(pipe);

#pragma unroll
            for (uint8_t n = 0; n != L0C_POINT_FIXPIPE_GROUP; n++) {
                mad2fixpipeFlag_[n] = EventFlag::template Alloc<HardEvent::M_FIX, HardEvent::FIX_M>(pipe);
                SetFlag<HardEvent::FIX_M>(mad2fixpipeFlag_[n].dst2src);
            }

            SetFlag<HardEvent::MTE1_MTE2>(mte2mte1Flag_[0].dst2src);
            SetFlag<HardEvent::MTE1_MTE2>(mte2mte1Flag_[1].dst2src);
            SetFlag<HardEvent::M_MTE1>(mte1madFlag_[0].dst2src);
            SetFlag<HardEvent::M_MTE1>(mte1madFlag_[1].dst2src);
            SetHF32Mode(hf32Flag_);
        }
    }

    __aicore__ inline void End()
    {
        if ASCEND_IS_AIC {
            WaitFlag<HardEvent::MTE1_MTE2>(mte2mte1Flag_[0].dst2src);
            WaitFlag<HardEvent::MTE1_MTE2>(mte2mte1Flag_[1].dst2src);
            WaitFlag<HardEvent::M_MTE1>(mte1madFlag_[0].dst2src);
            WaitFlag<HardEvent::M_MTE1>(mte1madFlag_[1].dst2src);
#pragma unroll
            for (uint8_t n = 0; n != L0C_POINT_FIXPIPE_GROUP; n++) {
                WaitFlag<HardEvent::FIX_M>(mad2fixpipeFlag_[n].dst2src);
            }
            SetHF32Mode(false);
        }
    }

    template <BlockConfig::InputTensor LoadTarget>
    __aicore__ inline void LoadL1(
        const GlobalTensor<T>& gm,
        const NK1C1K0C0::Shape<T>& shape,
        NK1C1K0C0::CopyK0Params& copyParams,
        bool l1PingPongFlag)
    {
        const EventFlag& mte2mte1 = mte2mte1Flag_[l1PingPongFlag];
        WaitFlag<HardEvent::MTE1_MTE2>(mte2mte1.dst2src);

        //TODO L1 要留一个(16-tile.elements%16)的空间
        // 让load2d取最后一个点的最后一个分形时凑满512字节
        auto l1Buf = GetL1Buf(l1PingPongFlag);

        if constexpr (LoadTarget == BlockConfig::InputTensor::FMAP) {
            LocalTensor<T>& l1b = Std::get<1>(l1Buf);
            NK1C1K0C0::CopyK0GM2L1(copyParams, gm, l1b, shape);
        } else {
            LocalTensor<T>& l1a = Std::get<0>(l1Buf);
            NK1C1K0C0::CopyK0GM2L1(copyParams, gm, l1a, shape);
        }

        SetFlag<HardEvent::MTE2_MTE1>(mte2mte1.src2dst);
    }

    template <bool FixpipeInLastK>
    __aicore__ inline void Compute(
        const HWBox& tiles, uint32_t cout, uint32_t coutC1, uint32_t cin, uint32_t cinC1,
        bool firstK, bool l1PingPongFlag,
        const LocalTensor<float>& outputTransformVBuf)
    {
        auto l1Buf = GetL1Buf(l1PingPongFlag);
        LocalTensor<T>& l1a = Std::get<0>(l1Buf);
        LocalTensor<T>& l1b = Std::get<1>(l1Buf);

        MmadParams mad;
        mad.m = cout;
        mad.n = cin;
        mad.k = tiles.elements;
        mad.cmatrixInitVal = firstK;

        LoadData2DParamsV2 load2d;
        load2d.mStartPosition = 0;
        load2d.kStartPosition = 0;
        load2d.ifTranspose = true;
        load2d.srcStride = static_cast<int32_t>(tiles.elements);

        // 数据按照 [C1,16,tile.elements,C0]排布,使用load2d一次搬运一个point,即[C1,tile.elements,C0]的数据进L0
        //
        //                          C0             C0
        //                         ----      /----
        // tile.elements(point0)    ..      /  ..
        //                         ----    /  ----
        //     ...                 ----   /   ----
        //                         ----  /    ----
        // tile.elements(point15)   ..  /      ..
        //                         ----/      ----

        uint32_t l0aMStep;
        uint32_t l0aKStep;
        uint32_t l0bMStep;
        uint32_t l0bKStep;

        if constexpr (sizeof(T) == 2) {
            l0aMStep = Ops::Base::CeilDiv(tiles.elements, static_cast<uint32_t>(BLOCK_CUBE));
            l0aKStep = coutC1;
            l0bMStep = l0aMStep;
            l0bKStep = cinC1;
        } else {
            l0aMStep = Ops::Base::CeilDiv(tiles.elements, static_cast<uint32_t>(BLOCK_CUBE));
            l0aKStep = Ops::Base::CeilAlign(coutC1, 2u);
            l0bMStep = l0aMStep;
            l0bKStep = Ops::Base::CeilAlign(cinC1, 2u);
        }

        uint32_t l0aPointElements = l0aMStep * l0aKStep * (AscendC::BYTE_PER_FRACTAL / sizeof(T));
        uint32_t l0bPointElements = l0bMStep * l0bKStep * (AscendC::BYTE_PER_FRACTAL / sizeof(T));

        //不需要baseK循环,L1上左右Tensor在PingPong后最多一共占用256kb
        //除以16后单个点最多16kb,L0上一定能全载,除非singleShapeHW传进来为1
        //然后l0上对齐放大到16这类异常情况,但tiling阶段应该防止这种情况

        const EventFlag& mte2mte1Flag = mte2mte1Flag_[l1PingPongFlag];
        WaitFlag<HardEvent::MTE2_MTE1>(mte2mte1Flag.src2dst);
        if (firstK) {
#pragma unroll
            for (uint8_t n = 0; n != L0C_POINT_FIXPIPE_GROUP; n++) {
                WaitFlag<HardEvent::FIX_M>(mad2fixpipeFlag_[n].dst2src);
            }
        }

        for (uint8_t g = 0; g < L0POINTS.group_; g++) {
            //通过奇偶性判断l0PingPong
            const int l0pingFlag = g & 1;

            const EventFlag& mte1madFlag = mte1madFlag_[l0pingFlag];
            WaitFlag<HardEvent::M_MTE1>(mte1madFlag.dst2src);

            uint8_t pointGroupOffset = g * L0POINTS.pointPerGroup_;

            LocalTensor<T> l0a = LocalTensor<T>(TPosition::A2, L0POINTS.l0aSize_ * l0pingFlag, L0POINTS.l0aSize_);
            LocalTensor<T> l0b = LocalTensor<T>(TPosition::B2, L0POINTS.l0bSize_ * l0pingFlag, L0POINTS.l0aSize_);

            for (uint8_t i = 0; i < L0POINTS.pointPerGroup_; i++) {
                uint8_t pointIdx = pointGroupOffset + i;
                uint32_t offsetL1 = pointIdx * tiles.elements * C0<T>();

                load2d.mStep = l0aMStep;
                load2d.kStep = l0aKStep;
                if constexpr (sizeof(T) == 4) {
                    load2d.dstStride = static_cast<int32_t>(l0aKStep) / 2;
                } else {
                    load2d.dstStride = static_cast<int32_t>(l0aKStep);
                }

                LoadData(l0a[i * l0aPointElements], l1a[offsetL1], load2d);

                load2d.mStep = l0bMStep;
                load2d.kStep = l0bKStep;
                if constexpr (sizeof(T) == 4) {
                    load2d.dstStride = static_cast<int32_t>(l0bKStep) / 2;
                } else {
                    load2d.dstStride = static_cast<int32_t>(l0bKStep);
                }

                LoadData(l0b[i * l0bPointElements], l1b[offsetL1], load2d);
            }

            SetFlag<HardEvent::MTE1_M>(mte1madFlag.src2dst);
            WaitFlag<HardEvent::MTE1_M>(mte1madFlag.src2dst);

            for (uint8_t i = 0; i < L0POINTS.pointPerGroup_; i++) {
                uint8_t pointIdx = pointGroupOffset + i;

                LocalTensor<float> l0cBuf = GetL0CPointBuf(pointIdx);
                uint32_t offsetA = i * l0aPointElements;
                uint32_t offsetB = i * l0bPointElements;

                AscendC::Mmad(l0cBuf, l0a[offsetA], l0b[offsetB], mad);

                if constexpr (FixpipeInLastK) {
                    if ((pointIdx + 1) % L0C_POINT_PER_FIXPIPE_GROUP == 0) {
                        uint8_t fixpipeGroupIdx = pointIdx / L0C_POINT_PER_FIXPIPE_GROUP;
                        const EventFlag& m2f = mad2fixpipeFlag_[fixpipeGroupIdx];
                        SetFlag<HardEvent::M_FIX>(m2f.src2dst);
                        WaitFlag<HardEvent::M_FIX>(m2f.src2dst);
                        fixpipe2UB(fixpipeGroupIdx, cout, cin, outputTransformVBuf);
                        SetFlag<HardEvent::FIX_M>(m2f.dst2src);
                    }
                }
            }

            SetFlag<HardEvent::M_MTE1>(mte1madFlag.dst2src);
        }

        SetFlag<HardEvent::MTE1_MTE2>(mte2mte1Flag.dst2src);
    }

    static __aicore__ inline Std::tuple<LocalTensor<T>, LocalTensor<T> > GetL1Buf(bool flagPingPong)
    {
        //PingPong按L1/2为界
        //整个L1被均分为2个bank,PingPong以L1SIZE/2为界分配到2个bank上
        //确保PingBuf上执行mte1/mte2时不会和PongBuf上的mte1/mte2产生bank冲突
        const LocalTensor<T> l1Buf = LocalTensor<T>(TPosition::A1, 0, TOTAL_L1_SIZE);

        constexpr uint32_t offsetPingPong = TOTAL_L1_SIZE / 2 / sizeof(T);
        uint32_t initOffset = offsetPingPong * flagPingPong;
        LocalTensor<T> l1a = l1Buf[initOffset];
        l1a.SetSize(L1A_LENGTH);
        LocalTensor<T> l1b = l1Buf[initOffset + L1A_LENGTH];
        l1b.SetSize(L1B_LENGTH);
        return Std::make_tuple(l1a, l1b);
    }

private:
    __aicore__ static inline void fixpipe2UB(
        uint8_t fixpipeGroupIdx,
        uint32_t cout,
        uint32_t cin,
        const LocalTensor<float>& outputTransformVBuf)
    {
        AscendC::FixpipeParamsC310 fp;
        //mSize对齐到2用于ub均分,由于实际计算分形一定是16的倍数，所以这么操作应当不会导致地址溢出
        //假设cout为16，那么对齐后还是16，如果是17那就会变成18，实际计算分形则是32，不存在溢出
        //TODO 判断尾块非C0对齐有没有问题
        fp.mSize = GetSubBlockNum() == 2 ? cout + (cout & 1) : cout;
        fp.nSize = cin;
        fp.srcStride = Ops::Base::CeilAlign(cout, static_cast<uint32_t>(BLOCK_CUBE));
        fp.dstStride = cin;
        fp.params.ndNum = L0C_POINT_PER_FIXPIPE_GROUP;
        fp.params.srcNdStride = L0C_SINGLE_POINT_BUF_BYTES / (BLOCK_CUBE * sizeof(float));
        //到UB上按C0对齐
        fp.params.dstNdStride = AivPartitioner::Get2DAlignBufLength<float>(cout, cin);
        fp.dualDstCtl = 1;

        LocalTensor<float> l0c = GetL0CPointBuf(fixpipeGroupIdx * L0C_POINT_PER_FIXPIPE_GROUP);
        static constexpr FixpipeConfig cfg = {CO2Layout::ROW_MAJOR, true};
        // Fixpipe<float, float, cfg>(outputTransformVBuf, l0c, fp);
    }

    struct L0Point {
        uint8_t group_;
        uint8_t pointPerGroup_;
        uint32_t l0aSize_;
        uint32_t l0bSize_;
    };

    static __aicore__ inline LocalTensor<float> GetL0CPointBuf(uint8_t pointIdx)
    {
        //l0c均分16片给每个点使用
        auto buf = LocalTensor<float>(
            TPosition::CO1,
            L0C_SINGLE_POINT_BUF_BYTES * pointIdx,
            L0C_SINGLE_POINT_BUF_BYTES);
        return buf;
    }

    constexpr static __aicore__ inline L0Point CalcWinoPointL0Group()
    {
        constexpr uint32_t l0KSize = ConstexprMaths::AlignUp(
                                         BlockConfig::SingleShapeTileHW<TilingT>(),
                                         BLOCK_CUBE) * sizeof(T);

        constexpr uint32_t singlePointL0ASize = ConstexprMaths::AlignUp(
                                                    BlockConfig::SingleShapeCout<TilingT>(),
                                                    BLOCK_CUBE) * l0KSize;
        constexpr uint32_t singlePointL0BSize = ConstexprMaths::AlignUp(
                                                    BlockConfig::SingleShapeCin<TilingT>(),
                                                    BLOCK_CUBE) * l0KSize;

        constexpr uint32_t l0BufLimit = TOTAL_L0A_SIZE / 2;
        constexpr uint32_t maxPointsL0A = l0BufLimit / singlePointL0ASize;
        constexpr uint32_t maxPointsL0B = l0BufLimit / singlePointL0BSize;
        constexpr uint32_t maxPointsL0 = ConstexprMaths::Min(maxPointsL0A, maxPointsL0B);

        //计算最多几个点一起批跑,从1,2,4,8这几个数里挑选,确保整除不会有尾轮处理
        //因为开了PingPong所以最多8个点一批,16个点一批PingPong就没意义了
        constexpr uint32_t pointsPerGroup = maxPointsL0 >= 8 ?
                                                8 :
                                                maxPointsL0 >= 4 ?
                                                4 :
                                                maxPointsL0 >= 2 ?
                                                2 :
                                                1;

        constexpr uint8_t outGroup = F23_TRANSFORM_TILE_ELEMENTS_16 / pointsPerGroup;
        constexpr uint8_t outPointPerGroup = pointsPerGroup;
        constexpr uint32_t outL0aSize = outPointPerGroup * singlePointL0ASize;
        constexpr uint32_t outL0bSize = outPointPerGroup * singlePointL0BSize;
        return {
            outGroup,
            outPointPerGroup,
            outL0aSize,
            outL0bSize
        };
    }



    struct EventFlag {
        TEventID src2dst = 0;
        TEventID dst2src = 0;

        template <HardEvent Src2DstEvent, HardEvent Dst2SrcEvent>
        static __aicore__ inline EventFlag Alloc(TPipe* pipe)
        {
            const TEventID src2dst = pipe->AllocEventID<Src2DstEvent>();
            const TEventID dst2src = pipe->AllocEventID<Dst2SrcEvent>();
            return {src2dst, dst2src};
        }
    };


    static constexpr uint32_t L1A_LENGTH = F23_TRANSFORM_TILE_ELEMENTS_16 *
                                           BlockConfig::SingleShapeTileHW<TilingT>() *
                                           BlockConfig::SingleShapeCout<TilingT>();
    static constexpr uint32_t L1B_LENGTH = F23_TRANSFORM_TILE_ELEMENTS_16 *
                                           BlockConfig::SingleShapeTileHW<TilingT>() *
                                           BlockConfig::SingleShapeCin<TilingT>();

    static constexpr L0Point L0POINTS = CalcWinoPointL0Group();

    static constexpr uint8_t L0C_POINT_FIXPIPE_GROUP = 4;
    static constexpr uint8_t L0C_POINT_PER_FIXPIPE_GROUP = F23_TRANSFORM_TILE_ELEMENTS_16 / L0C_POINT_FIXPIPE_GROUP;
    static constexpr uint32_t L0C_SINGLE_POINT_BUF_BYTES = TOTAL_L0C_SIZE / F23_TRANSFORM_TILE_ELEMENTS_16;

    EventFlag mad2fixpipeFlag_[L0C_POINT_FIXPIPE_GROUP];
    EventFlag mte2mte1Flag_[2];
    EventFlag mte1madFlag_[2];
    const bool hf32Flag_;
};

#endif //CONV_BP_WINO_MMAD_H