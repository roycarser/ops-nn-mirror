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
 * \file conv_bp_winograd.h
 * \brief
 */


#ifndef CONV_BP_WINOGRAD_H
#define CONV_BP_WINOGRAD_H

#include "basic_api/kernel_basic_intf.h"
#include "utils/std/algorithm.h"
#include "op_common/op_kernel/math_util.h"

using namespace AscendC;

//winograd在f23下单个tile长宽为4,元素数为16,fmap上stride为2
static constexpr uint8_t F23_WINO_TILE_SIZE_4 = 4;
static constexpr uint8_t F23_WINO_TILE_ELEMENTS_16 = 16;
static constexpr uint8_t F23_WINO_TILE_FMAP_STRIDE_2 = 2;

template <typename T>
static constexpr __aicore__ inline uint8_t C0()
{
    return DEFAULT_C0_SIZE / sizeof(T);
}

template <typename T>
static constexpr __aicore__ inline uint8_t VL()
{
    return VECTOR_REG_WIDTH / sizeof(T);
}

//一个位宽能包含几个C0
static constexpr uint8_t BLK_COUNT_IN_VL = VECTOR_REG_WIDTH / DEFAULT_C0_SIZE;


template <typename DType>
class WinoFmapTransformer {
public:
    __aicore__ inline WinoFmapTransformer(
        TPipe* pipe,
        GM_ADDR gm,
        const uint32_t cin,
        const uint32_t hi,
        const uint32_t wi,
        const uint16_t padH,
        const uint16_t padW,
        const uint16_t singleShapeCin,
        const uint16_t singleShapeK)
        : cin_(cin),
          cin1cin0_(Ops::Base::CeilAlign(cin_, C0<DType>())),
          hi_(hi),
          wi_(wi),
          padH_(padH),
          padW_(padW),
          tilesH_(CalculateF23Tiles(hi_ + 2 * padH_)),
          tilesW_(CalculateF23Tiles(wi_ + 2 * padW_)),
          singleShapeCin_(singleShapeCin),
          singleShapeK_(singleShapeK),
          //单次变换需要的buf为c1c0*tiles*16
          singleShapeBufLength_(
              Ops::Base::CeilAlign(singleShapeCin_, C0<DType>())
              * F23_WINO_TILE_ELEMENTS_16
              * singleShapeK_)
    {
        fmapGm_.SetGlobalBuffer(gm);
        uint32_t doubleBufferLength = singleShapeBufLength_ * 2 * sizeof(DType);
        //gm->vec->l1这条路这TQue里配置的同步事件不符合实际场景,所以自己用TBuf管理同步
        pipe->InitBuffer(vBuf_, doubleBufferLength);
    }

    struct TileKIter {
        uint32_t kIdx = 0;
        bool end = false;
    };

    __aicore__ inline void Transform(
        uint32_t batchIdx,
        uint32_t cIdx,
        TileKIter& iter)
    {
        //上层控制
        ascendc_assert(iter.end==false);

        uint32_t totalTiles = tilesH_ * tilesW_;
        uint16_t cLength = Std::min(singleShapeCin_, cin_ - cIdx);
        uint16_t tileKLength = Std::min(singleShapeK_, totalTiles - iter.kIdx);

        if ASCEND_IS_AIV {
            LocalTensor<DType> vBuf = GetPingPongBuffer();
            const TileBox box = CalculateF23TileBox(iter.kIdx, tileKLength);
            CopyInC1HTileWC0(vBuf, batchIdx, cIdx, cLength, box);
            //TODO setFlag/waitFlag
            // PipeBarrier<PIPE_ALL>();
            Compute(vBuf, cLength, box);
            // PipeBarrier<PIPE_ALL>();
        }

        iter.kIdx += tileKLength;
        iter.end = iter.kIdx >= totalTiles;
    }

private:
    struct HWBox {
        uint32_t hIdx;
        uint32_t wIdx;
        uint32_t hLength;
        uint32_t wLength;

        static __aicore__ inline uint32_t elements(const HWBox& box)
        {
            return box.wLength * box.hLength;
        }
    };

    struct HWPad {
        uint16_t hTop;
        uint16_t hBottom;
        uint16_t wLeft;
        uint16_t wRight;
    };

    struct TileBox {
        HWBox bodyTile;
        HWBox bodyFmap;
        HWPad bodyPad;

        HWBox tailTile;
        HWBox tailFmap;
        HWPad tailPad;

        HWBox headTile;
        HWBox headFmap;
        HWPad headPad;
    };

    //根据K将涉及的连续tile切分成body,head,tail3个矩形逐个处理
    //
    //   tileH=4,tileW=3,tileKIdx=2,tileKLength=8
    //        w0     w1     w2
    //     +------+------+------+
    //  h0 |      |      | head |
    //     +------+------+------+
    //  h1 | body | body | body |
    //     +------+------+------+
    //  h2 | body | body | body |
    //     +------+------+------+
    //  h3 | tail |      |      |
    //     +------+------+------+
    //
    __aicore__ inline TileBox CalculateF23TileBox(uint32_t tileKIdx, uint16_t tileKLength)
    {
        uint32_t tileKEndIdx = tileKIdx + tileKLength - 1;
        uint32_t startTileH = tileKIdx / tilesW_;
        uint32_t startTileW = tileKIdx - startTileH * tilesW_; // startTileH % tilesW_;
        uint32_t endTileH = tileKEndIdx / tilesW_;
        uint32_t endTileW = tileKEndIdx - endTileH * tilesW_; // tileKEndIdx % tilesW_;

        bool hasHead = startTileW > 0 && startTileH != endTileH;

        TileBox box;

        //
        //考虑让body始终存在,应该能减少一些处理body时的if/jump开销,
        //所以在移除head只有1行的情况下,[h1w0,h1w1]这种要判定为body而非tail
        //因此tail必须在bodyTileHIdx小于endTileH才存在,
        //        w0     w1    w2
        //     +------+------+------+
        //  h0 |      |      | head |
        //     +------+------+------+
        //  h1 | body | body |      |
        //     +------+------+------+
        //
        box.bodyTile.hIdx = hasHead ? startTileH + 1 : startTileH;
        box.bodyTile.wIdx = hasHead ? 0 : startTileW;

        bool hasTail = endTileW < (tilesW_ - 1) && box.bodyTile.hIdx != endTileH;
        uint32_t bodyTileHEndIdx = hasTail ? endTileH - 1 : endTileH;
        uint32_t bodyTileWEndIdx = hasTail ? tilesW_ - 1 : endTileW;

        box.bodyTile.hLength = bodyTileHEndIdx - box.bodyTile.hIdx + 1;
        box.bodyTile.wLength = bodyTileWEndIdx - box.bodyTile.wIdx + 1;
        F23TileBox2FmapBox(box.bodyTile, box.bodyFmap, box.bodyPad);

        box.headTile.hIdx = startTileH;
        box.headTile.wIdx = startTileW;
        box.headTile.wLength = hasHead ? tilesW_ - startTileW : 0;
        box.headTile.hLength = 1;
        F23TileBox2FmapBox(box.headTile, box.headFmap, box.headPad);

        box.tailTile.hIdx = endTileH;
        box.tailTile.wIdx = 0;
        box.tailTile.wLength = hasTail ? endTileW : 0;
        box.tailTile.hLength = 1;
        F23TileBox2FmapBox(box.tailTile, box.tailFmap, box.tailPad);

        return box;
    }

    //
    // 在[tileH,tileW]的区域中搬入fmap
    // 假设fmap大小为[6,6],那最终会生成2x2个tile,tile所占空间为[8,8]
    // 当前就在该[8,8]空间的左上角按照[fmapH,fmapW]搬入[6,6]的数据
    // 可以看到w方向并非连续,而是存在tileW-fmapW=8-6=2个空隙
    // 所以连续内存里实际搬入格式为[fmapH,tileW],一个tileW里只有fmapW个有效数据
    //
    //      w0 w1 w2 w3 w4 w5 w6 w7
    //     +--+--+--+--+--+--+--+--+
    //  h0 |xx|xx|xx|xx|xx|xx|  |  |
    //     +--+--+--+--+--+--+--+--+
    //  h1 |xx|xx|xx|xx|xx|xx|  |  |
    //     +--+--+--+--+--+--+--+--+
    //  h2 |xx|xx|xx|xx|xx|xx|  |  |
    //     +--+--+--+--+--+--+--+--+
    //  h3 |xx|xx|xx|xx|xx|xx|  |  |
    //     +--+--+--+--+--+--+--+--+
    //  h4 |xx|xx|xx|xx|xx|xx|  |  |
    //     +--+--+--+--+--+--+--+--+
    //  h5 |xx|xx|xx|xx|xx|xx|  |  |
    //     +--+--+--+--+--+--+--+--+
    //  h6 |  |  |  |  |  |  |  |  |
    //     +--+--+--+--+--+--+--+--+
    //  h7 |  |  |  |  |  |  |  |  |
    //     +--+--+--+--+--+--+--+--+
    //
    // 若存在pad,则只搬入fmap数据,但是位置按照fmap调整
    // 如下为padTop为1,padLeft为2的情况,当前tile所需要
    // 的[6,6]fmap区域中非padding区域大小只有[5,4],仅在
    // 这[5,4]对应的区域搬入
    //
    //      w0 w1 w2 w3 w4 w5 w6 w7
    //     +--+--+--+--+--+--+--+--+
    //  h0 |p |p |p |p |p |p |  |  |
    //     +--+--+--+--+--+--+--+--+
    //  h1 |p |p |xx|xx|xx|xx|  |  |
    //     +--+--+--+--+--+--+--+--+
    //  h2 |p |p |xx|xx|xx|xx|  |  |
    //     +--+--+--+--+--+--+--+--+
    //  h3 |p |p |xx|xx|xx|xx|  |  |
    //     +--+--+--+--+--+--+--+--+
    //  h4 |p |p |xx|xx|xx|xx|  |  |
    //     +--+--+--+--+--+--+--+--+
    //  h5 |p |p |xx|xx|xx|xx|  |  |
    //     +--+--+--+--+--+--+--+--+
    //  h6 |  |  |  |  |  |  |  |  |
    //     +--+--+--+--+--+--+--+--+
    //  h7 |  |  |  |  |  |  |  |  |
    //     +--+--+--+--+--+--+--+--+
    //
    __aicore__ inline void CopyInC1HTileWC0(
        LocalTensor<DType>& vBuf,
        uint32_t batchIdx,
        uint32_t cIdx,
        uint16_t cLength,
        const TileBox& box)
    {
        uint32_t fmapBatchCinOffset = (batchIdx * cin_ + cIdx) * hi_ * wi_;
        constexpr uint32_t tileElements = F23_WINO_TILE_ELEMENTS_16 * cin1cin0_;

        //按照head/body/tail的顺序分配ub
        uint32_t vBufBodyOffset = tileElements * HWBox::elements(box.headTile);
        uint32_t vBufTailOffset = tileElements * HWBox::elements(box.bodyTile) + vBufBodyOffset;

        CopyInTileBlockC1HTileWC0(
            vBuf[vBufBodyOffset],
            fmapBatchCinOffset, cLength,
            box.bodyTile, box.bodyFmap, box.bodyPad);

        if (HWBox::elements(box.headFmap) != 0) {
            CopyInTileBlockC1HTileWC0(
                vBuf,
                fmapBatchCinOffset, cLength,
                box.headTile, box.headFmap, box.headPad);
        }

        if (HWBox::elements(box.tailFmap) != 0) {
            CopyInTileBlockC1HTileWC0(
                vBuf[vBufTailOffset],
                fmapBatchCinOffset, cLength,
                box.tailTile, box.tailFmap, box.tailPad);
        }
    }

    __aicore__ inline void CopyInTileBlockC1HTileWC0(
        LocalTensor<DType>& vBuf,
        uint32_t fmapBatchCinOffset,
        uint16_t cLength,
        const HWBox& tile,
        const HWBox& fmap,
        const HWPad& pad)
    {
        //
        //          startFmapH=0,fmapHLength=2
        //          startFmapW=1,fmapWLength=3
        //
        //          n:  h0w0   h0w1   h0w2   h0w3   h1w0   h1w1   h1w2   h1w3
        //            +------+------+------+------+------+------+------+------+
        // d: cLength |      | xxxx DnBlock1 xxxx |      | xxxx DnBlock2 xxxx |
        //            +------+------+------+------+------+------+------+------+
        //

        uint32_t tileWLength = tile.wLength * F23_WINO_TILE_SIZE_4;
        Dn2NzParams dn2nz;
        dn2nz.dnNum = fmap.hLength;
        dn2nz.nValue = fmap.wLength;
        dn2nz.dValue = cLength;
        dn2nz.srcDnMatrixStride = wi_;
        dn2nz.srcDValue = hi_ * wi_;
        dn2nz.dstNzC0Stride = tileWLength * tile.hLength;
        dn2nz.dstNzNStride = 1;
        dn2nz.dstNzMatrixStride = tileWLength * C0<DType>();

        DataCopy(
            vBuf[(pad.hTop * tileWLength + pad.wLeft) * C0<DType>()],
            fmapGm_[fmapBatchCinOffset + fmap.hIdx * wi_ + fmap.wIdx], dn2nz);
    }


    __aicore__ inline void Compute(LocalTensor<DType>& vBuf, uint16_t cLength, const TileBox& box)
    {
        uint32_t headBufLen = HWBox::elements(box.headTile) * F23_WINO_TILE_ELEMENTS_16 * cin1cin0_;
        uint32_t bodyBufLen = HWBox::elements(box.bodyTile) * F23_WINO_TILE_ELEMENTS_16 * cin1cin0_;
        uint32_t tailBufLen = HWBox::elements(box.tailTile) * F23_WINO_TILE_ELEMENTS_16 * cin1cin0_;

        TransformToC1Th4Tw4C0(vBuf[headBufLen], cLength, box.bodyTile, box.bodyFmap, box.bodyPad);

        if (headBufLen != 0) {
            TransformToC1Th4Tw4C0(vBuf, cLength, box.headTile, box.headFmap, box.headPad);
        }

        if (tailBufLen != 0) {
            TransformToC1Th4Tw4C0(vBuf[headBufLen + bodyBufLen], cLength, box.tailTile, box.tailFmap, box.tailPad);
        }
    }


    __aicore__ inline void TransformToC1Th4Tw4C0(
       LocalTensor<DType>& vBuf,
        uint16_t cLength,
        const HWBox& tile,
        const HWBox& fmap,
        const HWPad& pad)
    {
        if (HWBox::elements(fmap) != 0) {
            UnfoldParams params;
            params.cLength = cLength;
            params.fmapH = fmap.hLength;
            params.fmapW = fmap.wLength;
            params.tileH = tile.hLength;
            params.tileW = tile.wLength;
            params.hElementsInUnfoldRows = tile.hLength * F23_WINO_TILE_SIZE_4;
            params.wElementsInUnfoldCols = fmap.wLength * C0<DType>();
            params.hRepeatTimesInUnfoldRows = Ops::Base::CeilDiv(
                params.hElementsInUnfoldRows, static_cast<uint32_t>(BLK_COUNT_IN_VL));
            params.wRepeatTimesInUnfoldCols = Ops::Base::CeilDiv(
                params.wElementsInUnfoldCols, static_cast<uint32_t>(VL<DType>()));

            params.pad = pad;
            params.wElementsInPadHBlock = (fmap.wLength + pad.wLeft + pad.wRight) * C0<DType>();
            params.hElementsInPadWBlock = fmap.hLength;
            params.wRepeatTimesInPadHBlock = Ops::Base::CeilDiv(params.wElementsInPadHBlock, VL<DType>());
            params.hRepeatTimesInPadWBlock = Ops::Base::CeilDiv(
                params.hElementsInPadWBlock, static_cast<uint32_t>(BLK_COUNT_IN_VL));

            VF_CALL<UnfoldFmapVf>(vBuf.GetPhyAddr(), params);

        } else {
            //整个tile都由padding区域产生,不做计算直接置0,
            Duplicate(vBuf, 0, HWBox::elements(tile) * F23_WINO_TILE_ELEMENTS_16 * cLength * C0<DType>());
        }
    }

    __aicore__ inline LocalTensor<DType> GetPingPongBuffer()
    {
        uint32_t offset = static_cast<uint32_t>(pingFlag_) * singleShapeBufLength_;
        pingFlag_ = !pingFlag_;
        return vBuf_.GetWithOffset<DType>(singleShapeBufLength_, offset);
    }

    TBuf<TPosition::VECIN> vBuf_;
    GlobalTensor<DType> fmapGm_;
    const uint32_t cin_;
    const uint32_t cin1cin0_;
    const uint32_t hi_;
    const uint32_t wi_;
    const uint16_t padH_;
    const uint16_t padW_;
    const uint32_t tilesH_;
    const uint32_t tilesW_;
    const uint16_t singleShapeCin_;
    const uint16_t singleShapeK_;
    const uint32_t singleShapeBufLength_;
    bool pingFlag_ = true;

    __aicore__ inline void F23TileBox2FmapBox(
        const HWBox& tile,
        HWBox& fmapBox,
        HWPad& padBox)
    {
        //将tile转换成(fmap+pad)中坐标[start,end)
        uint32_t startH = tile.hIdx * F23_WINO_TILE_FMAP_STRIDE_2;
        uint32_t startW = tile.wIdx * F23_WINO_TILE_FMAP_STRIDE_2;
        uint32_t endH = (tile.hIdx + tile.hLength - 1) * F23_WINO_TILE_FMAP_STRIDE_2 + F23_WINO_TILE_SIZE_4;
        uint32_t endW = (tile.wIdx + tile.wLength - 1) * F23_WINO_TILE_FMAP_STRIDE_2 + F23_WINO_TILE_SIZE_4;

        //(fmap+pad)中实际非pad区域的坐标[start,end)
        uint32_t startValidH = padH_;
        uint32_t startValidW = padW_;
        uint32_t endValidH = padH_ + hi_;
        uint32_t endValidW = padW_ + wi_;

        //计算两个区域的相交矩形[start,end)
        uint32_t startFmapH = Std::max(startValidH, startH);
        uint32_t startFmapW = Std::max(startValidW, startW);
        uint32_t endFmapH = Std::min(endH, endValidH);
        uint32_t endFmapW = Std::min(endW, endValidW);

        //tile区域和非padding区域不相交,整个tile都是在padding区域内
        bool allTilesInPadding = startFmapH >= endFmapH || startFmapW >= endFmapW;
        //tile根本不存在,所有数据直接设置成0,endH,endW这时候应该由于产生-1直接溢出了,不能用了
        bool emptyTiles = HWBox::elements(tile) == 0;

        if (allTilesInPadding || emptyTiles) {
            fmapBox.hIdx = 0;
            fmapBox.wIdx = 0;
            fmapBox.hLength = 0;
            fmapBox.wLength = 0;
            padBox.padHTop = 0;
            padBox.padHBottom = emptyTiles ? 0 : endH - startH;
            padBox.padWLeft = 0;
            padBox.padWRight = emptyTiles ? 0 : endW - startW;
        } else {
            fmapBox.hIdx = startFmapH - padH_;
            fmapBox.wIdx = startFmapW - padW_;
            fmapBox.hLength = endFmapH - startFmapH;
            fmapBox.wLength = endFmapW - startFmapW;
            padBox.padHTop = startFmapH - startH;
            padBox.padHBottom = endH - endFmapH;
            padBox.padWLeft = startFmapW - startW;
            padBox.padWRight = endW - endFmapW;
        }
    }


    struct UnfoldParams {
        uint32_t fmapH;
        uint32_t fmapW;
        uint32_t tileH;
        uint32_t tileW;
        uint16_t hRepeatTimesInUnfoldRows;
        uint16_t wRepeatTimesInUnfoldCols;
        uint32_t hElementsInUnfoldRows;
        uint32_t wElementsInUnfoldCols;
        HWPad pad;
        uint32_t wElementsInPadHBlock;
        uint32_t hElementsInPadWBlock;
        uint16_t wRepeatTimesInPadHBlock;
        uint16_t hRepeatTimesInPadWBlock;
        uint16_t cLength;
    };

    //
    // 将[c1,h,w,c0]格式的fmap展开成[c1,tileH*4,tileW*4,c0]格式的tile
    // 本质就是拿4x4的滑窗在fmap上滑出tileH*tileW个4x4的块
    //

    static __simd_vf__ inline void UnfoldFmapVf(
        __ubuf__ DType* buf,
        const UnfoldParams params)
    {
        const uint32_t tileBufWidthC0 = params.tileW * F23_WINO_TILE_SIZE_4;
        const uint32_t tileBufWidth = tileBufWidthC0 * C0<DType>();
        const uint32_t offsetC = tileBufWidth * params.tileH * F23_WINO_TILE_SIZE_4;
        const HWPad& pad = params.pad;
        const bool hasPadding = pad.wLeft != 0 || pad.wRight != 0 || pad.hTop != 0 || pad.hBottom != 0;
        const bool noHPadding = pad.hTop == 0 && pad.hBottom == 0;

        //用for(bool) 模拟if
        for (uint16_t i = 0; i < static_cast<uint16_t>(hasPadding); i++) {
            //变换前给padding区域补0,不一定是最高效的做法
            //但主要网络padding通常较小,预期这块开销占用不大,而且逻辑相对简单
            MicroAPI::MaskReg<DType> zero;
            MicroAPI::Duplicate(zero, 0);
            for (uint16_t c = 0; c < params.cLength; c++) {
                PadFmap(
                    buf + c * offsetC,
                    zero,
                    tileBufWidthC0,
                    tileBufWidth,
                    params.hElementsInPadWBlock,
                    params.wElementsInPadHBlock,
                    params.hRepeatTimesInPadHBlock,
                    params.wRepeatTimesInPadWBlock,
                    params.fmapH,
                    params.fmapW,
                    params.pad);
            }

            //只有左右pad时不需要加同步，因为接下来的列变换会跳过左右pad区域,两者处理不存在冲突
            //等列变换结束后统一的同步即可
            for (uint16_t p = 0; p < static_cast<uint16_t>(noHPadding); p++) {
                MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
            }

        }

        const bool tileHTailRepeatTimes = params.tileH & 1;
        const uint16_t tileHMainRepeatTimes = (params.tileH - tileHTailRepeatTimes) >> 1;
        const uint32_t wLeftPadOffset = pad.wLeft * C0<DType>();

        for (uint16_t c = 0; c < params.cLength; c++) {
            //行变换
            UnfoldColsVf(
                buf + c * offsetC,
                tileBufWidth,
                params.wElementsInUnfoldCols,
                params.tileH,
                wLeftPadOffset,
                params.wRepeatTimesInUnfoldCols,
                tileHMainRepeatTimes,
                tileHTailRepeatTimes);
        }

        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();

        const bool tileWTailRepeatTimes = params.tileW & 1;
        const uint16_t tileWMainRepeatTimes = (params.tileW - tileHTailRepeatTimes) >> 1;

        for (uint16_t c = 0; c < params.cLength; c++) {
            //列变换
            UnfoldRowsVf(
                buf + c * offsetC,
                tileBufWidthC0,
                tileBufWidth,
                params.hElementsInUnfoldRows,
                params.tileW,
                params.hRepeatTimesInUnfoldRows,
                tileWMainRepeatTimes,
                tileWTailRepeatTimes);
        }
    }


    // 执行行变换
    // 原始布局S:
    //
    //      w0 w1 w2 w3 w4 w5 w6 w7
    //     +--+--+--+--+--+--+--+--+
    //  h0 |xx|xx|xx|xx|xx|xx|  |  |
    //     +--+--+--+--+--+--+--+--+
    //  h1 |xx|xx|xx|xx|xx|xx|  |  |
    //     +--+--+--+--+--+--+--+--+
    //  h2 |xx|xx|xx|xx|xx|xx|  |  |
    //     +--+--+--+--+--+--+--+--+
    //  h3 |xx|xx|xx|xx|xx|xx|  |  |
    //     +--+--+--+--+--+--+--+--+
    //  h4 |xx|xx|xx|xx|xx|xx|  |  |
    //     +--+--+--+--+--+--+--+--+
    //  h5 |xx|xx|xx|xx|xx|xx|  |  |
    //     +--+--+--+--+--+--+--+--+
    //  h6 |  |  |  |  |  |  |  |  |
    //     +--+--+--+--+--+--+--+--+
    //  h7 |  |  |  |  |  |  |  |  |
    //     +--+--+--+--+--+--+--+--+
    //
    // 变换后内存:
    //
    //           w0         w1    w2-w4     w5     w6 w7
    //      +---------+---------+-------+---------+--+--+
    //  Th0 |h0w0-h2w0|h0w1-h2w1|.......|h0w5-h2w5|  |  |
    //      +---------+---------+-------+---------+--+--+
    //  Th1 |h1w0+h2w0|h1w1+h2w1|.......|h1w5+h2w5|  |  |
    //      +---------+---------+-------+---------+--+--+
    //  Th2 |h2w0-h1w0|h2w1-h1w1|.......|h2w5-h1w5|  |  |
    //      +---------+---------+-------+---------+--+--+
    //  Th3 |h1w0-h3w0|h1w1-h3w1|.......|h1w5-h3w5|  |  |
    //      +---------+---------+-------+---------+--+--+
    //  Th4 |h2w0-h4w0|h2w1-h4w1|.......|h2w5-h4w5|  |  |
    //      +---------+---------+-------+---------+--+--+
    //  Th5 |h4w0+h3w0|h4w1+h3w1|.......|h4w5+h3w5|  |  |
    //      +---------+---------+-------+---------+--+--+
    //  Th6 |h4w0-h3w0|h4w1-h3w1|.......|h4w5-h3w5|  |  |
    //      +---------+---------+-------+---------+--+--+
    //  Th7 |h3w0-h5w0|h3w1-h5w1|.......|h3w5-h5w5|  |  |
    //      +---------+---------+-------+---------+--+--+
    //
    // 由于fmap数据在原始内存左上角,所以需要从最下方的tile开始执行滑窗
    // 即先在h2-h5上滑窗后将结果写入h4-h7,在从h0-h3上处理并将结果写入h0-h3
    //

    static __simd_callee__ inline void UnfoldColsVf(
        __ubuf__ DType* buf,
        const uint32_t tileBufWidth,
        const uint32_t wValidElements,
        const uint32_t tileH,
        const uint32_t wLeftPaddingOffset,
        const uint16_t wRepeatTimes,
        const uint16_t tileHMainRepeatTimes,
        const bool tileHTailRepeatTimes)
    {
        MicroAPI::RegTensor<DType> s0;
        MicroAPI::RegTensor<DType> s1;
        MicroAPI::RegTensor<DType> s2;
        MicroAPI::RegTensor<DType> s3;

        MicroAPI::RegTensor<DType> d0;
        MicroAPI::RegTensor<DType> d1;
        MicroAPI::RegTensor<DType> d2;
        MicroAPI::RegTensor<DType> d3;

        constexpr uint16_t vLen = VL<DType>();

        uint32_t maskValue = wValidElements;
        for (uint16_t i = 0; i < wRepeatTimes; i++) {
            MicroAPI::MaskReg mask = MicroAPI::UpdateMask<DType>(maskValue);
            //循环fmapW,同时要跳过最左侧的padding区域
            const uint32_t wOffset = i * vLen + wLeftPaddingOffset;

            //
            // 从最下方的tile开始滑窗
            // 先读取fmap最底下2行,每次循环往上读2行凑成4行执行变换
            // 但若一个滑窗在fmap的1-4行分别读入s0,s1,s2,s3
            // 在下一个滑窗s0,s1就变成2-3行,不考虑重新读取的话1-2行就只能读入s2,s3,滑窗1-4行就变成s2,s3,s0,s1
            // 如果将s0,s1的数据拷贝到s2,s3可能会产生多余的指令并且产生数据依赖降低性能
            // vf内scalar能力较弱也不一定能用指针引用之类的方法处理,而且也会让编译器难以优化
            // 所以这里展开循环,一个循环内处理2个连续滑窗,如果滑窗为奇数,则通过tileHTailRepeatTimes额外执行一次滑窗
            //
            const uint32_t lastFmapTileOffset = wOffset + tileBufWidth * tileH * F23_WINO_TILE_FMAP_STRIDE_2;
            MicroAPI::LoadAlign(s2, buf + lastFmapTileOffset + 2 * tileBufWidth);
            MicroAPI::LoadAlign(s3, buf + lastFmapTileOffset + 3 * tileBufWidth);

            for (uint16_t th = 0; th < tileHMainRepeatTimes; th++) {
                const uint16_t thIdx = tileH - th * 2 - 1;
                const uint32_t fmapOffset = wOffset + tileBufWidth * thIdx * F23_WINO_TILE_FMAP_STRIDE_2;
                MicroAPI::LoadAlign(s0, buf + fmapOffset);
                MicroAPI::LoadAlign(s1, buf + fmapOffset + tileBufWidth);

                TransformVf(s0, s1, s2, s3, d0, d1, d2, d3, mask);

                const uint32_t tileOffset = wOffset + tileBufWidth * thIdx * F23_WINO_TILE_SIZE_4;
                MicroAPI::StoreAlign(tileOffset, d0, mask);
                MicroAPI::StoreAlign(tileOffset + tileBufWidth, d1, mask);
                MicroAPI::StoreAlign(tileOffset + tileBufWidth * 2, d2, mask);
                MicroAPI::StoreAlign(tileOffset + tileBufWidth * 3, d3, mask);

                MicroAPI::LoadAlign(s2, buf + fmapOffset - tileBufWidth * 2);
                MicroAPI::LoadAlign(s3, buf + fmapOffset - tileBufWidth);

                TransformVf(s2, s3, s0, s1, d0, d1, d2, d3, mask);
                MicroAPI::StoreAlign(tileOffset - tileBufWidth * 4, d0, mask);
                MicroAPI::StoreAlign(tileOffset - tileBufWidth * 3, d1, mask);
                MicroAPI::StoreAlign(tileOffset - tileBufWidth * 2, d2, mask);
                MicroAPI::StoreAlign(tileOffset - tileBufWidth, d3, mask);
            }

            for (uint16_t th = 0; th < static_cast<uint16_t>(tileHTailRepeatTimes); th++) {
                MicroAPI::LoadAlign(s0, buf + wOffset);
                MicroAPI::LoadAlign(s1, buf + wOffset + tileBufWidth);
                TransformVf(s0, s1, s2, s3, d0, d1, d2, d3, mask);
                MicroAPI::StoreAlign(buf, d0, mask);
                MicroAPI::StoreAlign(buf + tileBufWidth, d1, mask);
                MicroAPI::StoreAlign(buf + tileBufWidth * 2, d2, mask);
                MicroAPI::StoreAlign(buf + tileBufWidth * 3, d3, mask);
            }
        }
    }

    // 在行变换的结果上执行变换
    // 原始布局S:
    //
    //      w0 w1 w2 w3 w4 w5 w6 w7
    //     +--+--+--+--+--+--+--+--+
    //  h0 |xx|xx|xx|xx|xx|xx|  |  |
    //     +--+--+--+--+--+--+--+--+
    //  h1 |xx|xx|xx|xx|xx|xx|  |  |
    //     +--+--+--+--+--+--+--+--+
    //  h2 |xx|xx|xx|xx|xx|xx|  |  |
    //     +--+--+--+--+--+--+--+--+
    //  h3 |xx|xx|xx|xx|xx|xx|  |  |
    //     +--+--+--+--+--+--+--+--+
    //  h4 |xx|xx|xx|xx|xx|xx|  |  |
    //     +--+--+--+--+--+--+--+--+
    //  h5 |xx|xx|xx|xx|xx|xx|  |  |
    //     +--+--+--+--+--+--+--+--+
    //  h6 |xx|xx|xx|xx|xx|xx|  |  |
    //     +--+--+--+--+--+--+--+--+
    //  h7 |xx|xx|xx|xx|xx|xx|  |  |
    //     +--+--+--+--+--+--+--+--+
    //
    // 变换后内存:
    //
    //           w0        w1     w2-w6     w7
    //      +---------+---------+-------+---------+
    //  Th0 |h0w0-h0w2|h0w0+h0w2|.......|h0w3-h0w5|
    //      +---------+---------+-------+---------+
    //  Th1 |h1w0-h1w2|h1w0+h1w2|.......|h1w3-h1w5|
    //      +---------+---------+-------+---------+
    //  Th2 |h2w0-h2w2|h2w0+h2w2|.......|h2w3-h2w5|
    //      +---------+---------+-------+---------+
    //  Th3 |h3w0-h3w2|h3w0+h3w2|.......|h3w3-h3w5|
    //      +---------+---------+-------+---------+
    //  Th4 |h4w0-h4w2|h4w0+h4w2|.......|h4w3-h4w5|
    //      +---------+---------+-------+---------+
    //  Th5 |h5w0-h5w2|h5w0+h5w2|.......|h5w3-h5w5|
    //      +---------+---------+-------+---------+
    //  Th6 |h6w0-h6w2|h6w0+h6w2|.......|h6w3-h6w5|
    //      +---------+---------+-------+---------+
    //  Th7 |h7w0-h7w2|h7w0+h7w2|.......|h7w3-h7w5|
    //      +---------+---------+-------+---------+
    //
    // 和列变换一样,从最右侧的tile开始执行变换
    //
    static __simd_callee__ inline void UnfoldRowsVf(
        __ubuf__ DType* buf,
        const uint32_t tileBufWidthC0,
        const uint32_t tileBufWidth,
        const uint32_t hValidElements,
        const uint32_t tileW,
        const uint16_t hRepeatTimes,
        const uint16_t tileWMainRepeatTimes,
        const bool tileWTailRepeatTimes)
    {
        MicroAPI::RegTensor<DType> s0;
        MicroAPI::RegTensor<DType> s1;
        MicroAPI::RegTensor<DType> s2;
        MicroAPI::RegTensor<DType> s3;

        MicroAPI::RegTensor<DType> d0;
        MicroAPI::RegTensor<DType> d1;
        MicroAPI::RegTensor<DType> d2;
        MicroAPI::RegTensor<DType> d3;

        constexpr uint16_t c0 = C0<DType>();

        uint32_t maskValue = hValidElements;
        for (uint16_t i = 0; i < hRepeatTimes; i++) {
            MicroAPI::MaskReg mask = MicroAPI::UpdateMask<DType>(maskValue);
            const uint32_t hOffset = i * BLK_COUNT_IN_VL * tileBufWidth;

            const uint32_t lastFmapTileOffset = hOffset + tileW * F23_WINO_TILE_FMAP_STRIDE_2 * c0;
            MicroAPI::LoadAlign(s2, buf + lastFmapTileOffset + 2 * c0, tileBufWidthC0, mask);
            MicroAPI::LoadAlign(s3, buf + lastFmapTileOffset + 3 * c0, tileBufWidthC0, mask);

            for (uint16_t tw = 0; tw < tileWMainRepeatTimes; tw++) {
                const uint16_t twIdx = tileW - tw * 2 - 1;
                const uint32_t fmapOffset = hOffset + twIdx * F23_WINO_TILE_FMAP_STRIDE_2 * c0;
                MicroAPI::LoadAlign(s0, buf + fmapOffset, tileBufWidthC0, mask);
                MicroAPI::LoadAlign(s1, buf + fmapOffset + c0, tileBufWidthC0, mask);

                TransformVf(s0, s1, s2, s3, d0, d1, d2, d3, mask);

                const uint32_t tileOffset = hOffset + twIdx * F23_WINO_TILE_SIZE_4 * c0;
                MicroAPI::StoreAlign(tileOffset, d0, tileBufWidthC0, mask);
                MicroAPI::StoreAlign(tileOffset + c0, d1, tileBufWidthC0, mask);
                MicroAPI::StoreAlign(tileOffset + c0 * 2, d2, tileBufWidthC0, mask);
                MicroAPI::StoreAlign(tileOffset + c0 * 3, d3, tileBufWidthC0, mask);

                MicroAPI::LoadAlign(s2, buf + fmapOffset - c0 * 2, tileBufWidthC0, mask);
                MicroAPI::LoadAlign(s3, buf + fmapOffset - c0, tileBufWidthC0, mask);

                TransformVf(s2, s3, s0, s1, d0, d1, d2, d3, mask);
                MicroAPI::StoreAlign(tileOffset - c0 * 4, d0, mask);
                MicroAPI::StoreAlign(tileOffset - c0 * 3, d1, mask);
                MicroAPI::StoreAlign(tileOffset - c0 * 2, d2, mask);
                MicroAPI::StoreAlign(tileOffset - c0, d3, mask);
            }

            for (uint16_t th = 0; th < static_cast<uint16_t>(tileWTailRepeatTimes); th++) {
                MicroAPI::LoadAlign(s0, buf + hOffset, tileBufWidthC0, mask);
                MicroAPI::LoadAlign(s1, buf + hOffset + c0, tileBufWidthC0, mask);
                TransformVf(s0, s1, s2, s3, d0, d1, d2, d3, mask);
                MicroAPI::StoreAlign(buf, d0, mask);
                MicroAPI::StoreAlign(buf + c0, d1, mask);
                MicroAPI::StoreAlign(buf + c0 * 2, d2, mask);
                MicroAPI::StoreAlign(buf + c0 * 3, d3, mask);
            }
        }
    }

    //        wElementsInPadHBlock(fmapW+wLeft+wRight)
    //       +---------------------------------+
    //   hTop|             Padding             |
    //       +--wLeft--+-------------+---------+
    //       |         |  NoPadding  |         |
    //       | Padding |fmapH x fmapW| Padding |hElementsInPadWBlock(fmapH)
    //       |         |             |         |
    //       +---------+-------------+--wRight-+
    //       |             Padding             |hButton
    //       +---------------------------------+

    static __simd_callee__ inline void PadFmap(
        __ubuf__ DType* buf,
        MicroAPI::RegTensor<DType>& value,
        uint32_t tileBufWidthC0,
        uint32_t tileBufWidth,
        uint32_t hElementsInPadWBlock,
        uint32_t wElementsInPadHBlock,
        uint16_t hRepeatTimesInPadWBlock,
        uint16_t wRepeatTimesInPadHBlock,
        uint32_t fmapH,
        uint32_t fmapW,
        const HWPad& pad)
    {
        constexpr uint16_t vLen = VL<DType>();
        constexpr uint16_t c0 = C0<DType>();

        for (uint16_t i = 0; i < pad.hTop; i++) {
            uint32_t maskValue = wElementsInPadHBlock;
            uint32_t offset = i * tileBufWidth;
            for (uint16_t w = 0; w < wRepeatTimesInPadHBlock; w++) {
                MicroAPI::MaskReg<DType> mask = MicroAPI::UpdateMask<DType>(maskValue);
                MicroAPI::StoreAlign(buf + offset + w * vLen, value, mask);
            }
        }

        for (uint16_t i = 0; i < pad.hBottom; i++) {
            uint32_t maskValue = wElementsInPadHBlock;
            uint32_t offset = (i + pad.hTop + fmapH) * tileBufWidth;
            for (uint16_t w = 0; w < wRepeatTimesInPadHBlock; w++) {
                MicroAPI::MaskReg<DType> mask = MicroAPI::UpdateMask<DType>(maskValue);
                MicroAPI::StoreAlign(buf + offset +  w * vLen, value, mask);
            }
        }

        for (uint16_t i = 0; i < pad.wLeft; i++) {
            uint32_t maskValue = hElementsInPadWBlock;
            uint32_t offset = pad.hTop * tileBufWidth + i * c0;
            for (uint16_t w = 0; w < hRepeatTimesInPadWBlock; w++) {
                MicroAPI::MaskReg<DType> mask = MicroAPI::UpdateMask<DType>(maskValue);
                MicroAPI::StoreAlign(buf + offset + w * BLK_COUNT_IN_VL * tileBufWidth, value, tileBufWidthC0, mask);
            }
        }

        for (uint16_t i = 0; i < pad.wRight; i++) {
            uint32_t maskValue = hElementsInPadWBlock;
            uint32_t offset = pad.hTop * tileBufWidth + (i + pad.wLeft + fmapW) * c0;
            for (uint16_t w = 0; w < hRepeatTimesInPadWBlock; w++) {
                MicroAPI::MaskReg<DType> mask = MicroAPI::UpdateMask<DType>(maskValue);
                MicroAPI::StoreAlign(buf + offset + w * BLK_COUNT_IN_VL * tileBufWidth, value, tileBufWidthC0, mask);
            }
        }
    }

    static __simd_callee__ inline void TransformVf(
        MicroAPI::RegTensor<DType>& s0,
        MicroAPI::RegTensor<DType>& s1,
        MicroAPI::RegTensor<DType>& s2,
        MicroAPI::RegTensor<DType>& s3,
        MicroAPI::RegTensor<DType>& d0,
        MicroAPI::RegTensor<DType>& d1,
        MicroAPI::RegTensor<DType>& d2,
        MicroAPI::RegTensor<DType>& d3,
        MicroAPI::MaskReg& mask)
    {
        MicroAPI::Sub(d0, s0, s2, mask);
        MicroAPI::Add(d1, s1, s2, mask);
        MicroAPI::Sub(d2, s2, s1, mask);
        MicroAPI::Sub(d3, s1, s3, mask);
    }


    static __aicore__ inline uint32_t CalculateF23Tiles(uint32_t dim)
    {
        //F23下,Fmap正变换是一个4*4的滑窗在Fmap上按stride为2滑动
        //所以dim至少pad到4且是2的倍数
        uint32_t d = Ops::Base::CeilAlign(dim, 2u);
        return Std::max(d, F23_WINO_TILE_SIZE_4) / F23_WINO_TILE_FMAP_STRIDE_2 - 1;
    }
};

#endif //CONV_BP_WINOGRAD_H