#ifndef CONV_BP_BL1_FULLLOAD_H
#define CONV_BP_BL1_FULLLOAD_H

#include "conv_bp_bl1_fullload_config.h"
#include "conv_bp_bl1_fullload_compute.h"
#include "conv_bp_bl1_fullload_data_blocks.h"

namespace BpFullLoad {

// BL1 全载模板顶层串联类：BL1FullLoadCompute（单块 K 全载计算）+ BL1FullLoadBlockIterator
// （基本块走位，cin-major 块序保 B1 驻留）编排——上板可直调形态。
//
// 调用契约（三步，顺序不可变；fmap/dy/y 三个 GM 地址全部裸传，调用侧不构造任何
// GlobalTensor——__gm__ uint8_t* 即 GM_ADDR 展开形态，kernel 入口参数直传零转换）：
//   1. Init 一次：fmapGm/dyGm 输入 GM 裸地址 + config（shape + tiling）+ blockNum
//      （生产路径 0 = GetBlockNum()；kernel 直调/UT 场景 <<<blockDim>>> 上下文可能未按
//       aclnn 语义填充该寄存器，显式传入更稳），类内部构造 GlobalTensor 并预置事件链
//   2. Process 一次：yGm 输出 GM 裸地址（生产布局 [cout][cin][dhwK] 全局 ND），内部
//      完成全部块循环、装载、计算与块间同步
//   3. Drain 必须调用：消费跨块/跨半区背压残留 Set——漏调则残留 flag 污染同核
//      后续 kernel 的首块装载（TQue Reset 残留 freeBufEvt 消费先例）
//
// 迭代块宽 = tiling.singleShapeAlignedCout（host 侧应取 16 对齐块宽；sim 旧散装编排
// 传 raw singleShapeCout，16 对齐时两者等价）
template <typename SrcT>
class ConvBackpropFilterBL1FullLoad {
public:
    inline __aicore__ void Init(GM_ADDR fmapGm, GM_ADDR dyGm,
                                const BL1FullLoadConfig& config, uint32_t blockNum = 0)
    {
        config_ = config;
        blockNum_ = blockNum;
        fmap_.SetGlobalBuffer(reinterpret_cast<__gm__ SrcT*>(fmapGm));
        dy_.SetGlobalBuffer(reinterpret_cast<__gm__ SrcT*>(dyGm));
        computer_.Init(fmap_, dy_); // 事件预置位（跨块/跨半区背压链全套）
    }

    inline __aicore__ void Process(GM_ADDR yGm)
    {
        y_.SetGlobalBuffer(reinterpret_cast<__gm__ SrcT*>(yGm));
        BL1FullLoadBlockIterator blockIter(config_.shape.cout, config_.shape.cin,
                                           config_.tiling.singleShapeAligned16Cout,
                                           config_.tiling.singleShapeFullLoadAligned16Cin,
                                           blockNum_);
        while (blockIter.More()) {
            BpUtils::CoutCinRange cRange;
            blockIter.GetLocalBlock(cRange);
            computer_.IterateK(config_, cRange, y_, CalcYBase(cRange));
            blockIter.Next();
        }
    }

    inline __aicore__ void Drain()
    {
        computer_.Drain();
    }

private:
    // 生产布局块基址：y[cout][cin][dhwK] 全局 ND，
    // 本块 = coutIdx*cin*dhwK + cinIdx*dhwK（与 host golden 同式）
    inline __aicore__ uint64_t CalcYBase(const BpUtils::CoutCinRange& cRange) const
    {
        const uint32_t dhwk = config_.shape.dk * config_.shape.hk * config_.shape.wk;
        return static_cast<uint64_t>(cRange.coutIdx) * config_.shape.cin * dhwk +
               static_cast<uint64_t>(cRange.cinIdx) * dhwk;
    }

    BL1FullLoadConfig config_ = {};
    uint32_t blockNum_ = 0;
    AscendC::GlobalTensor<SrcT> fmap_;
    AscendC::GlobalTensor<SrcT> dy_;
    AscendC::GlobalTensor<SrcT> y_;
    BL1FullLoadCompute<SrcT> computer_;
};

} // namespace BpFullLoad
#endif // CONV_BP_BL1_FULLLOAD_H
