#ifndef CONV_BP_BL1_DLOAD_H
#define CONV_BP_BL1_DLOAD_H

#include "conv_bp_bl1_dload_config.h"
#include "conv_bp_bl1_dload_compute.h"
#include "../util/conv_bp_common_data_blocks.h" // 蛇形分核 BlockIterator（原空壳转发头已删除）

namespace BpDLoad {

// BL1 全载模板顶层串联类：BL1DLoadCompute（单块 K 全载计算）+ 蛇形分核块走位
// （BpUtils::BlockIterator，第十七轮起——原 cin 绑核迭代器随 B1 半区化删除）编排
// ——上板可直调形态。
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
class ConvBackpropFilterBL1DLoad {
public:
    inline __aicore__ void Init(GM_ADDR fmapGm, GM_ADDR dyGm,
                                const BL1DLoadConfig& config, uint32_t blockNum = 0)
    {
        config_ = config;
        blockNum_ = blockNum;
        fmap_.SetGlobalBuffer(reinterpret_cast<__gm__ SrcT*>(fmapGm));
        dy_.SetGlobalBuffer(reinterpret_cast<__gm__ SrcT*>(dyGm));
        // 事件预置位（跨块/跨半区背压链全套）+ ★第二十轮修正 1：bBufCnt_/bBufTileBytes_
        // 在此一次计算（tiling 上界口径跨块恒定，config 传入）
        computer_.Init(fmap_, dy_, config_);
    }

    inline __aicore__ void Process(GM_ADDR yGm)
    {
        y_.SetGlobalBuffer(reinterpret_cast<__gm__ SrcT*>(yGm));
        // ★第十七轮专家裁决（历史轮次措辞保留）：分核改蛇形走位（不绑 cin）——B1 半区化（每块每 batch 必装
        // fmap）后无需 cin 绑核驻留复用，改用 util 蛇形 BlockIterator（SwizzleTopology2D
        // 核网格蛇形递进，L2 友好）。IterDir=CIN：cout 块沿拓扑 H、cin 块沿 W（与旧
        // cin-major 块序最接近的拓扑方向）。空块跳过（尾轮某核可能无块，valid=false 时
        // length=0——IterateK 不调用，事件链不受影响：Init 预置/Drain 消费为 kernel 级）
        auto blockIter = BpUtils::BlockIterator<BpUtils::BlockIterDirection::CIN>::Create(
            false, config_.shape.cout, config_.shape.cin, config_.tiling.singleShapeAligned16Cout,
            config_.tiling.singleShapeAligned16Cin, blockNum_);
        while (blockIter.More()) {
            BpUtils::CoutCinRange cRange;
            if (blockIter.GetLocalBlock(cRange)) {
                computer_.IterateK(config_, cRange, y_, CalcYBase(cRange));
            }
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

    BL1DLoadConfig config_ = {};
    uint32_t blockNum_ = 0;
    AscendC::GlobalTensor<SrcT> fmap_;
    AscendC::GlobalTensor<SrcT> dy_;
    AscendC::GlobalTensor<SrcT> y_;
    BL1DLoadCompute<SrcT> computer_;
};

} // namespace BpDLoad
#endif // CONV_BP_BL1_DLOAD_H
