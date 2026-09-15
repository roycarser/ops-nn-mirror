#ifndef CONV_BP_BL1_FULLLOAD_CONFIG_H
#define CONV_BP_BL1_FULLLOAD_CONFIG_H

namespace BpFullLoad {

struct ShapeAttribute {
    uint16_t hk;
    uint16_t wk;
    uint16_t dk;
    uint16_t hPad;
    uint16_t wPad;
    uint16_t dPad;
    uint32_t batch;
    uint32_t cout;
    uint32_t dout;
    uint32_t hout;
    uint32_t wout;
    uint32_t cin;
    uint32_t din;
    uint32_t hin;
    uint32_t win;
};

struct FullLoadTiling {
    //传进来的基本块c轴要16byte对齐
    uint16_t singleShapeFullLoadAligned16Cin;
    uint16_t singleShapeAligned16Cout;
    uint16_t kl0HoWo;
};

struct BL1FullLoadConfig {
    ShapeAttribute shape;
    FullLoadTiling tiling;
};

} // namespace BpFullLoad
#endif // CONV_BP_BL1_FULLLOAD_CONFIG_H