#ifndef YOLOV8PPUPOSTPROCESS_H
#define YOLOV8PPUPOSTPROCESS_H

#include <vector>
#include <string>
#include <dxrt/dxrt_api.h>
#include "Common.h"

class YoloV8PpuPostProcess {
public:
    YoloV8PpuPostProcess(int inputW, int inputH, float scoreThresh, float nmsThresh);
    ~YoloV8PpuPostProcess() = default;

    std::vector<DetectionBox> postprocess(const dxrt::TensorPtrs& outputs, int originalW, int originalH);

private:
    int m_inputWidth;
    int m_inputHeight;
    float m_scoreThresh;
    float m_nmsThresh;
    
    // NMS helper
    std::vector<DetectionBox> applyNMS(std::vector<DetectionBox>& detections);
};

#endif // YOLOV8PPUPOSTPROCESS_H
