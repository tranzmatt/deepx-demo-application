#ifndef YOLO_POSTPROCESS_H
#define YOLO_POSTPROCESS_H

#include <dxrt/dxrt_cxx_api.h>
#include <vector>
#include <string>
#include <map>
#include "Common.h"

class YoloPostProcess {
public:
    YoloPostProcess(int inputW, int inputH, float objThresh, float scoreThresh, float nmsThresh);
    YoloPostProcess(); // Default

    std::vector<DetectionBox> postprocess(const dxrt::TensorPtrs& outputs, int imgW, int imgH);

    void setThresholds(float objThresh, float scoreThresh, float nmsThresh);

private:
    int m_inputW;
    int m_inputH;
    float m_objThreshold;
    float m_scoreThreshold;
    float m_nmsThreshold;
    
    std::vector<std::string> m_classNames;

public:
    void setClassNames(const std::vector<std::string>& names) { m_classNames = names; }

    // Model specific
    std::map<int, std::vector<std::pair<int, int>>> m_anchorsByStrides;

    struct PpuResult {
        float x, y, w, h; // center x, y, width, height
        float confidence;
        int classId;
        
        float area() const { return w * h; }
        float iou(const PpuResult& other) const;
    };

    std::vector<PpuResult> decodePpuOutputs(const dxrt::TensorPtrs& outputs) const;
    std::vector<PpuResult> applyNms(const std::vector<PpuResult>& detections) const;
};

#endif // YOLO_POSTPROCESS_H
