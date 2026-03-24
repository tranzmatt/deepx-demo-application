#ifndef YOLOPOSEPOSTPROCESS_H
#define YOLOPOSEPOSTPROCESS_H

#include <vector>
#include <string>
#include <map>
#include <dxrt/dxrt_api.h>
#include "Common.h"

class YoloPosePostProcess {
public:
    YoloPosePostProcess(int inputW, int inputH, float scoreThresh, float nmsThresh);
    ~YoloPosePostProcess() = default;

    std::vector<DetectionBox> postprocess(const dxrt::TensorPtrs& outputs, int originalW, int originalH);

private:
    int m_inputWidth;
    int m_inputHeight;
    float m_scoreThresh;
    float m_nmsThresh;

    struct PoseResult {
        std::vector<float> box; // x1, y1, x2, y2
        float confidence;
        std::vector<float> landmarks; 
        
        float area() const { return (box[2] - box[0]) * (box[3] - box[1]); }
        float iou(const PoseResult& other) const;
    };

    std::map<int, std::vector<std::pair<int, int>>> m_anchors;

    std::vector<PoseResult> decode(const dxrt::TensorPtrs& outputs);
    std::vector<PoseResult> nms(const std::vector<PoseResult>& dets);
};

#endif // YOLOPOSEPOSTPROCESS_H
