#ifndef YOLOHANDPOSEPOSTPROCESS_H
#define YOLOHANDPOSEPOSTPROCESS_H

#include <vector>
#include <string>
#include <map>
#include <dxrt/dxrt_cxx_api.h>
#include "Common.h"

class YoloHandPosePostProcess {
public:
    YoloHandPosePostProcess(int inputW, int inputH, float scoreThresh, float nmsThresh, int kptCount = 21);
    ~YoloHandPosePostProcess() = default;

    std::vector<DetectionBox> postprocess(const dxrt::TensorPtrs& outputs, int originalW, int originalH);

    void setAnchors(const std::map<int, std::vector<std::pair<int, int>>>& anchors);

private:
    int m_inputWidth;
    int m_inputHeight;
    float m_scoreThresh;
    float m_nmsThresh;
    int m_kptCount;

    struct HandResult {
        std::vector<float> box; // x1, y1, x2, y2
        float confidence;
        std::vector<float> landmarks; // x, y, score * kptCount

        float area() const { return (box[2] - box[0]) * (box[3] - box[1]); }
        float iou(const HandResult& other) const;
    };

    std::map<int, std::vector<std::pair<int, int>>> m_anchors;

    // PPU POSE output (DevicePose_t, YOLOv5-style anchor-based, 17 kpt max)
    std::vector<HandResult> decodePose(const dxrt::TensorPtrs& outputs);
    // Raw FLOAT tensor — YOLOv8-style: shape (1, 5+kpt*3, N) transposed layout
    std::vector<HandResult> decodeFloat(const dxrt::TensorPtrs& outputs);
    std::vector<HandResult> nms(const std::vector<HandResult>& dets);
};

#endif // YOLOHANDPOSEPOSTPROCESS_H
