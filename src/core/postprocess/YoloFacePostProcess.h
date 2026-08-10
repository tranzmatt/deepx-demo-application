#ifndef YOLO_FACE_POSTPROCESS_H
#define YOLO_FACE_POSTPROCESS_H

#include <dxrt/dxrt_cxx_api.h>
#include <vector>
#include <string>
#include <map>
#include "Common.h"

class YoloFacePostProcess {
public:
    YoloFacePostProcess(int inputW, int inputH, float confThresh, float nmsThresh);
    
    std::vector<DetectionBox> postprocess(const dxrt::TensorPtrs& outputs, int imgW, int imgH);

private:
    int m_inputW;
    int m_inputH;
    float m_confThresh;
    float m_nmsThresh;
    
    // Stride -> Anchors [ (w,h), (w,h), (w,h) ]
    std::map<int, std::vector<std::pair<float, float>>> m_anchors;

    struct FaceResult {
        float x, y, w, h;
        float confidence;
        std::vector<float> landmarks; // x1, y1, x2, y2...
        
        float area() const { return w * h; }
        float iou(const FaceResult& other) const;
    };

    std::vector<FaceResult> decode(const float* data, int numBoxes);
    std::vector<FaceResult> nms(const std::vector<FaceResult>& detections);
};

#endif // YOLO_FACE_POSTPROCESS_H
