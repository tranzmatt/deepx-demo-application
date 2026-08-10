#include "YoloPostProcess.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <dxrt/dxrt_cxx_api.h>

// Helper for IoU
float YoloPostProcess::PpuResult::iou(const PpuResult& other) const {
    float x1 = x - w / 2.0f;
    float y1 = y - h / 2.0f;
    float x2 = x + w / 2.0f;
    float y2 = y + h / 2.0f;

    float ox1 = other.x - other.w / 2.0f;
    float oy1 = other.y - other.h / 2.0f;
    float ox2 = other.x + other.w / 2.0f;
    float oy2 = other.y + other.h / 2.0f;

    float xx1 = std::max(x1, ox1);
    float yy1 = std::max(y1, oy1);
    float xx2 = std::min(x2, ox2);
    float yy2 = std::min(y2, oy2);

    float w_inter = std::max(0.0f, xx2 - xx1);
    float h_inter = std::max(0.0f, yy2 - yy1);
    float inter_area = w_inter * h_inter;
    float union_area = area() + other.area() - inter_area;

    return (union_area > 0) ? (inter_area / union_area) : 0.0f;
}

YoloPostProcess::YoloPostProcess(int inputW, int inputH, float objThresh, float scoreThresh, float nmsThresh)
    : m_inputW(inputW), m_inputH(inputH), m_objThreshold(objThresh), m_scoreThreshold(scoreThresh), m_nmsThreshold(nmsThresh) {
    // Default anchors for YOLOv5s
    m_anchorsByStrides = {
        {8, {{10, 13}, {16, 30}, {33, 23}}},
        {16, {{30, 61}, {62, 45}, {59, 119}}},
        {32, {{116, 90}, {156, 198}, {373, 326}}}
    };
}

YoloPostProcess::YoloPostProcess() : YoloPostProcess(640, 640, 0.25f, 0.3f, 0.45f) {}

void YoloPostProcess::setThresholds(float objThresh, float scoreThresh, float nmsThresh) {
    m_objThreshold = objThresh;
    m_scoreThreshold = scoreThresh;
    m_nmsThreshold = nmsThresh;
}

std::vector<DetectionBox> YoloPostProcess::postprocess(const dxrt::TensorPtrs& outputs, int imgW, int imgH) {
    if (outputs.empty()) return {};

    // Validate type
    if (outputs.front()->type() != dxrt::DataType::BBOX) {
        std::cerr << "Error: Output tensor is not BBOX type.\n";
        return {};
    }

    auto ppuResults = decodePpuOutputs(outputs);
    auto nmsResults = applyNms(ppuResults);

    std::vector<DetectionBox> finalResults;
    for (const auto& res : nmsResults) {
        DetectionBox box;
        box.classId = res.classId;
        box.confidence = res.confidence;
        box.classId = res.classId;
        box.confidence = res.confidence;
        
        if (res.classId >= 0 && res.classId < m_classNames.size()) {
            box.label = m_classNames[res.classId];
        } else {
            box.label = "Class " + std::to_string(res.classId);
        }

        // Convert center-wh to top-left-wh and normalize
        float x1 = res.x - res.w / 2.0f;
        float y1 = res.y - res.h / 2.0f;
        
        // Clamp
        x1 = std::max(0.0f, x1);
        y1 = std::max(0.0f, y1);
        
        // Normalize
        box.box = cv::Rect2f(x1 / m_inputW, y1 / m_inputH, res.w / m_inputW, res.h / m_inputH);

        // Color (simple hash based)
        int r = (res.classId * 50) % 255;
        int g = (res.classId * 80 + 100) % 255;
        int b = (res.classId * 120 + 200) % 255;
        box.color = cv::Scalar(b, g, r);

        finalResults.push_back(box);
    }
    return finalResults;
}

std::vector<YoloPostProcess::PpuResult> YoloPostProcess::decodePpuOutputs(const dxrt::TensorPtrs& outputs) const {
    std::vector<PpuResult> detections;
    auto numElements = outputs[0]->shape()[1];
    auto* rawData = static_cast<dxrt::DeviceBoundingBox_t*>(outputs[0]->data());

    for (int i = 0; i < numElements; i++) {
        auto& bbox = rawData[i];
        if (bbox.score < m_scoreThreshold) continue;

        int gX = bbox.grid_x;
        int gY = bbox.grid_y;
        
        // Determine stride based on layer index 
        // Note: Assuming layer_idx maps to 0, 1, 2 for strides 8, 16, 32
        int stride = 8;
        if (bbox.layer_idx == 1) stride = 16;
        else if (bbox.layer_idx == 2) stride = 32;

        const auto& anchors = m_anchorsByStrides.at(stride);
        
        PpuResult res;
        res.x = (bbox.x * 2.0f - 0.5f + gX) * stride;
        res.y = (bbox.y * 2.0f - 0.5f + gY) * stride;
        res.w = (bbox.w * bbox.w * 4.0f) * anchors[bbox.box_idx].first;
        res.h = (bbox.h * bbox.h * 4.0f) * anchors[bbox.box_idx].second;
        res.confidence = bbox.score;
        res.classId = bbox.label;

        detections.push_back(res);
    }
    return detections;
}

std::vector<YoloPostProcess::PpuResult> YoloPostProcess::applyNms(const std::vector<PpuResult>& detections) const {
    if (detections.empty()) return {};

    std::vector<PpuResult> sorted = detections;
    std::sort(sorted.begin(), sorted.end(), [](const PpuResult& a, const PpuResult& b) {
        return a.confidence > b.confidence;
    });

    std::vector<bool> suppressed(sorted.size(), false);
    std::vector<PpuResult> result;

    for (size_t i = 0; i < sorted.size(); ++i) {
        if (suppressed[i]) continue;
        result.push_back(sorted[i]);

        for (size_t j = i + 1; j < sorted.size(); ++j) {
            if (!suppressed[j] && sorted[i].iou(sorted[j]) > m_nmsThreshold) {
                suppressed[j] = true;
            }
        }
    }
    return result;
}
