#include "YoloV8PpuPostProcess.h"
#include <dxrt/dxrt_cxx_api.h>
#include <iostream>
#include <algorithm>
#include <cmath>

YoloV8PpuPostProcess::YoloV8PpuPostProcess(int inputW, int inputH, float scoreThresh, float nmsThresh)
    : m_inputWidth(inputW), m_inputHeight(inputH), m_scoreThresh(scoreThresh), m_nmsThresh(nmsThresh) {}

std::vector<DetectionBox> YoloV8PpuPostProcess::postprocess(const dxrt::TensorPtrs& outputs, int originalW, int originalH) {
    if (outputs.empty() || outputs[0]->type() != dxrt::DataType::BBOX) {
        return {}; 
    }

    std::vector<DetectionBox> detections;
    auto numElements = outputs[0]->shape()[1];
    auto* rawData = static_cast<dxrt::DeviceBoundingBox_t*>(outputs[0]->data());
    
    for (size_t i = 0; i < numElements; i++) {
        auto& bbox = rawData[i];
        
        if (bbox.score < m_scoreThresh) continue;

        DetectionBox det;
        det.classId = static_cast<int>(bbox.label);
        
        det.confidence = bbox.score;
        
        // YOLOv8 PPU output is raw pixel coords [x, y, w, h] (center format)
        // Convert to Normalized [0, 1] relative to Input Size
        
        // Input: x (center), y (center), w, h
        float x = bbox.x;
        float y = bbox.y;
        float w = bbox.w;
        float h = bbox.h;
        
        // Convert to Top-Left
        float x1 = x - w / 2.0f;
        float y1 = y - h / 2.0f;
        
        det.box.x = x1 / m_inputWidth;
        det.box.y = y1 / m_inputHeight;
        det.box.width = w / m_inputWidth;
        det.box.height = h / m_inputHeight;
        
        // Assign color based on classId (simple hash)
        int r = (det.classId * 50) % 255;
        int g = (det.classId * 80 + 50) % 255;
        int b = (det.classId * 120 + 100) % 255;
        det.color = cv::Scalar(b, g, r); // BGR

        detections.push_back(det);
    }

    return applyNMS(detections);
}

std::vector<DetectionBox> YoloV8PpuPostProcess::applyNMS(std::vector<DetectionBox>& detections) {
    if (detections.empty()) return {};
    
    std::vector<DetectionBox> sorted = detections;
    std::sort(sorted.begin(), sorted.end(), [](const DetectionBox& a, const DetectionBox& b) {
        return a.confidence > b.confidence;
    });
    
    std::vector<bool> suppressed(sorted.size(), false);
    std::vector<DetectionBox> result;
    
    for (size_t i = 0; i < sorted.size(); ++i) {
        if (suppressed[i]) continue;
        result.push_back(sorted[i]);
        
        for (size_t j = i + 1; j < sorted.size(); ++j) {
            if (suppressed[j]) continue;
            if (sorted[i].classId != sorted[j].classId) continue; // Only suppress same class
            
            // IoU Calculation
            float x1 = std::max(sorted[i].box.x, sorted[j].box.x);
            float y1 = std::max(sorted[i].box.y, sorted[j].box.y);
            float x2 = std::min(sorted[i].box.x + sorted[i].box.width, sorted[j].box.x + sorted[j].box.width);
            float y2 = std::min(sorted[i].box.y + sorted[i].box.height, sorted[j].box.y + sorted[j].box.height);
            
            float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
            float area1 = sorted[i].box.width * sorted[i].box.height;
            float area2 = sorted[j].box.width * sorted[j].box.height;
            
            float iou = intersection / (area1 + area2 - intersection + 1e-6f);
            
            if (iou > m_nmsThresh) {
                suppressed[j] = true;
            }
        }
    }
    return result;
}
