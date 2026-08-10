#include "YoloPosePostProcess.h"
#include <cmath>
#include <algorithm>
#include <dxrt/dxrt_cxx_api.h>

float YoloPosePostProcess::PoseResult::iou(const PoseResult& other) const {
    float x_left = std::max(box[0], other.box[0]);
    float y_top = std::max(box[1], other.box[1]);
    float x_right = std::min(box[2], other.box[2]);
    float y_bottom = std::min(box[3], other.box[3]);

    if (x_right < x_left || y_bottom < y_top) return 0.0f;

    float intersection_area = (x_right - x_left) * (y_bottom - y_top);
    float union_area = area() + other.area() - intersection_area;
    return intersection_area / union_area;
}

YoloPosePostProcess::YoloPosePostProcess(int inputW, int inputH, float scoreThresh, float nmsThresh)
    : m_inputWidth(inputW), m_inputHeight(inputH), m_scoreThresh(scoreThresh), m_nmsThresh(nmsThresh) {
    
    m_anchors = {
        {8, {{19, 27}, {44, 40}, {38, 94}}},
        {16, {{96, 68}, {86, 152}, {180, 137}}},
        {32, {{140, 301}, {303, 264}, {238, 542}}},
        {64, {{436, 615}, {739, 380}, {925, 792}}}
    };
}

std::vector<DetectionBox> YoloPosePostProcess::postprocess(const dxrt::TensorPtrs& outputs, int originalW, int originalH) {
    if (outputs.empty() || outputs[0]->type() != dxrt::DataType::POSE) {
        return {}; 
    }

    auto results = decode(outputs);
    results = nms(results);

    std::vector<DetectionBox> detections;
    for (const auto& res : results) {
        DetectionBox det;
        det.confidence = res.confidence;
        det.classId = 0;
        det.label = "Person";
        det.color = cv::Scalar(0, 255, 0); // Green
        
        float w = res.box[2] - res.box[0];
        float h = res.box[3] - res.box[1];
        float x = res.box[0];
        float y = res.box[1];

        det.box.x = x / m_inputWidth;
        det.box.y = y / m_inputHeight;
        det.box.width = w / m_inputWidth;
        det.box.height = h / m_inputHeight;

        det.landmarks = res.landmarks;
        for (size_t i = 0; i < det.landmarks.size(); i += 3) {
            det.landmarks[i] /= m_inputWidth;
            det.landmarks[i+1] /= m_inputHeight;
        }

        detections.push_back(det);
    }
    return detections;
}

std::vector<YoloPosePostProcess::PoseResult> YoloPosePostProcess::decode(const dxrt::TensorPtrs& outputs) {
    std::vector<PoseResult> detections;
    auto num_elements = outputs[0]->shape()[1];
    auto* raw_data = static_cast<dxrt::DevicePose_t*>(outputs[0]->data());

    for (int i = 0; i < num_elements; i++) {
        auto pose_data = raw_data[i];
        if (pose_data.score < m_scoreThresh) continue;

        PoseResult result;
        int gX = pose_data.grid_x;
        int gY = pose_data.grid_y;

        if (m_anchors.find(0) == m_anchors.end()) { 
        }
        
        auto it = m_anchors.begin();
        std::advance(it, pose_data.layer_idx);
        int stride = it->first;
        const auto& anchors = it->second;

        float x = (pose_data.x * 2. - 0.5 + gX) * stride;
        float y = (pose_data.y * 2. - 0.5 + gY) * stride;
        float w = (pose_data.w * pose_data.w * 4.) * anchors[pose_data.box_idx].first;
        float h = (pose_data.h * pose_data.h * 4.) * anchors[pose_data.box_idx].second;

        result.confidence = pose_data.score;
        result.box = {x - w / 2, y - h / 2, x + w / 2, y + h / 2};

        for (int kpt = 0; kpt < 17; ++kpt) {
            float lx = (pose_data.grid_x - 0.5 + pose_data.kpts[kpt][0] * 2) * stride;
            float ly = (pose_data.grid_y - 0.5 + pose_data.kpts[kpt][1] * 2) * stride;
            float ls = pose_data.kpts[kpt][2];
            
            result.landmarks.push_back(lx);
            result.landmarks.push_back(ly);
            result.landmarks.push_back(ls);
        }
        detections.push_back(result);
    }
    return detections;
}

std::vector<YoloPosePostProcess::PoseResult> YoloPosePostProcess::nms(const std::vector<PoseResult>& dets) {
    if (dets.empty()) return {};

    std::vector<PoseResult> sorted = dets;
    std::sort(sorted.begin(), sorted.end(), [](const PoseResult& a, const PoseResult& b) {
        return a.confidence > b.confidence;
    });

    std::vector<bool> suppressed(sorted.size(), false);
    std::vector<PoseResult> result;

    for (size_t i = 0; i < sorted.size(); ++i) {
        if (suppressed[i]) continue;
        result.push_back(sorted[i]);

        for (size_t j = i + 1; j < sorted.size(); ++j) {
            if (!suppressed[j] && sorted[i].iou(sorted[j]) > m_nmsThresh) {
                suppressed[j] = true;
            }
        }
    }
    return result;
}
