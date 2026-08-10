#include "YoloHandPosePostProcess.h"
#include <cmath>
#include <algorithm>
#include <dxrt/dxrt_cxx_api.h>
#include <iostream>
#include <QDebug>

float YoloHandPosePostProcess::HandResult::iou(const HandResult& other) const {
    float x_left = std::max(box[0], other.box[0]);
    float y_top = std::max(box[1], other.box[1]);
    float x_right = std::min(box[2], other.box[2]);
    float y_bottom = std::min(box[3], other.box[3]);

    if (x_right < x_left || y_bottom < y_top) return 0.0f;

    float intersection_area = (x_right - x_left) * (y_bottom - y_top);
    float union_area = area() + other.area() - intersection_area;
    return intersection_area / union_area;
}

YoloHandPosePostProcess::YoloHandPosePostProcess(int inputW, int inputH, float scoreThresh, float nmsThresh, int kptCount)
    : m_inputWidth(inputW), m_inputHeight(inputH), m_scoreThresh(scoreThresh), m_nmsThresh(nmsThresh), m_kptCount(kptCount) {
    
    // Default Anchors for YOLOv5 Pose PPU (only used in POSE decode path)
    m_anchors = {
        {8, {{19, 27}, {44, 40}, {38, 94}}},
        {16, {{96, 68}, {86, 152}, {180, 137}}},
        {32, {{140, 301}, {303, 264}, {238, 542}}},
        {64, {{436, 615}, {739, 380}, {925, 792}}}
    };
}

void YoloHandPosePostProcess::setAnchors(const std::map<int, std::vector<std::pair<int, int>>>& anchors) {
    m_anchors = anchors;
}

std::vector<DetectionBox> YoloHandPosePostProcess::postprocess(const dxrt::TensorPtrs& outputs, int originalW, int originalH) {
    if (outputs.empty()) return {};

    // Debug: print output info
    auto& tensor = outputs[0];
    auto shape = tensor->shape();
    QString shapeStr;
    for (auto d : shape) shapeStr += QString::number(d) + " x ";
    if (!shapeStr.isEmpty()) shapeStr.chop(3);

    std::vector<HandResult> results;

    if (outputs[0]->type() == dxrt::DataType::POSE) {
        // PPU POSE output — needs byte-level parsing for 21 keypoints
        // DevicePose_t has kpts[17][3], struct misalignment for 21 kpts!
        results = decodePose(outputs);
    } else {
        // Raw float tensor — YOLOv8 Pose format
        results = decodeFloat(outputs);
    }

    results = nms(results);

    std::vector<DetectionBox> detections;
    for (const auto& res : results) {
        DetectionBox det;
        det.confidence = res.confidence;
        det.classId = 0;
        det.label = "hand";
        det.color = cv::Scalar(0, 255, 255); // Yellow (BGR)

        float w = res.box[2] - res.box[0];
        float h = res.box[3] - res.box[1];

        det.box.x = res.box[0] / m_inputWidth;
        det.box.y = res.box[1] / m_inputHeight;
        det.box.width = w / m_inputWidth;
        det.box.height = h / m_inputHeight;

        // Normalize landmarks to [0,1]
        det.landmarks = res.landmarks;
        for (size_t i = 0; i < det.landmarks.size(); i += 3) {
            det.landmarks[i] /= m_inputWidth;
            det.landmarks[i+1] /= m_inputHeight;
            // score (i+2) stays same
        }

        detections.push_back(det);
    }

    return detections;
}

std::vector<YoloHandPosePostProcess::HandResult> YoloHandPosePostProcess::decodePose(const dxrt::TensorPtrs& outputs) {
    std::vector<HandResult> detections;
    auto num_elements = outputs[0]->shape()[1];
    auto* raw_bytes = static_cast<uint8_t*>(outputs[0]->data());

    // DevicePose_t struct layout (byte offsets):
    //   0: float x       (4)
    //   4: float y       (4)
    //   8: float w       (4)
    //  12: float h       (4)
    //  16: uint8 grid_y, grid_x, box_idx, layer_idx (4)
    //  20: float score   (4)
    //  24: uint32 label  (4)
    //  28: float kpts[N][3]  (N*12 bytes)
    //
    // For 17 kpts: 28 + 204 + 24 padding = 256 bytes (DevicePose_t)
    // For 21 kpts: 28 + 252 = 280 bytes + unknown padding
    //
    int headerSize = 28; // bytes before kpts
    int kptsSize = m_kptCount * 3 * sizeof(float);
    int rawStructSize = headerSize + kptsSize;

    // Align stride to nearest multiple of 32 (PPU alignment)
    int stride = ((rawStructSize + 31) / 32) * 32;

    // For 17 kpts: rawStructSize=232, stride=256 (matches DevicePose_t)
    // For 21 kpts: rawStructSize=280, stride=288

    for (int i = 0; i < num_elements; i++) {
        uint8_t* elem = raw_bytes + i * stride;

        float x     = *reinterpret_cast<float*>(elem + 0);
        float y     = *reinterpret_cast<float*>(elem + 4);
        float w     = *reinterpret_cast<float*>(elem + 8);
        float h     = *reinterpret_cast<float*>(elem + 12);
        uint8_t gY  = *(elem + 16);
        uint8_t gX  = *(elem + 17);
        uint8_t bIdx = *(elem + 18);
        uint8_t lIdx = *(elem + 19);
        float score = *reinterpret_cast<float*>(elem + 20);

        if (score < m_scoreThresh) continue;

        auto it = m_anchors.begin();
        std::advance(it, lIdx);
        int anchorStride = it->first;
        const auto& anchors = it->second;

        float cx = (x * 2. - 0.5 + gX) * anchorStride;
        float cy = (y * 2. - 0.5 + gY) * anchorStride;
        float bw = (w * w * 4.) * anchors[bIdx].first;
        float bh = (h * h * 4.) * anchors[bIdx].second;

        HandResult result;
        result.confidence = score;
        result.box = {cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2};

        float* kpts = reinterpret_cast<float*>(elem + headerSize);
        for (int kpt = 0; kpt < m_kptCount; ++kpt) {
            float lx = (gX - 0.5 + kpts[kpt * 3 + 0] * 2) * anchorStride;
            float ly = (gY - 0.5 + kpts[kpt * 3 + 1] * 2) * anchorStride;
            float ls = kpts[kpt * 3 + 2];

            result.landmarks.push_back(lx);
            result.landmarks.push_back(ly);
            result.landmarks.push_back(ls);
        }
        detections.push_back(result);
    }
    return detections;
}

std::vector<YoloHandPosePostProcess::HandResult> YoloHandPosePostProcess::decodeFloat(const dxrt::TensorPtrs& outputs) {
    std::vector<HandResult> detections;

    auto& tensor = outputs[0];
    auto shape = tensor->shape();
    float* data = static_cast<float*>(tensor->data());

    if (!data || shape.size() < 2) return detections;

    // YOLOv8 Pose output format: (1, channels, num_candidates) or (channels, num_candidates)
    // channels = 5 + kptCount * 3  (e.g. 68 for 21 keypoints)
    // num_candidates = e.g. 8400
    //
    // Auto-detect orientation: check which dimension matches expected channels
    int expectedChannels = 5 + m_kptCount * 3;
    int channels, numCandidates;
    bool transposed; // true = (1, C, N), false = (1, N, C)

    if (shape.size() == 3) {
        if (shape[1] == expectedChannels || (shape[1] < shape[2] && shape[1] >= 5)) {
            // (1, C, N) — transposed/channel-first (standard YOLOv8 output)
            channels = shape[1];
            numCandidates = shape[2];
            transposed = true;
        } else {
            // (1, N, C) — row-major
            numCandidates = shape[1];
            channels = shape[2];
            transposed = false;
        }
    } else if (shape.size() == 2) {
        if (shape[0] == expectedChannels || (shape[0] < shape[1] && shape[0] >= 5)) {
            channels = shape[0];
            numCandidates = shape[1];
            transposed = true;
        } else {
            numCandidates = shape[0];
            channels = shape[1];
            transposed = false;
        }
    } else {
        return detections;
    }

    if (channels < 5) {
        return detections;
    }

    int kptsAvailable = std::min(m_kptCount, (channels - 5) / 3);

    for (int i = 0; i < numCandidates; i++) {
        // Access data based on layout
        // transposed (C, N): element(c, i) = data[c * N + i]
        // row-major  (N, C): element(c, i) = data[i * C + c]
        auto at = [&](int c, int idx) -> float {
            if (transposed) return data[c * numCandidates + idx];
            else return data[idx * channels + c];
        };

        float score = at(4, i);
        if (score < m_scoreThresh) continue;

        float cx = at(0, i);
        float cy = at(1, i);
        float w  = at(2, i);
        float h  = at(3, i);

        HandResult result;
        result.confidence = score;
        result.box = {cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2};

        // Parse keypoints
        for (int kpt = 0; kpt < kptsAvailable; ++kpt) {
            int chBase = 5 + kpt * 3;
            float lx   = at(chBase + 0, i);
            float ly   = at(chBase + 1, i);
            float conf = at(chBase + 2, i);

            result.landmarks.push_back(lx);
            result.landmarks.push_back(ly);
            result.landmarks.push_back(conf);
        }

        detections.push_back(result);
    }

    return detections;
}

std::vector<YoloHandPosePostProcess::HandResult> YoloHandPosePostProcess::nms(const std::vector<HandResult>& dets) {
    if (dets.empty()) return {};

    std::vector<HandResult> sorted = dets;
    std::sort(sorted.begin(), sorted.end(), [](const HandResult& a, const HandResult& b) {
        return a.confidence > b.confidence;
    });

    std::vector<bool> suppressed(sorted.size(), false);
    std::vector<HandResult> result;

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
