#include "YoloFacePostProcess.h"
#include <cmath>
#include <algorithm>
#include <dxrt/datatype.h>
#include <QDebug>


static float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

float YoloFacePostProcess::FaceResult::iou(const FaceResult& other) const {
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

YoloFacePostProcess::YoloFacePostProcess(int inputW, int inputH, float confThresh, float nmsThresh)
    : m_inputW(inputW), m_inputH(inputH), m_confThresh(confThresh), m_nmsThresh(nmsThresh) {
    m_anchors = {
        {8,  {{4, 5}, {8, 10}, {13, 16}}},
        {16, {{23, 29}, {43, 55}, {73, 105}}},
        {32, {{146, 217}, {231, 300}, {335, 433}}}
    };
}

std::vector<DetectionBox> YoloFacePostProcess::postprocess(const dxrt::TensorPtrs& outputs, int imgW, int imgH) {
    if (outputs.empty()) return {};

    auto output = outputs[0];
    auto shape = output->shape();
    
    if (shape.size() != 3 || shape[1] != 25200 || shape[2] != 16) {
        qWarning() << "YoloFacePostProcess: Unexpected shape " << shape[1] << "x" << shape[2];
    }

    const float* data = static_cast<const float*>(output->data());
    
    auto results = decode(data, shape[1]);
    auto nmsResults = nms(results);

    std::vector<DetectionBox> finalDets;
    for (const auto& res : nmsResults) {
        DetectionBox box;
        box.classId = 0;
        box.label = "face";
        box.confidence = res.confidence;
        
        float x1 = res.x - res.w / 2.0f;
        float y1 = res.y - res.h / 2.0f;
        
        box.box.x = std::max(0.0f, x1 / m_inputW);
        box.box.y = std::max(0.0f, y1 / m_inputH);
        box.box.width = res.w / m_inputW;
        box.box.height = res.h / m_inputH;
        
        box.color = cv::Scalar(0, 255, 0); // Green for face
        
        for (size_t k = 0; k < res.landmarks.size(); k += 2) {
            float lx = res.landmarks[k];
            float ly = res.landmarks[k+1];
            box.landmarks.push_back(lx / m_inputW);
            box.landmarks.push_back(ly / m_inputH);
            box.landmarks.push_back(1.0f);
        }
        
        finalDets.push_back(box);
    }
    
    return finalDets;
}

std::vector<YoloFacePostProcess::FaceResult> YoloFacePostProcess::decode(const float* data, int numBoxes) {
    std::vector<FaceResult> detections;
    int strides[] = {8, 16, 32};
    int gridSizes[] = {80, 40, 20}; // For 640 input
    
    int ptr = 0;
    
    for (int i = 0; i < 3; ++i) { // Per Stride
        int stride = strides[i];
        int grid = gridSizes[i];
        const auto& anchors = m_anchors[stride];
        int numAnchors = anchors.size(); // 3
        
        // Total cells per stride = grid*grid * numAnchors
        int numCells = grid * grid * numAnchors;
        
        // Iterate cells
        for (int a = 0; a < numAnchors; ++a) {
            for (int y = 0; y < grid; ++y) {
                for (int x = 0; x < grid; ++x) {
                    
                    float objConf = sigmoid(data[ptr + 4]);
                    
                    if (objConf > m_confThresh) {
                        float bx = (sigmoid(data[ptr + 0]) * 2.0f - 0.5f + x) * stride;
                        float by = (sigmoid(data[ptr + 1]) * 2.0f - 0.5f + y) * stride;
                        float bw = pow(sigmoid(data[ptr + 2]) * 2.0f, 2) * anchors[a].first;
                        float bh = pow(sigmoid(data[ptr + 3]) * 2.0f, 2) * anchors[a].second;
                        
                        std::vector<float> lms;
                        for (int k = 0; k < 5; ++k) {
                            float lx = data[ptr + 5 + k*2];
                            float ly = data[ptr + 6 + k*2];
                            lx = lx * anchors[a].first + x * stride;
                             lx = (lx * 2.0f - 0.5f + x) * stride;
                             ly = (ly * 2.0f - 0.5f + y) * stride;
                            
                             lms.push_back(lx);
                             lms.push_back(ly);
                        }
                        
                         // Class conf (index 15)
                         float clsConf = sigmoid(data[ptr + 15]);
                         float finalScore = objConf * clsConf;
                         
                         if (finalScore > m_confThresh) {
                             FaceResult res;
                             res.x = bx;
                             res.y = by;
                             res.w = bw;
                             res.h = bh;
                             res.confidence = finalScore;
                             res.landmarks = lms;
                             detections.push_back(res);
                         }
                    }
                    ptr += 16;
                }
            }
        }
    }
    return detections;
}

std::vector<YoloFacePostProcess::FaceResult> YoloFacePostProcess::nms(const std::vector<FaceResult>& detections) {
     if (detections.empty()) return {};

    std::vector<FaceResult> sorted = detections;
    std::sort(sorted.begin(), sorted.end(), [](const FaceResult& a, const FaceResult& b) {
        return a.confidence > b.confidence;
    });

    std::vector<bool> suppressed(sorted.size(), false);
    std::vector<FaceResult> result;

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
