#ifndef COMMON_H
#define COMMON_H

#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <QString>
#include <vector>

struct DetectionBox {
    int classId;
    std::string label;
    float confidence;
    cv::Rect2f box; // Normalized coordinates [0,1]
    cv::Scalar color; 
    std::vector<float> landmarks; // x, y, score order
    std::vector<cv::Point2f> overlayPolygon; // Normalized [0,1]
};

struct DetectionResult {
    std::vector<DetectionBox> detections;
    long long timestamp;
    double inferenceTime;
    
    // Overlay logic
    std::vector<cv::Point2f> overlayPolygon; // Normalized [0,1]
    bool alertTriggered = false;
    QString alertMessage;
    QString overlayText;
};

#endif // COMMON_H
