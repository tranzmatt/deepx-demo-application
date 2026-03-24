#include "OcrPostProcess.h"
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <QDebug>

OcrPostProcess::OcrPostProcess() {
    // Default char dict (can be overridden)
    m_characterDict.push_back("blank"); // Placeholder for CTC blank
}

OcrPostProcess::~OcrPostProcess() {}

bool OcrPostProcess::loadCharDict(const std::string& dictPath) {
    m_characterDict.clear();
    m_characterDict.push_back("blank");
    
    std::ifstream file(dictPath);
    if (!file.is_open()) {
        qWarning() << "Failed to open character dict: " << QString::fromStdString(dictPath);
        return false;
    }
    qDebug() << "Loading char dict from:" << QString::fromStdString(dictPath);
    
    std::string line;
    while (std::getline(file, line)) {
        // Remove CR/LF
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (!line.empty() && line.back() == '\n') line.pop_back();
        m_characterDict.push_back(line);
    }
    
    m_characterDict.push_back(" ");
    return true;
}

void OcrPostProcess::orderPoints(std::vector<cv::Point>& pts) {
    if (pts.size() != 4) return;
    
    std::vector<cv::Point> rect(4);
    std::vector<float> s(4);
    std::vector<float> diff(4);
    
    for (int i = 0; i < 4; ++i) {
        s[i] = (float)pts[i].x + (float)pts[i].y;
        diff[i] = (float)pts[i].y - (float)pts[i].x;
    }
    
    int argmin_s = (int)std::distance(s.begin(), std::min_element(s.begin(), s.end()));
    int argmax_s = (int)std::distance(s.begin(), std::max_element(s.begin(), s.end()));
    int argmin_diff = (int)std::distance(diff.begin(), std::min_element(diff.begin(), diff.end()));
    int argmax_diff = (int)std::distance(diff.begin(), std::max_element(diff.begin(), diff.end()));
    
    rect[0] = pts[argmin_s];
    rect[1] = pts[argmin_diff];
    rect[2] = pts[argmax_s];
    rect[3] = pts[argmax_diff];
    
    pts = rect;
}

std::vector<cv::Point> OcrPostProcess::unclip(const std::vector<cv::Point>& box, float unclipRatio) {
    // Approximate unclip using RotatedRect inflation
    // 1. Get RotatedRect
    cv::RotatedRect rect = cv::minAreaRect(box);
    
    // 2. Calculate distance
    // poly.area * unclip_ratio / poly.length
    float area = rect.size.width * rect.size.height;
    float length = 2 * (rect.size.width + rect.size.height);
    float distance = area * unclipRatio / length;
    
    // 3. Inflate
    rect.size.width += 2 * distance;
    rect.size.height += 2 * distance;
    
    // 4. Get points back
    cv::Point2f points[4];
    rect.points(points);
    
    std::vector<cv::Point> expandedBox;
    for (int i = 0; i < 4; ++i) {
        expandedBox.push_back(points[i]);
    }
    return expandedBox;
}

float OcrPostProcess::getBoxScore(const std::vector<cv::Point>& box, const cv::Mat& pred) {
    int width = pred.cols;
    int height = pred.rows;
    
    cv::Rect boundingRect = cv::boundingRect(box);
    int xmin = std::max(0, boundingRect.x);
    int xmax = std::min(width - 1, boundingRect.x + boundingRect.width);
    int ymin = std::max(0, boundingRect.y);
    int ymax = std::min(height - 1, boundingRect.y + boundingRect.height);

    if (xmax <= xmin || ymax <= ymin) return 0.0f;

    cv::Mat mask = cv::Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_8UC1);
    std::vector<cv::Point> shiftedBox = box;
    for (auto& pt : shiftedBox) {
        pt.x -= xmin;
        pt.y -= ymin;
    }
    
    // Fill polygon
    std::vector<std::vector<cv::Point>> contours = {shiftedBox};
    cv::fillPoly(mask, contours, cv::Scalar(1));
    
    cv::Scalar mean = cv::mean(pred(cv::Range(ymin, ymax + 1), cv::Range(xmin, xmax + 1)), mask);
    return (float)mean[0];
}

std::vector<OcrBox> OcrPostProcess::postProcessDet(const float* mapData, int mapW, int mapH, int inputW, int inputH) {
    std::vector<OcrBox> results;
    
    // 1. Create bitmap from threshold
    cv::Mat pred(mapH, mapW, CV_32F, (void*)mapData);
    cv::Mat bitmap;
    // Simple thresholding
    cv::threshold(pred, bitmap, m_detThresh, 1.0, cv::THRESH_BINARY);
    bitmap.convertTo(bitmap, CV_8U); // Convert to uint8 for findContours
    
    // 2. Find Contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bitmap, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    
    for (const auto& contour : contours) {
        // 3. Filter small boxes (dimensions)
        cv::RotatedRect rect = cv::minAreaRect(contour);
        if (std::min(rect.size.width, rect.size.height) < 3) continue;
        
        // 4. Box Score
        cv::Point2f rectPoints[4];
        rect.points(rectPoints);
        std::vector<cv::Point> boxPoints;
        for(int i=0; i<4; ++i) boxPoints.push_back(rectPoints[i]);
        orderPoints(boxPoints);
        
        float score = getBoxScore(boxPoints, pred);
        if (score < m_detBoxThresh) continue;
        
        // 5. Unclip
        std::vector<cv::Point> unclippedBox = unclip(boxPoints, m_unclipRatio);
        
        // 6. Check size again
         cv::RotatedRect unclippedRect = cv::minAreaRect(unclippedBox);
         if (std::min(unclippedRect.size.width, unclippedRect.size.height) < 3 + 2) continue;

        // 7. Scale back to original image size relative to the map
        for (auto& pt : unclippedBox) {
            pt.x = std::min(std::max((int)round(pt.x * inputW / mapW), 0), inputW);
            pt.y = std::min(std::max((int)round(pt.y * inputH / mapH), 0), inputH);
        }
        
        results.push_back({unclippedBox, score});
    }
    
    return results;
}

std::pair<std::string, float> OcrPostProcess::postProcessRec(const float* outputData, int seqLen, int numClasses) {
    std::string text = "";
    std::vector<float> confidences;
    
    int lastIndex = -1;
    
    for (int t = 0; t < seqLen; ++t) {
        // Find max element for this timestep
        int maxIdx = 0;
        float maxVal = -1e9f;
        const float* stepData = outputData + (size_t)t * (size_t)numClasses;
        
        for (int c = 0; c < numClasses; ++c) {
            if (stepData[c] > maxVal) {
                maxVal = stepData[c];
                maxIdx = c;
            }
        }
        
        if (maxIdx != 0 && maxIdx != lastIndex) {
            if (maxIdx < (int)m_characterDict.size()) {
                text += m_characterDict[maxIdx];
                confidences.push_back(maxVal); 
            }
        }
        lastIndex = maxIdx;
    }
    
    float avgConf = 0.0f;
    if (!confidences.empty()) {
        for (float c : confidences) avgConf += c;
        avgConf /= confidences.size();
    }
    
    return {text, avgConf};
}

std::pair<int, float> OcrPostProcess::postProcessCls(const float* outputData, int numClasses) {
    int maxIdx = 0;
    float maxVal = -1e9f;
    for (int c = 0; c < numClasses; ++c) {
        if (outputData[c] > maxVal) {
            maxVal = outputData[c];
            maxIdx = c;
        }
    }
    return {maxIdx, maxVal};
}
