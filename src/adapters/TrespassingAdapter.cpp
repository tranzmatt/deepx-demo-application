#include "TrespassingAdapter.h"
#include <QJsonArray>
#include <QDebug>
#include <opencv2/imgproc.hpp>

TrespassingAdapter::TrespassingAdapter(QObject* parent) : YoloAsyncAdapter(parent) {}

bool TrespassingAdapter::loadConfig(const QJsonObject& config) {
    if (!YoloAsyncAdapter::loadConfig(config)) return false;
    
    m_polygon.clear();
    QJsonArray pointsArr = config["points"].toArray();
    if (pointsArr.isEmpty()) {
        qWarning() << "No points found in config for TrespassingAdapter";
    } else {
        auto appObj = config["application"].toObject();
        QJsonArray res = appObj["resolution"].toArray();
        float refW = res.isEmpty() ? 960.0f : res[0].toDouble();
        float refH = res.isEmpty() ? 540.0f : res[1].toDouble();

        for (const auto& ptVal : pointsArr) {
            QJsonArray coords = ptVal.toArray();
            if (coords.size() >= 2) {
                float x = coords[0].toDouble() / refW;
                float y = coords[1].toDouble() / refH;
                m_polygon.push_back(cv::Point2f(x, y));
            }
        }
    }
    
    return true;
}

void TrespassingAdapter::onResultReady(DetectionResult& result) {
    // 1. Copy normalized polygon to result for rendering
    result.overlayPolygon.assign(m_polygon.begin(), m_polygon.end());
    
    // 2. Check for Trespassing
    bool alarm = false;
    
    for (const auto& box : result.detections) {
        if (box.label == "person") {
            // Bottom Center
            float bx = box.box.x + box.box.width / 2.0f;
            float by = box.box.y + box.box.height; 
            
            if (cv::pointPolygonTest(m_polygon, cv::Point2f(bx, by), false) >= 0) {
                alarm = true;
                break;
            }
        }
    }
    
    if (alarm) {
        result.alertTriggered = true;
        result.alertMessage = "ALEART: TRESPASSING DETECTED!";
    }
}
