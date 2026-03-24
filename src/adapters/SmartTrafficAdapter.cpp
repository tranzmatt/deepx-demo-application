#include "SmartTrafficAdapter.h"
#include <QJsonArray>
#include <QDebug>
#include <cmath>

SmartTrafficAdapter::SmartTrafficAdapter(QObject* parent) 
    : YoloAsyncAdapter(parent)
    , m_nextId(1)
    , m_maxMissingFrames(15) 
    , m_maxMatchDistance(0.1f) 
    , m_count(0)
{
    // Default line (can be overridden by config)
    m_line = {cv::Point2f(0.0f, 0.5f), cv::Point2f(1.0f, 0.5f)};
}

bool SmartTrafficAdapter::loadConfig(const QJsonObject& config) {
    if (!YoloAsyncAdapter::loadConfig(config)) return false;

    if (config.contains("line")) {
         QJsonArray lineArr = config["line"].toArray();
         if (lineArr.size() >= 4) {
             m_line.first = cv::Point2f(lineArr[0].toDouble(), lineArr[1].toDouble());
             m_line.second = cv::Point2f(lineArr[2].toDouble(), lineArr[3].toDouble());
             qDebug() << "SmartTraffic: Line set to" << m_line.first.x << m_line.first.y << "->" << m_line.second.x << m_line.second.y;
         }
    } else if (config["application"].toObject().contains("line")) {
         QJsonArray lineArr = config["application"].toObject()["line"].toArray();
         if (lineArr.size() >= 4) {
             m_line.first = cv::Point2f(lineArr[0].toDouble(), lineArr[1].toDouble());
             m_line.second = cv::Point2f(lineArr[2].toDouble(), lineArr[3].toDouble());
         }
    }

    return true;
}

float SmartTrafficAdapter::calculateDistance(const cv::Point2f& p1, const cv::Point2f& p2) {
    float dx = p1.x - p2.x;
    float dy = p1.y - p2.y;
    return std::sqrt(dx*dx + dy*dy);
}

float SmartTrafficAdapter::crossProduct(const cv::Point2f& a, const cv::Point2f& b, const cv::Point2f& c) {
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

bool SmartTrafficAdapter::checkLineCrossing(const cv::Point2f& p1, const cv::Point2f& p2) {
    cv::Point2f q1 = m_line.first;
    cv::Point2f q2 = m_line.second;
    
    float cp1 = crossProduct(q1, q2, p1);
    float cp2 = crossProduct(q1, q2, p2);
    
    if (((cp1 > 0 && cp2 < 0) || (cp1 < 0 && cp2 > 0))) {
        float cp3 = crossProduct(p1, p2, q1);
        float cp4 = crossProduct(p1, p2, q2);
        
        if (((cp3 > 0 && cp4 < 0) || (cp3 < 0 && cp4 > 0))) {
            return true;
        }
    }
    return false;
}

void SmartTrafficAdapter::onResultReady(DetectionResult& result) {
    struct DetInfo {
        size_t index;
        cv::Point2f centroid;
    };
    std::vector<DetInfo> vehicles;
    
    for (size_t i = 0; i < result.detections.size(); ++i) {
        const auto& d = result.detections[i];
        if (d.label == "car" || d.label == "truck" || d.label == "bus" || d.label == "motorcycle") {
             float cx = d.box.x + d.box.width/2.0f;
             float cy = d.box.y + d.box.height/2.0f;
             vehicles.push_back({i, cv::Point2f(cx, cy)});
        }
    }
    
    for (auto& pair : m_tracks) pair.second.missingFrames++;
    
    std::vector<bool> matched(vehicles.size(), false);
    
    for (auto& pair : m_tracks) {
        TrafficObject& track = pair.second;
        int bestIdx = -1;
        float bestDist = m_maxMatchDistance;
        
        for (size_t i = 0; i < vehicles.size(); ++i) {
            if (matched[i]) continue;
            float d = calculateDistance(track.centroid, vehicles[i].centroid);
            if (d < bestDist) {
                bestDist = d;
                bestIdx = i;
            }
        }
        
        if (bestIdx != -1) {
            matched[bestIdx] = true;
            track.missingFrames = 0;
            
            if (!track.crossed) {
                if (checkLineCrossing(track.centroid, vehicles[bestIdx].centroid)) {
                    m_count++;
                    track.crossed = true;
                }
            }
            
            track.centroid = vehicles[bestIdx].centroid;
            
            result.detections[vehicles[bestIdx].index].label += " #" + std::to_string(track.id);
        }
    }
    
    for (size_t i = 0; i < vehicles.size(); ++i) {
        if (!matched[i]) {
            int newId = m_nextId++;
            m_tracks[newId] = {newId, vehicles[i].centroid, 0, false};
            result.detections[vehicles[i].index].label += " #" + std::to_string(newId);
        }
    }
    
    for (auto it = m_tracks.begin(); it != m_tracks.end(); ) {
        if (it->second.missingFrames > m_maxMissingFrames) {
            it = m_tracks.erase(it);
        } else {
            ++it;
        }
    }
    
    result.overlayPolygon.clear();
    result.overlayPolygon.push_back(m_line.first);
    result.overlayPolygon.push_back(m_line.second);
    
    result.overlayText = QString("Count: %1").arg(m_count);
}
