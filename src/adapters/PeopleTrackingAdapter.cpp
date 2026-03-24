#include "PeopleTrackingAdapter.h"
#include <QDebug>
#include <cmath>
#include <limits>
#include <algorithm>

PeopleTrackingAdapter::PeopleTrackingAdapter(QObject* parent) 
    : YoloAdapter(parent)
    , m_nextId(1)
    , m_maxMissingFrames(10) 
    , m_maxDistance(0.1f)
{
    qDebug() << "[PeopleTrackingAdapter] Initialized";
}

bool PeopleTrackingAdapter::loadConfig(const QJsonObject& config) {
    return YoloAdapter::loadConfig(config);
}

float PeopleTrackingAdapter::calculateDistance(const cv::Point2f& p1, const cv::Point2f& p2) {
    float dx = p1.x - p2.x;
    float dy = p1.y - p2.y;
    return std::sqrt(dx*dx + dy*dy);
}

void PeopleTrackingAdapter::onResultReady(DetectionResult& result) {
    struct DetectionInfo {
        size_t index;
        cv::Point2f centroid;
    };
    
    std::vector<DetectionInfo> persons;
    for (size_t i = 0; i < result.detections.size(); ++i) {
        // Filter by Class ID 0 (Person/Human)
        if (result.detections[i].classId == 0) {
            const auto& box = result.detections[i].box;
            float cx = box.x + box.width / 2.0f;
            float cy = box.y + box.height / 2.0f;
            persons.push_back({i, cv::Point2f(cx, cy)});
        }
    }
    
    for (auto& pair : m_tracks) {
        pair.second.missingFrames++;
    }

    std::set<int> matchedTrackIds;
    std::set<size_t> matchedDetectionIndices;

    for (auto& pair : m_tracks) {
        int trackId = pair.first;
        TrackedObject& track = pair.second;

        float bestDist = m_maxDistance;
        int bestIdx = -1;

        for (size_t i = 0; i < persons.size(); ++i) {
            if (matchedDetectionIndices.count(persons[i].index)) continue;
            
            float d = calculateDistance(track.centroid, persons[i].centroid);
            if (d < bestDist) {
                bestDist = d;
                bestIdx = i;
            }
        }

        if (bestIdx != -1) {
            track.centroid = persons[bestIdx].centroid;
            track.missingFrames = 0;
            matchedTrackIds.insert(trackId);
            matchedDetectionIndices.insert(persons[bestIdx].index);
            result.detections[persons[bestIdx].index].label += " [ID: " + std::to_string(trackId) + "]";
        }
    }

    for (size_t i = 0; i < persons.size(); ++i) {
        if (matchedDetectionIndices.count(persons[i].index) == 0) {
            // New Person
            int newId = m_nextId++;
            TrackedObject obj;
            obj.id = newId;
            obj.centroid = persons[i].centroid;
            obj.missingFrames = 0;
            
            m_tracks[newId] = obj;
            
            // Assign ID
             result.detections[persons[i].index].label += " [ID: " + std::to_string(newId) + "]";
        }
    }

    for (auto it = m_tracks.begin(); it != m_tracks.end(); ) {
        if (it->second.missingFrames > m_maxMissingFrames) {
            it = m_tracks.erase(it);
        } else {
            ++it;
        }
    }
    
    std::vector<DetectionBox> finalDetections;
    finalDetections.reserve(persons.size()); 
    
    for (const auto& pInfo : persons) {
        finalDetections.push_back(result.detections[pInfo.index]);
    }
    
    result.detections = finalDetections;

    result.detections = finalDetections;

    result.overlayText = QString("Total People: %1").arg(persons.size());
}
