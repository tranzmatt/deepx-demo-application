#include "StoreQueueAnalysisAdapter.h"
#include <QDebug>
#include <QJsonArray>
#include <cmath>
#include <algorithm>

StoreQueueAnalysisAdapter::StoreQueueAnalysisAdapter(QObject* parent)
    : YoloV8Adapter(parent)
    , m_nextId(1)
    , m_maxMissingFrames(10)
    , m_maxDistance(0.1f)
{
    qDebug() << "[StoreQueueAnalysisAdapter] Initialized";
}

bool StoreQueueAnalysisAdapter::loadConfig(const QJsonObject& config) {
    bool success = YoloV8Adapter::loadConfig(config);
    if (!success) return false;

    // Load Regions
    if (config.contains("regions") && config["regions"].isArray()) {
        // Get Resolution for Normalization
        float width = 1280.0f;
        float height = 720.0f;
        
        if (config.contains("application") && config["application"].toObject().contains("resolution")) {
            QJsonArray res = config["application"].toObject()["resolution"].toArray();
            if (res.size() >= 2) {
                width = res[0].toDouble();
                height = res[1].toDouble();
            }
        }

        QJsonArray regionsArray = config["regions"].toArray();
        if (!regionsArray.isEmpty()) {
            m_queueRegions.clear();
            
            for (const auto& polygonVal : regionsArray) {
                QJsonArray polygon = polygonVal.toArray();
                std::vector<cv::Point2f> region;
                
                for (const auto& pointVal : polygon) {
                    QJsonArray point = pointVal.toArray();
                    if (point.size() >= 2) {
                        float x = point[0].toDouble();
                        float y = point[1].toDouble();
                        
                        // Check if pixel coordinates (assuming value > 1.0 is pixel)
                        if (x > 1.0f || y > 1.0f) {
                            x /= width;
                            y /= height;
                        }
                        
                        region.push_back(cv::Point2f(x, y));
                    }
                }
                
                if (!region.empty()) {
                    m_queueRegions.push_back(region);
                }
            }
            qDebug() << "[StoreQueueAnalysisAdapter] Loaded" << m_queueRegions.size() << "regions";
        }
    }

    return true;
}

float StoreQueueAnalysisAdapter::calculateDistance(const cv::Point2f& p1, const cv::Point2f& p2) {
    float dx = p1.x - p2.x;
    float dy = p1.y - p2.y;
    return std::sqrt(dx*dx + dy*dy);
}

bool StoreQueueAnalysisAdapter::isInsideQueue(const cv::Point2f& p) {
    for (const auto& region : m_queueRegions) {
        // strict > 0 means inside, = 0 on edge, < 0 outside
        if (cv::pointPolygonTest(region, p, false) >= 0) {
            return true;
        }
    }
    return false;
}

void StoreQueueAnalysisAdapter::onResultReady(DetectionResult& result) {
    // 1. Extract persons
    struct DetectionInfo {
        size_t index;
        cv::Point2f centroid;
    };
    
    std::vector<DetectionInfo> persons;
    for (size_t i = 0; i < result.detections.size(); ++i) {
        // Filter by Class ID 0 (Person) or label "person"
        if (result.detections[i].classId == 0 || result.detections[i].label == "person") {
            const auto& box = result.detections[i].box;
            float cx = box.x + box.width / 2.0f;
            float cy = box.y + box.height / 2.0f;
            persons.push_back({i, cv::Point2f(cx, cy)});
        }
    }

    // 2. Update tracks (Greedy Matching)
    for (auto& pair : m_tracks) {
        pair.second.missingFrames++;
    }

    std::set<size_t> matchedDetectionIndices;
    
    // Match existing tracks
    for (auto& pair : m_tracks) {
        QueueTrack& qTrack = pair.second;

        float bestDist = m_maxDistance;
        int bestIdx = -1;

        for (size_t i = 0; i < persons.size(); ++i) {
            if (matchedDetectionIndices.count(persons[i].index)) continue;
            
            float d = calculateDistance(qTrack.centroid, persons[i].centroid);
            if (d < bestDist) {
                bestDist = d;
                bestIdx = i;
            }
        }

        if (bestIdx != -1) {
            // Update Track
            qTrack.centroid = persons[bestIdx].centroid;
            qTrack.missingFrames = 0;
            matchedDetectionIndices.insert(persons[bestIdx].index);
            
            // Queue Logic
            bool currentlyInQueue = isInsideQueue(qTrack.centroid);
            
            if (currentlyInQueue && !qTrack.inQueue) {
                // Determine Entry
                qTrack.inQueue = true;
                qTrack.queueEnterTime = std::chrono::steady_clock::now();
            } else if (!currentlyInQueue && qTrack.inQueue) {
                // Determine Exit
                qTrack.inQueue = false;
            }
            
            // Calculate Duration if in queue
            long long duration = 0;
            if (qTrack.inQueue) {
                auto now = std::chrono::steady_clock::now();
                duration = std::chrono::duration_cast<std::chrono::seconds>(now - qTrack.queueEnterTime).count();
            }
            
            // Update Visualization
            DetectionBox& box = result.detections[persons[bestIdx].index];
            
            if (qTrack.inQueue) {
                // Insider: In Queue with time
                box.label = "In queue " + std::to_string(qTrack.id) + ": " + std::to_string(duration) + "s";
                
                // Color Logic (BGRA) -> Alpha ~100 for transparency
                int alpha = 100;
                
                if (duration < 7) {
                    box.color = cv::Scalar(0, 255, 0, alpha); // Green
                } else if (duration < 15) {
                    box.color = cv::Scalar(0, 255, 255, alpha); // Yellow
                } else {
                    box.color = cv::Scalar(0, 0, 255, alpha); // Red
                }
            } else {
                // Outsider: Person with ID, standard color
                box.label = "Person " + std::to_string(qTrack.id);
                box.color = cv::Scalar(255, 200, 0, 255); // Blue-ish, opaque
            }
        }
    }

    // 3. Create new tracks
    for (size_t i = 0; i < persons.size(); ++i) {
        if (matchedDetectionIndices.count(persons[i].index) == 0) {
            int newId = m_nextId++;
            QueueTrack t;
            t.id = newId;
            t.centroid = persons[i].centroid;
            t.startTime = std::chrono::steady_clock::now();
            t.missingFrames = 0;
            
            // Check initial queue status
            t.inQueue = isInsideQueue(t.centroid);
            if (t.inQueue) {
                t.queueEnterTime = t.startTime;
            }
            
            m_tracks[newId] = t;
            
            // Initial Viz
            DetectionBox& box = result.detections[persons[i].index];
            if (t.inQueue) {
                box.label = "In Queue " + std::to_string(newId) + ": 0s";
                box.color = cv::Scalar(0, 255, 0, 100); // Green, semi-transparent
            } else {
                box.label = "Person " + std::to_string(newId);
                box.color = cv::Scalar(255, 200, 0, 255); // Blue-ish, opaque
            }
        }
    }

    // 4. Remove lost tracks
    for (auto it = m_tracks.begin(); it != m_tracks.end(); ) {
        if (it->second.missingFrames > m_maxMissingFrames) {
            it = m_tracks.erase(it);
        } else {
            ++it;
        }
    }
    
    // 5. Filter result to show ONLY tracked persons
    std::vector<DetectionBox> finalDetections;
    finalDetections.reserve(persons.size());

    for (const auto& pInfo : persons) {
        finalDetections.push_back(result.detections[pInfo.index]);
    }

    result.detections = finalDetections;

    // 6. Update overlay info
    int queueCount = 0;
    for (const auto& pair : m_tracks) {
        if (pair.second.inQueue) {
            queueCount++;
        }
    }
    result.overlayText = QString("Queue Count: %1").arg(queueCount);
    
    // Visualization of ALL queue regions (flattened with separators)
    result.overlayPolygon.clear();
    for (size_t i = 0; i < m_queueRegions.size(); ++i) {
        const auto& region = m_queueRegions[i];
        result.overlayPolygon.insert(result.overlayPolygon.end(), region.begin(), region.end());
        
        // Add separator if not the last region
        if (i < m_queueRegions.size() - 1) {
            result.overlayPolygon.push_back(cv::Point2f(-1.0f, -1.0f));
        }
    }
}
