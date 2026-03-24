#ifndef STOREQUEUEANALYSISADAPTER_H
#define STOREQUEUEANALYSISADAPTER_H

#include "YoloV8Adapter.h"
#include <map>
#include <vector>
#include <chrono>

struct QueueTrack {
    int id;
    cv::Point2f centroid; // Normalized coordinates (0-1)
    std::chrono::steady_clock::time_point startTime; // Time when first seen (for ID persistence)
    std::chrono::steady_clock::time_point queueEnterTime; // Time when entered queue region
    bool inQueue; // Is currently inside queue region
    int missingFrames;
};

class StoreQueueAnalysisAdapter : public YoloV8Adapter {
    Q_OBJECT
public:
    explicit StoreQueueAnalysisAdapter(QObject* parent = nullptr);
    ~StoreQueueAnalysisAdapter() override = default;

    bool loadConfig(const QJsonObject& config) override;

protected:
    void onResultReady(DetectionResult& result) override;

private:
    std::map<int, QueueTrack> m_tracks;
    int m_nextId;
    int m_maxMissingFrames;
    float m_maxDistance; // Threshold for matching (normalized)
    
    // Queue Segmentation
    std::vector<std::vector<cv::Point2f>> m_queueRegions; // Multiple polygons
    bool isInsideQueue(const cv::Point2f& p);

    float calculateDistance(const cv::Point2f& p1, const cv::Point2f& p2);
};

#endif // STOREQUEUEANALYSISADAPTER_H
