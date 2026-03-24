#ifndef SMARTTRAFFICADAPTER_H
#define SMARTTRAFFICADAPTER_H

#include "YoloAsyncAdapter.h" // Inherit from Async adapter
#include <map>
#include <opencv2/core.hpp>

struct TrafficObject {
    int id;
    cv::Point2f centroid;
    int missingFrames;
    bool crossed; // Has this object already been counted?
};

class SmartTrafficAdapter : public YoloAsyncAdapter {
    Q_OBJECT
public:
    explicit SmartTrafficAdapter(QObject* parent = nullptr);
    ~SmartTrafficAdapter() override = default;

    bool loadConfig(const QJsonObject& config) override;

protected:
    void onResultReady(DetectionResult& result) override;

private:
    std::pair<cv::Point2f, cv::Point2f> m_line; // Normalized coordinates
    std::map<int, TrafficObject> m_tracks;
    int m_nextId;
    int m_maxMissingFrames;
    float m_maxMatchDistance;
    int m_count;

    float calculateDistance(const cv::Point2f& p1, const cv::Point2f& p2);
    
    // Check intersection between line segment (p1, p2) and m_line
    bool checkLineCrossing(const cv::Point2f& p1, const cv::Point2f& p2);
    
    // Helper for cross product
    float crossProduct(const cv::Point2f& a, const cv::Point2f& b, const cv::Point2f& c);
};

#endif // SMARTTRAFFICADAPTER_H
