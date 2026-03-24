#ifndef PEOPLETRACKINGADAPTER_H
#define PEOPLETRACKINGADAPTER_H

#include "YoloAdapter.h"
#include <map>
#include <set>
#include <vector>
#include <opencv2/core.hpp>

struct TrackedObject {
    int id;
    cv::Point2f centroid; // Normalized coordinates (0-1)
    int missingFrames;    // How many frames since last seen
    // Could add color or other history here
};

class PeopleTrackingAdapter : public YoloAdapter {
    Q_OBJECT
public:
    explicit PeopleTrackingAdapter(QObject* parent = nullptr);
    ~PeopleTrackingAdapter() override = default;

    bool loadConfig(const QJsonObject& config) override;

protected:
    void onResultReady(DetectionResult& result) override;

private:
    // Tracking state
    std::map<int, TrackedObject> m_tracks;
    int m_nextId;
    int m_maxMissingFrames;
    float m_maxDistance; // Threshold for matching (normalized)

    float calculateDistance(const cv::Point2f& p1, const cv::Point2f& p2);
};

#endif // PEOPLETRACKINGADAPTER_H
