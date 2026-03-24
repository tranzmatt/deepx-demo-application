#ifndef TRESPASSINGADAPTER_H
#define TRESPASSINGADAPTER_H

#include "YoloAsyncAdapter.h"
#include <vector>
#include <opencv2/core.hpp>

class TrespassingAdapter : public YoloAsyncAdapter {
    Q_OBJECT
public:
    explicit TrespassingAdapter(QObject* parent = nullptr);
    bool loadConfig(const QJsonObject& config) override;

protected:
    void onResultReady(DetectionResult& result) override;

private:
    std::vector<cv::Point2f> m_polygon;
};

#endif // TRESPASSINGADAPTER_H
