#ifndef IADAPTER_H
#define IADAPTER_H

#include <QObject>
#include <QImage>
#include <QJsonObject>
#include <opencv2/core.hpp>
#include "Common.h"
#include <QString>

struct SourceInfo {
    QString path;
    QString type;
    bool flip = false;  // Vertical flip (cv::flip code 0)
};

class IAdapter : public QObject {
    Q_OBJECT
public:
    explicit IAdapter(QObject* parent = nullptr) : QObject(parent) {}
    virtual ~IAdapter() = default;

    virtual bool loadConfig(const QJsonObject& config) = 0;
    virtual void start() = 0;
    virtual void stop() = 0;
    virtual void handleMouseEvent(int x, int y, int type) {}
    virtual void setSourceIndex(int index) = 0;

signals:
    void frameReady(const QImage& frame, const DetectionResult& result);
    void errorOccurred(const QString& message);
    void fpsUpdate(double processingFps);
};

#endif // IADAPTER_H
