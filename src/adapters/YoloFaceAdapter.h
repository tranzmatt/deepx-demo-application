#ifndef YOLO_FACE_ADAPTER_H
#define YOLO_FACE_ADAPTER_H

#include "YoloAdapter.h"
#include "YoloFacePostProcess.h"

class YoloFaceAdapter : public YoloAdapter {
    Q_OBJECT
public:
    explicit YoloFaceAdapter(QObject* parent = nullptr);
    ~YoloFaceAdapter() override = default;

    bool loadConfig(const QJsonObject& config) override;

protected:
    void processLoop() override;

private:
    std::unique_ptr<YoloFacePostProcess> m_facePostProcess;
};

#endif // YOLO_FACE_ADAPTER_H
