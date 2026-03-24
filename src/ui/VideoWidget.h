#ifndef VIDEOWIDGET_H
#define VIDEOWIDGET_H

#include <QWidget>
#include <QImage>
#include <QPainter>
#include <mutex>
#include "Common.h"

class VideoWidget : public QWidget {
    Q_OBJECT
public:
    explicit VideoWidget(QWidget* parent = nullptr);
    
public slots:
    void updateFrame(const QImage& frame, const DetectionResult& result);

protected:
    void paintEvent(QPaintEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;

signals:
    void mouseClicked(int x, int y);

private:
    QImage m_currentFrame;
    DetectionResult m_currentResult;
    std::mutex m_mutex;
};

#endif // VIDEOWIDGET_H
