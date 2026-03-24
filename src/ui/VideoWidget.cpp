#include "VideoWidget.h"
#include <QDebug>
#include <QPainterPath>
#include <QMouseEvent>

VideoWidget::VideoWidget(QWidget* parent) : QWidget(parent) {
    setAttribute(Qt::WA_OpaquePaintEvent);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
}

void VideoWidget::updateFrame(const QImage& frame, const DetectionResult& result) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_currentFrame = frame;
    m_currentResult = result;
    update(); // Schedule repaint
}

void VideoWidget::paintEvent(QPaintEvent* event) {
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing); // Smooth lines

    // Blue background color (same as top bar)
    QColor bgColor("#204c63");

    // Draw Frame
    QImage frame;
    DetectionResult result;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_currentFrame.isNull()) {
            painter.fillRect(rect(), bgColor);
            painter.setPen(Qt::white);
            painter.drawText(rect(), Qt::AlignCenter, "No Signal");
            return;
        }
        frame = m_currentFrame;
        result = m_currentResult;
    }

    // Fill background first (for letterboxing)
    painter.fillRect(rect(), bgColor);

    // Scale aspect ratio
    QSize widgetSize = size();
    QSize frameSize = frame.size();
    
    // Fit Aspect
    float s = std::min((float)widgetSize.width() / frameSize.width(), (float)widgetSize.height() / frameSize.height());
    int w = frameSize.width() * s;
    int h = frameSize.height() * s;
    int x = (widgetSize.width() - w) / 2;
    int y = (widgetSize.height() - h) / 2;

    QRect targetRect(x, y, w, h);
    painter.drawImage(targetRect, frame);

    // Draw Detections
    for (const auto& box : result.detections) {
        // box coords are normalized [0,1] relative to the frame
        int bx = x + box.box.x * w;
        int by = y + box.box.y * h;
        int bw = box.box.width * w;
        int bh = box.box.height * h;

        QRect r(bx, by, bw, bh);
        
        int alpha = (box.color[3] > 0) ? (int)box.color[3] : 255;
        QColor color(box.color[2], box.color[1], box.color[0], alpha); // BGR to RGBA
        QPen pen(color);
        pen.setWidth(2);
        painter.setPen(pen);
        painter.drawRect(r);
        
        if (alpha < 255) {
            painter.fillRect(r, color);
        }

        // Draw Label (skip if label is empty)
        if (!box.label.empty()) {
            QString label = QString("%1 %2%").arg(QString::fromStdString(box.label)).arg((int)(box.confidence * 100));
            
            // Background for text
            QFont font("Sans Serif", 18, QFont::Bold);

            font.setStyleStrategy(QFont::PreferAntialias);
            painter.setFont(font);
            
            QFontMetrics fm(painter.font());
            int th = fm.height();
            int tw = fm.horizontalAdvance(label);
            
            int ty = by - th;
            if (ty < y) ty = by; // If label goes out of top, push it down
            
            QRect textRect(bx, ty, tw + 8, th);
            
            painter.fillRect(textRect, color); // Same color as bbox
            painter.setPen(Qt::white); // White text
            painter.drawText(textRect, Qt::AlignCenter, label);
        }

        // Draw Landmarks (Pose or Hand)
        if (!box.landmarks.empty()) {
            QVector<QPoint> points;
            for (size_t i = 0; i < box.landmarks.size(); i += 3) {
                float lx = box.landmarks[i];
                float ly = box.landmarks[i+1];
                float s = box.landmarks[i+2];

                if (s > 0.5f) { // Visibility threshold
                    int px = x + lx * w;
                    int py = y + ly * h;
                    painter.setPen(QPen(Qt::red, 5));
                    painter.drawPoint(px, py);
                    points.push_back(QPoint(px, py));
                } else {
                    points.push_back(QPoint(0,0)); // Placeholders
                }
            }

            // Select skeleton based on detection type
            std::vector<std::pair<int, int>> skeleton;
            
            if (box.label == "hand") {
                // Hand landmark connections (21 keypoints - MediaPipe style)
                // Wrist=0, Thumb: 1-4, Index: 5-8, Middle: 9-12, Ring: 13-16, Pinky: 17-20
                skeleton = {
                    // Thumb
                    {0,1}, {1,2}, {2,3}, {3,4},
                    // Index finger
                    {0,5}, {5,6}, {6,7}, {7,8},
                    // Middle finger
                    {0,9}, {9,10}, {10,11}, {11,12},
                    // Ring finger
                    {0,13}, {13,14}, {14,15}, {15,16},
                    // Pinky
                    {0,17}, {17,18}, {18,19}, {19,20},
                    // Palm connections
                    {5,9}, {9,13}, {13,17}
                };
            } else {
                // COCO 17 Keypoints (Body Pose)
                // Nose=0, LEye=1, REye=2, LEar=3, REar=4, LSho=5, RSho=6, LElb=7, RElb=8, LWri=9, RWri=10
                // LHip=11, RHip=12, LKne=13, RKne=14, LAnk=15, RAnk=16
                skeleton = {
                    {5,7}, {7,9}, {6,8}, {8,10}, // Arms
                    {11,13}, {13,15}, {12,14}, {14,16}, // Legs
                    {5,6}, {11,12}, {5,11}, {6,12}, // Body
                    {0,1}, {0,2}, {1,3}, {2,4} // Face
                };
            }

            painter.setPen(QPen(Qt::green, 2));
            for (const auto& pair : skeleton) {
                if (pair.first < points.size() && pair.second < points.size()) {
                    QPoint p1 = points[pair.first];
                    QPoint p2 = points[pair.second];
                    if (!p1.isNull() && !p2.isNull()) {
                        painter.drawLine(p1, p2);
                    }
                }
            }
        }
    }

    // Draw Overlay Polygon
    if (!result.overlayPolygon.empty()) {
        // Alert: Red with 100 alpha
        // Safe: Cyan/White with very low alpha (30) for "very light/transparent" look
        QColor polyColor = result.alertTriggered ? QColor(255, 0, 0, 100) : QColor(200, 255, 255, 30);
        QBrush polyBrush(polyColor);
        QPainterPath path;
        
        // Convert normalized points
        bool first = true;
        for (const auto& pt : result.overlayPolygon) {
            // Check for separator
            if (pt.x < -0.5f && pt.y < -0.5f) {
                path.closeSubpath();
                first = true;
                continue;
            }

            float px = x + pt.x * w;
            float py = y + pt.y * h;
            if (first) {
                path.moveTo(px, py);
                first = false;
            } else {
                path.lineTo(px, py);
            }
        }
        path.closeSubpath();
        
        painter.setPen(QPen(result.alertTriggered ? Qt::red : Qt::green, 2));
        painter.setBrush(polyBrush);
        painter.drawPath(path);
    }

    // Draw Alert Message
    if (result.alertTriggered && !result.alertMessage.isEmpty()) {
        painter.setFont(QFont("Arial", 18, QFont::Bold));
        painter.setPen(Qt::red);
        painter.drawText(rect(), Qt::AlignTop | Qt::AlignLeft, result.alertMessage);
    }

    // Draw General Overlay Text (e.g. Total Count) - Top Right
    if (!result.overlayText.isEmpty()) {
        QFont font("Segoe UI", 18, QFont::Bold);
        painter.setFont(font);
        
        QFontMetrics fm(font);
        int tw = fm.horizontalAdvance(result.overlayText);
        int th = fm.height();
        
        int padding = 10;
        int tx = rect().width() - tw - padding - 20; // 20px margin right
        int ty = padding + 10; // 10px margin top
        
        QRect textRect(tx - padding, ty - padding, tw + padding*2, th + padding*2);
        
        // Semi-transparent background
        painter.setBrush(QColor(0, 0, 0, 150));
        painter.setPen(Qt::NoPen);
        painter.drawRoundedRect(textRect, 8, 8);
        
        painter.setPen(QColor("#58a6ff")); // nice blue
        painter.drawText(textRect, Qt::AlignCenter, result.overlayText);
    }
}

void VideoWidget::mousePressEvent(QMouseEvent* event) {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_currentFrame.isNull()) return;

    QSize widgetSize = size();
    QSize frameSize = m_currentFrame.size();
    
    float s = std::min((float)widgetSize.width() / frameSize.width(), (float)widgetSize.height() / frameSize.height());
    int w = frameSize.width() * s;
    int h = frameSize.height() * s;
    int x = (widgetSize.width() - w) / 2;
    int y = (widgetSize.height() - h) / 2;

    int mx = event->x();
    int my = event->y();

    // Check if click is inside the frame
    if (mx >= x && mx < x + w && my >= y && my < y + h) {
        // Convert to frame coordinates (absolute)
        int fx = (mx - x) / s;
        int fy = (my - y) / s;
        emit mouseClicked(fx, fy);
    }
    
    QWidget::mousePressEvent(event);
}
