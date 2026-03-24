#include "VideoSource.h"
#include <QDebug>
#include <thread>
#include <chrono>

VideoSource::VideoSource() {}

VideoSource::~VideoSource() {
    close();
}

bool VideoSource::open(const std::string& path) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_path = path;
    m_isFile = true;  // Assume file source
    return m_cap.open(path);
}

bool VideoSource::openCamera(int index) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_isFile = false;  // Camera source, no looping
    return m_cap.open(index);
}

bool VideoSource::openLibcamera(int index, int width, int height, int fps) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_isFile = false;

    // Build GStreamer pipeline for libcamera.
    // libcamerasrc camera-name expects the full device path, not a numeric index.
    // Use camera-name only when a non-default index is needed; otherwise omit it
    // so libcamerasrc picks the first available camera.
    std::string src = "libcamerasrc";
    if (index > 0) {
        src += " camera-name=" + std::to_string(index);
    }

    // Pipeline mirrors the confirmed-working gst-launch command:
    //   libcamerasrc ! video/x-raw,format=NV12,width=W,height=H,framerate=FPS/1
    //   ! videoconvert ! video/x-raw,format=BGR ! appsink …
    std::string pipeline =
        src + " ! "
        "video/x-raw,format=NV12"
        ",width="     + std::to_string(width)  +
        ",height="    + std::to_string(height) +
        ",framerate=" + std::to_string(fps) + "/1"
        " ! videoconvert"
        " ! video/x-raw,format=BGR"
        " ! appsink max-buffers=1 drop=true sync=false";

    qDebug() << "[VideoSource] Opening libcamera with pipeline:" << QString::fromStdString(pipeline);

    if (!m_cap.open(pipeline, cv::CAP_GSTREAMER)) {
        qWarning() << "[VideoSource] openLibcamera: cv::VideoCapture::open() failed";
        return false;
    }

    // Warm-up: libcamera may take a few frames to negotiate caps.
    // Attempt to grab a frame with retries before declaring success.
    const int kMaxRetries = 30;
    const int kRetryMs    = 100;
    for (int i = 0; i < kMaxRetries; ++i) {
        if (m_cap.grab()) {
            qDebug() << "[VideoSource] openLibcamera: pipeline ready after" << (i + 1) << "attempt(s)";
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(kRetryMs));
    }

    qWarning() << "[VideoSource] openLibcamera: pipeline opened but no frames received after"
               << kMaxRetries << "retries — closing";
    m_cap.release();
    return false;
}

void VideoSource::setFlip(bool flip) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_flip = flip;
}

bool VideoSource::read(cv::Mat& frame) {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_cap.isOpened()) return false;
    
    bool success = m_cap.read(frame);
    
    // If read failed and this is a file, loop back to start
    if (!success && m_isFile && !m_path.empty()) {
        m_cap.set(cv::CAP_PROP_POS_FRAMES, 0);  // Seek to beginning
        success = m_cap.read(frame);  // Try reading again
    }
    
    if (success && m_flip && !frame.empty()) {
        cv::flip(frame, frame, 0);  // 0 = vertical flip (around x-axis)
    }
    
    return success;
}

void VideoSource::close() {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_cap.isOpened()) {
        m_cap.release();
        // Small delay to allow GStreamer internal cleanup to settle
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

bool VideoSource::isOpened() const {
    // std::lock_guard<std::mutex> lock(m_mutex); // const method, be careful with mutex if cap usage changes
    return m_cap.isOpened();
}

double VideoSource::getFPS() const {
    if(!m_cap.isOpened()) return 0.0;
    return m_cap.get(cv::CAP_PROP_FPS);
}
