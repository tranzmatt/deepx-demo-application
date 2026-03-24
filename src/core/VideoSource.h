#ifndef VIDEOSOURCE_H
#define VIDEOSOURCE_H

#include <string>
#include <mutex>
#include <opencv2/opencv.hpp>

class VideoSource {
public:
    VideoSource();
    ~VideoSource();

    bool open(const std::string& path);
    bool openCamera(int index);
    bool openLibcamera(int index, int width, int height, int fps);
    bool read(cv::Mat& frame);
    void close();
    bool isOpened() const;
    double getFPS() const;
    void setFlip(bool flip);  // Vertical flip applied on every read()

private:
    cv::VideoCapture m_cap;
    std::mutex m_mutex;
    std::string m_path;  // Store path for looping
    bool m_isFile = false;  // True if source is a file (not camera)
    bool m_flip = false;    // Apply vertical flip in read()
};

#endif // VIDEOSOURCE_H
