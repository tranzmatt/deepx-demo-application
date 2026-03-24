#ifndef LIBCAMERASOURCE_H
#define LIBCAMERASOURCE_H

#include <opencv2/core.hpp>
#include <memory>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <map>
#include <vector>

#ifdef HAVE_LIBCAMERA

// Forward declarations - avoid exposing libcamera headers
namespace libcamera {
class CameraManager;
class Camera;
class CameraConfiguration;
class FrameBufferAllocator;
class FrameBuffer;
class Request;
class Stream;
}

/**
 * @brief Native libcamera video source for Raspberry Pi CSI cameras.
 *
 * Uses the libcamera C++ API directly to capture frames from
 * Raspberry Pi camera modules (v1/v2/v3/HQ) connected via CSI/MIPI.
 *
 * Frames are captured asynchronously and converted to BGR cv::Mat
 * for seamless integration with OpenCV-based processing pipelines.
 */
class LibcameraSource {
public:
    LibcameraSource();
    ~LibcameraSource();

    // Non-copyable, non-movable
    LibcameraSource(const LibcameraSource&) = delete;
    LibcameraSource& operator=(const LibcameraSource&) = delete;

    /**
     * @brief Open a libcamera-managed camera.
     * @param cameraIndex Index of the camera (0 = first CSI camera)
     * @param width Desired capture width (will be adjusted to nearest supported)
     * @param height Desired capture height
     * @param fps Desired frame rate
     * @return true if camera opened successfully
     */
    bool open(int cameraIndex = 0, int width = 640, int height = 480, double fps = 30.0);

    /**
     * @brief Read the latest captured frame (blocks until available or timeout).
     * @param frame Output BGR cv::Mat
     * @return true if frame was read successfully
     */
    bool read(cv::Mat& frame);

    /**
     * @brief Stop capture and release all resources.
     */
    void close();

    bool isOpened() const;
    double getFPS() const;

    /// Get actual capture width after configuration
    int getWidth() const { return m_width; }
    /// Get actual capture height after configuration  
    int getHeight() const { return m_height; }

private:
    /// Called by libcamera when a capture request completes
    void requestComplete(libcamera::Request* request);

    /// Pixel format tracking (avoids libcamera types in header)
    enum class PixFmt { BGR888, RGB888, NV12, NV21, YUV420, YUYV, Unknown };

    std::unique_ptr<libcamera::CameraManager> m_cameraManager;
    std::shared_ptr<libcamera::Camera> m_camera;
    std::unique_ptr<libcamera::CameraConfiguration> m_config;
    std::unique_ptr<libcamera::FrameBufferAllocator> m_allocator;
    libcamera::Stream* m_stream = nullptr;
    std::vector<std::unique_ptr<libcamera::Request>> m_requests;

    /// Memory-mapped buffer descriptor
    struct MappedBuffer {
        void* data = nullptr;
        size_t length = 0;
    };
    std::map<const libcamera::FrameBuffer*, MappedBuffer> m_mappedBuffers;

    /// Thread-safe frame exchange
    cv::Mat m_latestFrame;
    std::mutex m_frameMutex;
    std::condition_variable m_frameCv;
    bool m_newFrame = false;

    /// State
    std::atomic<bool> m_active{false};
    double m_fps = 30.0;
    int m_width = 640;
    int m_height = 480;
    int m_stride = 0;
    PixFmt m_pixFmt = PixFmt::Unknown;
};

#else // !HAVE_LIBCAMERA

/**
 * @brief Stub LibcameraSource when libcamera is not available.
 * All methods return failure gracefully.
 */
class LibcameraSource {
public:
    LibcameraSource() = default;
    ~LibcameraSource() = default;

    bool open(int = 0, int = 640, int = 480, double = 30.0) { return false; }
    bool read(cv::Mat&) { return false; }
    void close() {}
    bool isOpened() const { return false; }
    double getFPS() const { return 0.0; }
    int getWidth() const { return 0; }
    int getHeight() const { return 0; }
};

#endif // HAVE_LIBCAMERA

#endif // LIBCAMERASOURCE_H
