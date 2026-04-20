# Sixfab | DeepX AI Vision Platform Demo

This interactive demo provides a "DeepX AI Vision Platform" UI with real-time hardware monitoring, multi-stream video processing, and a diverse set of AI inference examples.

## Video Source Configuration

The application supports multiple camera and video input types. Video sources are configured in the JSON config files located in `src/configs/`. Each config file has an `input.sources` array where you define one or more video sources.

### Source Types

#### USB Camera (`camera`)

Use a standard USB camera (V4L2 device) as input.

```json
"input": {
    "format": "RGB",
    "sources": [
        {
            "type": "camera",
            "path": "/dev/video0"
        }
    ]
}
```

To find available USB cameras on your system:

```bash
ls /dev/video*
```

#### Raspberry Pi Camera (`libcamera`)

Use a Raspberry Pi Camera Module (CSI) via `libcamera`. The path format is `camera_id:width:height:fps`.

```json
"sources": [
    {
        "type": "libcamera",
        "path": "0:1280:720:30",
        "flip": true
    }
]
```

| Path Segment | Description              | Example |
|:-------------|:-------------------------|:--------|
| `0`          | Camera ID                | `0` (first CSI camera) |
| `1280`       | Width in pixels          | `1280` |
| `720`        | Height in pixels         | `720` |
| `30`         | Frame rate (fps)         | `30` |

Set `"flip": true` if your camera image is upside-down (common with certain mounting orientations).

#### RTSP Stream (`rtsp`)

Connect to an IP camera or network video stream via RTSP.

```json
"sources": [
    {
        "type": "rtsp",
        "path": "rtsp://192.168.1.100:554/live"
    }
]
```

Common RTSP URL patterns by manufacturer:

| Manufacturer | Example URL                                          |
|:-------------|:-----------------------------------------------------|
| Generic      | `rtsp://<ip>:<port>/live`                            |
| Hikvision    | `rtsp://<user>:<pass>@<ip>:554/Streaming/Channels/1` |
| Dahua        | `rtsp://<user>:<pass>@<ip>:554/cam/realmonitor?channel=1&subtype=0` |
| Amcrest      | `rtsp://<user>:<pass>@<ip>:554/cam/realmonitor?channel=1&subtype=0` |

#### Video File (`video`)

Use a local video file (MP4, AVI, etc.) as input.

```json
"sources": [
    {
        "type": "video",
        "path": "src/videos/traffic.mp4"
    }
]
```

#### Image File (`image`)

Use static image files as input (useful for OCR or single-frame analysis).

```json
"sources": [
    {
        "type": "image",
        "path": "src/images/sample.png"
    }
]
```

### Example Configurations

| Config File | Source Type | Description |
|:------------|:-----------|:------------|
| `yolov8n_ppu_libcamera_rpi.json` | RPi Camera | Object detection with Raspberry Pi Camera Module |
| `yolov11_ppu_realtime_example.json` | USB Camera + RTSP fallback | PPE compliance detection |
| `yolov5s_ppu_mask_detection_realtime.json` | Dual USB Camera | Mask detection with cameras |
| `smart_traffic_config.json` | Video File + RTSP | Traffic flow monitoring |
| `yolov8n_ppu_realtime_example_trespassing.json` | Multi-channel Video | 4-channel perimeter intrusion detection |
| `multi_videofiles_36ch.json` | Multi-channel Video | 36-channel large-scale video analytics |
| `ocr_v4.json` | Image + RPi Camera | OCR with static images and live camera |

## Getting Started

### Quick Install (One-Liner)

Clone and build the project with a single command. The installer checks for `git`, installs it if missing, clones this repo, and runs `build.sh`.

```bash
curl -fsSL https://raw.githubusercontent.com/sixfab/deepx-demo-application/master/install.sh | bash
```

### Manual Build

You can use the provided `build.sh` script to install dependencies, download required AI model resources (`resources.zip`), and compile the project automatically.

```bash
./build.sh
```

### 2. Run the Application

Once the build finishes successfully, the application executable will be located in the `build/` directory.

```bash
./build/cpp_qt_app
```

