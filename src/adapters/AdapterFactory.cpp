#include "AdapterFactory.h"
#include "YoloAdapter.h"
#include "YoloPoseAdapter.h"
#include "YoloV8Adapter.h"
#include "YoloAsyncAdapter.h"
#include "TrespassingAdapter.h"
#include "PeopleTrackingAdapter.h"
#include "YoloFaceAdapter.h"
#include "MultiChannelAdapter.h"
#include "SmartTrafficAdapter.h"
#include "HandLandmarkAdapter.h"
#include "FaceAlignmentAdapter.h"
#include "OcrAdapter.h"
#include "StoreQueueAnalysisAdapter.h"
#include "YoloV5_512Adapter.h"
#include <QDebug>

std::shared_ptr<IAdapter> AdapterFactory::createAdapter(const QString& type, int sourceCount) {
    if (type == "ocr") {
        return std::make_shared<OcrAdapter>();
    }

    if (type == "multichannel") {
        return std::make_shared<MultiChannelAdapter>();
    }
    
    if (type == "yolo_async") {
        return std::make_shared<YoloAsyncAdapter>();
    }
    if (type == "yolo_trespassing") {
        return std::make_shared<TrespassingAdapter>();
    }
    if (type == "yolo_people_tracking") {
        return std::make_shared<PeopleTrackingAdapter>();
    }
    if (type == "yolo_face") {
        return std::make_shared<YoloFaceAdapter>();
    }
    if (type == "smart_traffic") {
        return std::make_shared<SmartTrafficAdapter>();
    }
    if (type == "hand_landmark") {
        return std::make_shared<HandLandmarkAdapter>();
    }
    if (type == "face_alignment") {
        return std::make_shared<FaceAlignmentAdapter>();
    }
    if (type == "store_queue_analysis") {
        return std::make_shared<StoreQueueAnalysisAdapter>();
    }

    if (type.startsWith("yolov", Qt::CaseInsensitive)) {
        if (type.contains("pose", Qt::CaseInsensitive)) {
            return std::make_shared<YoloPoseAdapter>();
        }
        if (type == "yolov8_ppu") {
            return std::make_shared<YoloV8Adapter>();
        }
        if (type == "yolov5_512") {
            return std::make_shared<YoloV5_512Adapter>();
        }
        return std::make_shared<YoloAdapter>(); // Default for yolov5_ppu or others
    }
    
    // Future: classification, segmentation adapters
    qWarning() << "Unknown adapter type:" << type;
    return nullptr;
}
