#include "ConfigLoader.h"
#include <QFile>
#include <QJsonDocument>
#include <QDebug>
#include <iostream>
#include <algorithm>

ConfigLoader::ConfigLoader(const QString& configDir, QObject* parent)
    : QObject(parent), m_configDir(configDir) {}

void ConfigLoader::scan() {
    m_configs.clear();
    QDir dir(m_configDir);
    QStringList filters;
    filters << "*.json";
    dir.setNameFilters(filters);

    for (const QString& filename : dir.entryList()) {
        QString path = dir.absoluteFilePath(filename);
        QFile file(path);
        if (!file.open(QIODevice::ReadOnly)) {
            qWarning() << "Could not open config file:" << path;
            continue;
        }

        QByteArray data = file.readAll();
        QJsonDocument doc = QJsonDocument::fromJson(data);
        if (doc.isNull() || !doc.isObject()) {
             qWarning() << "Invalid JSON in:" << path;
             continue;
        }

        QJsonObject obj = doc.object();
        AdapterConfig config;
        config.name = obj.value("name").toString(filename);
        config.description = obj.value("description").toString(config.name); // Fallback to name/filename
        config.subDescription = obj.value("sub_description").toString();
        config.adapterType = obj.value("adapter").toString();
        config.filePath = path;
        config.json = obj;
        
        // Parse input type from input.sources[0].type
        QJsonObject input = obj.value("input").toObject();
        QJsonArray sources = input.value("sources").toArray();
        config.sourceCount = sources.size();
        if (!sources.isEmpty()) {
            config.inputType = sources[0].toObject().value("type").toString("video");
        } else {
            config.inputType = "video"; // Default
        }

        // Adapter inference logic
        if (config.adapterType.isEmpty()) {
            // First check explicitly for multichannel
            if (obj.value("multichannel").toBool(false)) {
                config.adapterType = "multichannel";
            } else {
                QJsonObject output = obj.value("output").toObject();
                QString type = output.value("type").toString();
                
                QJsonObject model = obj.value("model").toObject();
                QString modelPath = model.value("path").toString();

            // Heuristics
            QString usage = obj.value("usage").toString();
            if (usage == "ocr" || filename.contains("ocr", Qt::CaseInsensitive)) {
                config.adapterType = "ocr";
            }
            else if (filename.contains("trespassing", Qt::CaseInsensitive)) {
                 config.adapterType = "yolo_trespassing"; // Explicit request for trespassing
            }
            else if (type == "yolo_people_tracking") {
                 config.adapterType = "yolo_people_tracking";
            }
            else if (type == "store_queue_analysis") {
                 config.adapterType = "store_queue_analysis";
            }
            else if (modelPath.contains("pose", Qt::CaseInsensitive)) {
                 config.adapterType = "yolov8_pose"; 
            }
            else if (type == "yolov5_512") {
                 config.adapterType = "yolov5_512";
            }
            else if (path.contains("ppu", Qt::CaseInsensitive) || type == "yolo") {
                QJsonObject param = model.value("param").toObject();
                
                // Check explicit decoding method
                if (param.value("decoding_method").toString() == "yolo_face") {
                     config.adapterType = "yolo_face";
                }
                else {
                    QJsonArray layers = param.value("layer").toArray();
                    bool hasAnchors = false;
                    if (!layers.isEmpty()) {
                         QJsonObject l = layers[0].toObject();
                         QJsonArray anch = l.value("anchor_width").toArray();
                         for(const auto& val: layers) {
                             if(!val.toObject().value("anchor_width").toArray().isEmpty()) {
                                 hasAnchors = true; 
                                 break;
                             }
                         }
                    }
                    
                    if (!hasAnchors) {
                         config.adapterType = "yolov8_ppu"; // Use V8 PPU Adapter (Direct)
                    } else {
                         config.adapterType = "yolov5_ppu"; // Use Standard Yolo Adapter (Anchors)
                    }
                     }
                }
            }
        }

        if (!config.adapterType.isEmpty()) {
            m_configs.push_back(config);
        } else {
             qDebug() << "Skipping config (no adapter inferred):" << filename;
        }
    }
}

std::vector<AdapterConfig> ConfigLoader::getConfigs() const {
    return m_configs;
}
