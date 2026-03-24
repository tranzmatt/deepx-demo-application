#ifndef CONFIGLOADER_H
#define CONFIGLOADER_H

#include <QObject>
#include <QJsonArray>
#include <QJsonObject>
#include <QDir>
#include <vector>

struct AdapterConfig {
    QString name;
    QString description;
    QString subDescription;
    QString adapterType;
    QString inputType;  // "video" or "camera"
    QString filePath;
    int sourceCount = 1;  // Number of input sources
    QJsonObject json;
};

class ConfigLoader : public QObject {
    Q_OBJECT
public:
    explicit ConfigLoader(const QString& configDir, QObject* parent = nullptr);
    void scan();
    std::vector<AdapterConfig> getConfigs() const;

private:
    QString m_configDir;
    std::vector<AdapterConfig> m_configs;
};

#endif // CONFIGLOADER_H
