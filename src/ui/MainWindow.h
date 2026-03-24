#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QListWidget>
#include <QLabel>
#include <QTabWidget>
#include <QPushButton>
#include <QRadioButton>
#include <QButtonGroup>
#include <QJsonArray>
#include <QTableWidget>
#include <QProgressBar>
#include "SystemMonitor.h"
#include "VideoWidget.h"
#include "ConfigLoader.h"
#include "IAdapter.h"
#include <memory>

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow() override;

private slots:
    void onConfigSelected(QListWidgetItem* item);
    void onAdapterFrame(const QImage& frame, const DetectionResult& result);
    void onAdapterError(const QString& msg);
    void onFpsUpdate(double fps);
    void updateSystemStats();
    void toggleSidebar();
    void onSourceIndexChanged(int index);

private:
    void initUi();
    void loadConfigs();
    void startAdapter(const AdapterConfig& config);
    void stopAdapter();
    void rebuildSourceRadioButtons(const QJsonArray& sources);

    QListWidget* m_configList;
    VideoWidget* m_videoWidget;

    
    // Top Bar Widgets
    QWidget* m_topBar;
    QLabel* m_fpsLabel;
    QLabel* m_cpuUsageLabel;
    QLabel* m_cpuTempLabel; // Note: cpuTempLabel wasn't used in cpp yet, but it's in header
    QLabel* m_npuUsageLabel;
    QLabel* m_npuTempLabel;

    // Sidebar
    QWidget* m_leftPanel;
    QPushButton* m_toggleSidebarBtn;
    
    // Source Selection
    QWidget* m_sourceGroupWidget;
    QButtonGroup* m_sourceButtonGroup;
    QList<QRadioButton*> m_sourceRadioButtons;

    ConfigLoader* m_configLoader;
    std::shared_ptr<IAdapter> m_currentAdapter;
    AdapterConfig m_currentConfig;
    
    QThread* m_monitorThread;
    SystemMonitor* m_sysMonitor;
    double m_currentFps;
};

#endif // MAINWINDOW_H
