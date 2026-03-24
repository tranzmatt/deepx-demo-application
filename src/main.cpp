#include <QApplication>
#include "MainWindow.h"
#include <dxrt/device_pool.h>

int main(int argc, char *argv[]) {
    dxrt::DevicePool::GetInstance().InitCores();
    QApplication app(argc, argv);
    
    QApplication::setApplicationName("CppQtUnifiedDemo");
    QApplication::setApplicationVersion("1.0");
    
    // Style
    app.setStyle("Fusion");

    qRegisterMetaType<DetectionResult>("DetectionResult");

    MainWindow window;
    window.show();

    return app.exec();
}
