#ifndef ADAPTERFACTORY_H
#define ADAPTERFACTORY_H

#include <memory>
#include <QString>
#include "IAdapter.h"
#include <functional>
#include <map>

class AdapterFactory {
public:
    static std::shared_ptr<IAdapter> createAdapter(const QString& type, int sourceCount = 1);
};

#endif // ADAPTERFACTORY_H
