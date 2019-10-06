#pragma once

#include <QStringBuilder>
#include <QStringList>
#include <QTreeWidget>

#include <algorithm>
#include <functional>
#include <memory>
#include <sstream>

#include "VoxelGeometry.hpp"

template <typename T>
QString vecToQStr(T vec) {
  return QStringLiteral("(%1, %2, %3)").arg(vec.x()).arg(vec.y()).arg(vec.z());
}

/**
 * @brief This QT tree class shows details about boundary conditions
 *
 */
class CFDTreeWidget : public QTreeWidget {
  Q_OBJECT

 public:
  explicit CFDTreeWidget(QWidget *);
  ~CFDTreeWidget();
  void buildModel(std::shared_ptr<VoxelGeometry> voxelGeometry);
};
