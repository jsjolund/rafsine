#pragma once

#include <QStringBuilder>
#include <QStringList>
#include <QTreeWidget>

#include <algorithm>
#include <functional>
#include <sstream>

#include "VoxelGeometry.hpp"

/**
 * @brief This QT tree class shows details about boundary conditions
 * 
 */
class CFDTreeWidget : public QTreeWidget {
  Q_OBJECT
 private:
  QString vecToQStr(vec3<real> vec);

 public:
  explicit CFDTreeWidget(QWidget *);
  ~CFDTreeWidget();
  void buildModel(std::shared_ptr<VoxelGeometry> voxelGeometry);
};
