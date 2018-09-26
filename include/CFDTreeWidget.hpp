#pragma once

#include <QTreeWidget>
#include <QStringList>
#include <QStringBuilder>

#include <algorithm>
#include <functional>
#include <sstream>

#include "VoxelGeometry.hpp"

class CFDTreeWidget : public QTreeWidget
{
  Q_OBJECT
private:
  QString vecToQStr(vec3<real> vec);
public:
  CFDTreeWidget(QWidget *);
  ~CFDTreeWidget();
  void buildModel(std::shared_ptr<VoxelGeometry> voxelGeometry);
};