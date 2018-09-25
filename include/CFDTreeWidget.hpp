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
  std::shared_ptr<VoxelGeometry> m_voxels;

public:
  CFDTreeWidget(QWidget *);
  ~CFDTreeWidget();
  QString vecToQStr(vec3<real> vec);
  void buildModel(std::shared_ptr<VoxelGeometry> voxelGeometry);
};