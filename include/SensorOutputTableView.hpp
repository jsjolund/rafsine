#pragma once

#include <QAbstractItemView>
#include <QDoubleValidator>
#include <QHeaderView>
#include <QItemDelegate>
#include <QLineEdit>
#include <QModelIndex>
#include <QMouseEvent>
#include <QStandardItemModel>
#include <QStyleOptionViewItem>
#include <QTableView>
#include <QWidget>

#include <memory>

#include <glm/glm.hpp>

#include "VoxelGeometry.hpp"

class SensorOutputTableView : public QTableView {
  Q_OBJECT
 private:
  QStandardItemModel *m_model;

 public:
  explicit SensorOutputTableView(QWidget *parent);
  ~SensorOutputTableView();
  virtual void clear();
  void buildModel(std::shared_ptr<VoxelGeometry> voxelGeometry,
                  std::shared_ptr<UnitConverter> unitConverter);

  int updateModel(std::shared_ptr<VoxelGeometry> voxelGeometry,
                  std::shared_ptr<UnitConverter> unitConverter);
};
