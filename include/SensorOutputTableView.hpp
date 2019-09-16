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
#include <string>
#include <vector>

#include <glm/glm.hpp>

#include "AverageObserver.hpp"
#include "VoxelGeometry.hpp"

class SensorOutputTableModel : public QStandardItemModel {
  Q_OBJECT
 public:
  inline SensorOutputTableModel(int rows, int columns, QObject *parent = nullptr)
      : QStandardItemModel(rows, columns, parent) {}
  void update(const AverageData &avgs);
};

class SensorOutputTableView : public QTableView, public AverageObserver {
  Q_OBJECT

 private:
  SensorOutputTableModel *m_model;

 public:
  explicit SensorOutputTableView(QWidget *parent);
  ~SensorOutputTableView();
  virtual void clear();
  void buildModel(const VoxelVolumeArray &volumes);

  void notify(const AverageData &avgs);
};
