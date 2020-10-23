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

#include "AverageObserver.hpp"
#include "VoxelGeometry.hpp"

#define AVG_NAME_COL_TITLE "Out.Name"
#define AVG_TEMP_COL_TITLE "Temp."
#define AVG_FLOW_COL_TITLE "Vol.Flow"
#define AVG_NAME_COL_IDX 0
#define AVG_TEMP_COL_IDX 1
#define AVG_FLOW_COL_IDX 2

class AverageOutputTableModel : public QStandardItemModel {
  Q_OBJECT
 public:
  inline AverageOutputTableModel(int rows,
                                 int columns,
                                 QObject* parent = nullptr)
      : QStandardItemModel(rows, columns, parent) {}
  void update(const AverageMatrix& avgs);
};

class AverageOutputTableView : public QTableView, public AverageObserver {
  Q_OBJECT

 private:
  AverageOutputTableModel* m_model;

 public:
  explicit AverageOutputTableView(QWidget* parent);
  ~AverageOutputTableView();
  virtual void clear();
  void buildModel(const VoxelCuboidArray& volumes);

  void notify(const AverageMatrix& avgs);
};
