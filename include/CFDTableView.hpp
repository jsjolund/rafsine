#pragma once
#pragma push
#pragma diag_suppress = 1427
#pragma pop

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

#include <glm/glm.hpp>

#include "VoxelGeometry.hpp"

class CFDTableModel : public QStandardItemModel {
  Q_OBJECT
 public:
  inline CFDTableModel(int rows, int columns, QObject *parent = nullptr)
      : QStandardItemModel(rows, columns, parent) {}
  Qt::ItemFlags flags(const QModelIndex &index) const;
};

class CFDTableDelegate : public QItemDelegate {
  Q_OBJECT
 private:
  QModelIndex m_index;
  QWidget *m_mainWindow;

 public:
  explicit CFDTableDelegate(QWidget *mainWindow) : m_mainWindow(mainWindow) {}

  QWidget *createEditor(QWidget *parent, const QStyleOptionViewItem &option,
                        const QModelIndex &index) const;
};

class CFDTableView : public QTableView {
  Q_OBJECT
 private:
  QStandardItemModel *m_model;

 public:
  explicit CFDTableView(QWidget *mainWindow);
  ~CFDTableView();
  virtual void clear();
  void buildModel(std::shared_ptr<VoxelGeometry> voxelGeometry,
                  std::shared_ptr<UnitConverter> unitConverter);

  void updateBoundaryConditions(BoundaryConditionsArray *bcs,
                                std::shared_ptr<VoxelGeometry> voxelGeometry,
                                std::shared_ptr<UnitConverter> uc);

  virtual void mousePressEvent(QMouseEvent *event);
};
