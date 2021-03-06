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

#include "VoxelGeometry.hpp"

#define BC_NAME_COL_TITLE "In.Name"
#define BC_TEMP_COL_TITLE "Temp."
#define BC_FLOW_COL_TITLE "Vol.Flow"
#define BC_NAME_COL_IDX 0
#define BC_TEMP_COL_IDX 1
#define BC_FLOW_COL_IDX 2

/**
 * @brief These QT table classes show the list for setting the dynamic boundary
 * conditions
 *
 */
class BCInputTableModel : public QStandardItemModel {
  Q_OBJECT
 public:
  inline BCInputTableModel(int rows, int columns, QObject* parent = nullptr)
      : QStandardItemModel(rows, columns, parent) {}
  Qt::ItemFlags flags(const QModelIndex& index) const;
};

class BCInputTableDelegate : public QItemDelegate {
  Q_OBJECT
 private:
  QModelIndex m_index;
  QWidget* m_mainWindow;

 public:
  explicit BCInputTableDelegate(QWidget* mainWindow)
      : m_mainWindow(mainWindow) {}

  QWidget* createEditor(QWidget* parent,
                        const QStyleOptionViewItem& option,
                        const QModelIndex& index) const;
};

class BCInputTableView : public QTableView {
  Q_OBJECT

 private:
  BCInputTableModel* m_model;

 public:
  explicit BCInputTableView(QWidget* mainWindow);
  ~BCInputTableView();
  virtual void clear();
  void buildModel(std::shared_ptr<VoxelGeometry> voxelGeometry,
                  std::shared_ptr<UnitConverter> unitConverter);

  int updateModel(std::shared_ptr<VoxelGeometry> voxelGeometry,
                  std::shared_ptr<UnitConverter> unitConverter);

  void updateBoundaryConditions(std::shared_ptr<BoundaryConditions> bcs,
                                std::shared_ptr<VoxelGeometry> voxelGeometry,
                                std::shared_ptr<UnitConverter> uc);

  void setEditable(bool state) {
    QModelIndex parent = QModelIndex();
    if (!state) {
      setEditTriggers(QAbstractItemView::NoEditTriggers);
      for (int row = 0; row < m_model->rowCount(parent); ++row) {
        for (int col = 0; col < 3; col++)
          m_model->item(row, col)->setFlags(m_model->item(row, col)->flags() &
                                            ~Qt::ItemIsEditable);
      }
    } else {
      setEditTriggers(QAbstractItemView::AllEditTriggers);
      for (int row = 0; row < m_model->rowCount(parent); ++row) {
        for (int col = 0; col < 3; col++)
          m_model->item(row, col)->setFlags(m_model->item(row, col)->flags() |
                                            Qt::ItemIsEditable);
      }
    }
  }

  virtual void mousePressEvent(QMouseEvent* event);
};
