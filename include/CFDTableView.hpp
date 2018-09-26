#pragma once

#include <QHeaderView>
#include <QMouseEvent>
#include <QStandardItemModel>
#include <QTableView>
#include <QItemDelegate>
#include <QLineEdit>
#include <QDoubleValidator>
#include <QStyleOptionViewItem>
#include <QWidget>
#include <QModelIndex>

#include "VoxelGeometry.hpp"

class CFDTableModel : public QStandardItemModel
{
  Q_OBJECT
public:
  inline CFDTableModel(int rows, int columns, QObject *parent = nullptr)
      : QStandardItemModel(rows, columns, parent) {}
  Qt::ItemFlags flags(const QModelIndex &index) const;
};

class Delegate : public QItemDelegate
{
  Q_OBJECT
private:
  QModelIndex m_index;

public:
  Q_SLOT void onTextEdited()
  {
    std::cout << "hi " << std::endl;
  }

  QWidget *createEditor(QWidget *parent,
                        const QStyleOptionViewItem &option,
                        const QModelIndex &index) const
  {
    QLineEdit *lineEdit = new QLineEdit(parent);
    QDoubleValidator *validator = new QDoubleValidator();
    validator->setNotation(QDoubleValidator::ScientificNotation);
    lineEdit->setValidator(validator);
    connect(lineEdit, SIGNAL(editingFinished()), this, SLOT(onTextEdited()));
    return lineEdit;
  }
};

class CFDTableView : public QTableView
{
  Q_OBJECT
private:
  // std::shared_ptr<VoxelGeometry> m_voxels;
  QStandardItemModel *m_model;

public:
  CFDTableView(QWidget *);
  ~CFDTableView();
  void clear();
  void buildModel(std::shared_ptr<VoxelGeometry> voxelGeometry,
                  std::shared_ptr<UnitConverter> unitConverter);
  virtual void mousePressEvent(QMouseEvent *event);
};