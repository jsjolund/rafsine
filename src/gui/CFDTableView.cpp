#include "CFDTableView.hpp"

Qt::ItemFlags CFDTableModel::flags(const QModelIndex &index) const
{
  Qt::ItemFlags flags;

  flags = QStandardItemModel::flags(index);

  if (index.column() == 0)
  {
    flags &= ~Qt::ItemIsEditable;
    return flags;
  }
  return QStandardItemModel::flags(index);
}

CFDTableView::CFDTableView(QWidget *parent) : QTableView(parent)
{
  setAlternatingRowColors(true);
  setItemDelegate(new Delegate);
}

CFDTableView::~CFDTableView() {}

void CFDTableView::buildModel(std::shared_ptr<VoxelGeometry> voxelGeometry,
                              std::shared_ptr<UnitConverter> uc)
{
  std::vector<std::string> names = voxelGeometry->getGeometryNames();

  m_model = new CFDTableModel(names.size(), 3);
  m_model->setHeaderData(0, Qt::Horizontal, tr("Geometry"));
  m_model->setHeaderData(1, Qt::Horizontal, tr("Temp."));
  m_model->setHeaderData(2, Qt::Horizontal, tr("Vol.Flow"));

  for (int row = 0; row < names.size(); ++row)
  {
    std::string name = names.at(row);
    QStandardItem *nameItem = new QStandardItem(QString::fromStdString(name));
    m_model->setItem(row, 0, nameItem);

    std::unordered_set<VoxelQuad> quads = voxelGeometry->getQuadsByName(name);

    // VoxelQuad *zeroGradientQuad = nullptr;
    // VoxelQuad *constantOrRelativeQuad = nullptr;

    for (VoxelQuad quad : quads)
    {
      if (quad.m_bc.m_type == VoxelType::INLET_CONSTANT || quad.m_bc.m_type == VoxelType::INLET_RELATIVE)
      {
        QStandardItem *tempItem = new QStandardItem(QString::number(uc->luTemp_to_Temp(quad.m_bc.m_temperature)));
        m_model->setItem(row, 1, tempItem);
      }
    }
  }

  setModel(m_model);
  verticalHeader()->hide();
  resizeRowsToContents();
}

void CFDTableView::clear()
{
}

void CFDTableView::mousePressEvent(QMouseEvent *event)
{
  if (event->button() == Qt::LeftButton)
  {
    QModelIndex index = indexAt(event->pos());
    if (index.column() == 1 || index.column() == 2)
    { // column you want to use for one click
      edit(index);
    }
  }
  QTableView::mousePressEvent(event);
}