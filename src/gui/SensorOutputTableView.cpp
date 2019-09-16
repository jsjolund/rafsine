#include "SensorOutputTableView.hpp"

void SensorOutputTableModel::update(const AverageData &avgs) {
  for (int row = 0; row < avgs.rows.size(); row++) {
    Average avg = avgs.rows.at(row);
    for (int col = 0; col < 3; col++) {
      QStandardItem *item = takeItem(row, col);
      if (col == 0)
        item->setText(QString::fromStdString(avg.m_volume.getName()));
      else if (col == 1)
        item->setText(QString::number(avg.m_temperature));
      else if (col == 2)
        item->setText(QString::number(avg.m_flow));
      setItem(row, col, item);
    }
  }
  QModelIndex topLeft = index(0, 0);
  QModelIndex bottomRight = index(rowCount() - 1, columnCount() - 1);
  emit dataChanged(topLeft, bottomRight);
  emit layoutChanged();
}

SensorOutputTableView::SensorOutputTableView(QWidget *parent)
    : QTableView(parent), m_model(nullptr) {
  setAlternatingRowColors(true);
  setEditTriggers(QAbstractItemView::NoEditTriggers);
}

SensorOutputTableView::~SensorOutputTableView() {}

void SensorOutputTableView::buildModel(const VoxelVolumeArray &volumes) {
  m_model = new SensorOutputTableModel(volumes.size(), 3);
  m_model->setHeaderData(0, Qt::Horizontal, tr("Geometry"));
  m_model->setHeaderData(1, Qt::Horizontal, tr("Temp."));
  m_model->setHeaderData(2, Qt::Horizontal, tr("Vol.Flow"));

  for (int row = 0; row < volumes.size(); row++) {
    std::string name = volumes.at(row).getName();
    real tempC = NaN;
    real flow = NaN;
    m_model->setItem(row, 0, new QStandardItem(QString::fromStdString(name)));
    m_model->setItem(row, 1, new QStandardItem(QString::number(tempC)));
    m_model->setItem(row, 2, new QStandardItem(QString::number(flow)));
  }
  setModel(m_model);
  verticalHeader()->hide();
  resizeRowsToContents();
  resizeColumnsToContents();
}

void SensorOutputTableView::clear() {
  if (m_model && m_model->rowCount() > 0) m_model->clear();
}

void SensorOutputTableView::notify(const AverageData &avgs) {
  m_model->update(avgs);
  viewport()->update();
}
