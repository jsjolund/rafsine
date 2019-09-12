#include "SensorOutputTableView.hpp"

SensorOutputTableView::SensorOutputTableView(QWidget *parent)
    : QTableView(parent), m_model(nullptr) {
  setAlternatingRowColors(true);
  setEditTriggers(QAbstractItemView::NoEditTriggers);
}

SensorOutputTableView::~SensorOutputTableView() {}

int SensorOutputTableView::updateModel(
    std::shared_ptr<VoxelGeometry> voxelGeometry,
    std::shared_ptr<UnitConverter> uc) {
  std::shared_ptr<VoxelVolumeArray> volumes = voxelGeometry->getSensors();
  int row;
  for (row = 0; row < volumes->size(); row++) {
    std::string name = volumes->at(row).getName();

    QStandardItem *nameItem = new QStandardItem(QString::fromStdString(name));
    m_model->setItem(row, 0, nameItem);

    // Set temperature cell
    real tempC = NaN;
    QStandardItem *tempItem = new QStandardItem(QString::number(tempC));
    m_model->setItem(row, 1, tempItem);

    // Set volumetric flow rate cell
    real flow = NaN;
    QStandardItem *flowItem = new QStandardItem(QString::number(flow));
    m_model->setItem(row, 2, flowItem);
  }
  return row;
}

void SensorOutputTableView::buildModel(
    std::shared_ptr<VoxelGeometry> voxelGeometry,
    std::shared_ptr<UnitConverter> uc) {
  std::shared_ptr<VoxelVolumeArray> volumes = voxelGeometry->getSensors();

  m_model = new QStandardItemModel(volumes->size(), 3);
  m_model->setHeaderData(0, Qt::Horizontal, tr("Geometry"));
  m_model->setHeaderData(1, Qt::Horizontal, tr("Temp."));
  m_model->setHeaderData(2, Qt::Horizontal, tr("Vol.Flow"));

  updateModel(voxelGeometry, uc);

  setModel(m_model);
  verticalHeader()->hide();
  resizeRowsToContents();
  resizeColumnsToContents();
}

void SensorOutputTableView::clear() {
  if (m_model && m_model->rowCount() > 0) m_model->clear();
}
