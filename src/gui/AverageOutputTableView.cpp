#include "AverageOutputTableView.hpp"

void AverageOutputTableModel::update(const AverageMatrix& avgs) {
  std::unordered_map<std::string, Average> avgsByName;
  for (int row = 0; row < avgs.m_rows.back().m_measurements.size(); row++) {
    Average avg = avgs.m_rows.back().m_measurements.at(row);
    avgsByName[avgs.m_columns.at(row)] = avg;
  }
  for (int row = 0; row < rowCount(); row++) {
    // Read the name
    std::string name =
        data(index(row, AVG_NAME_COL_IDX)).toString().toUtf8().constData();
    Average avg = avgsByName[name];
    for (int col = 0; col < 3; col++) {
      QStandardItem* cell = item(row, col);
      if (col == AVG_TEMP_COL_IDX)
        cell->setText(QString::number(avg.temperature));
      else if (col == AVG_FLOW_COL_IDX)
        cell->setText(QString::number(avg.flow));
    }
  }
  QModelIndex topLeft = index(0, 0);
  QModelIndex bottomRight = index(rowCount() - 1, columnCount() - 1);
  emit dataChanged(topLeft, bottomRight);
  emit layoutChanged();
}

AverageOutputTableView::AverageOutputTableView(QWidget* parent)
    : QTableView(parent), m_model(nullptr) {
  setAlternatingRowColors(true);
  setEditTriggers(QAbstractItemView::NoEditTriggers);
}

AverageOutputTableView::~AverageOutputTableView() {}

void AverageOutputTableView::buildModel(const VoxelVolumeArray& volumes) {
  m_model = new AverageOutputTableModel(volumes.size(), 3);
  m_model->setHeaderData(AVG_NAME_COL_IDX, Qt::Horizontal,
                         tr(AVG_NAME_COL_TITLE));
  m_model->setHeaderData(AVG_TEMP_COL_IDX, Qt::Horizontal,
                         tr(AVG_TEMP_COL_TITLE));
  m_model->setHeaderData(AVG_FLOW_COL_IDX, Qt::Horizontal,
                         tr(AVG_FLOW_COL_TITLE));

  for (int row = 0; row < volumes.size(); row++) {
    std::string name = volumes.at(row).getName();
    real tempC = NaN;
    real flow = NaN;
    m_model->setItem(row, AVG_NAME_COL_IDX,
                     new QStandardItem(QString::fromStdString(name)));
    m_model->setItem(row, AVG_TEMP_COL_IDX,
                     new QStandardItem(QString::number(tempC)));
    m_model->setItem(row, AVG_FLOW_COL_IDX,
                     new QStandardItem(QString::number(flow)));
  }

  setModel(m_model);
  verticalHeader()->hide();
  resizeRowsToContents();
  setSortingEnabled(true);
  horizontalHeader()->setSortIndicator(0, Qt::AscendingOrder);
  horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
}

void AverageOutputTableView::clear() {
  if (m_model && m_model->rowCount() > 0) m_model->clear();
}

void AverageOutputTableView::notify(const AverageMatrix& avgs) {
  m_model->update(avgs);
  viewport()->update();
}
