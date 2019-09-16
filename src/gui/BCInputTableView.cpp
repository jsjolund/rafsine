#include "BCInputTableView.hpp"

Qt::ItemFlags BCInputTableModel::flags(const QModelIndex &index) const {
  Qt::ItemFlags flags;

  flags = QStandardItemModel::flags(index);

  if (index.column() == 0) {
    flags &= ~Qt::ItemIsEditable;
    return flags;
  }
  return QStandardItemModel::flags(index);
}

QWidget *BCInputTableDelegate::createEditor(QWidget *parent,
                                            const QStyleOptionViewItem &option,
                                            const QModelIndex &index) const {
  QLineEdit *lineEdit = new QLineEdit(parent);
  QDoubleValidator *validator = new QDoubleValidator();
  validator->setNotation(QDoubleValidator::ScientificNotation);
  lineEdit->setValidator(validator);
  int row = index.row();
  int column = index.column();
  connect(lineEdit, SIGNAL(editingFinished()), m_mainWindow,
          SLOT(onTableEdited()));
  return lineEdit;
}

void BCInputTableView::updateBoundaryConditions(
    std::shared_ptr<BoundaryConditions> bcs,
    std::shared_ptr<VoxelGeometry> voxelGeometry,
    std::shared_ptr<UnitConverter> uc) {
  QModelIndex parent = QModelIndex();
  // Loop over each named geometry in table
  for (int row = 0; row < m_model->rowCount(parent); ++row) {
    QModelIndex nameIndex = m_model->index(row, 0, parent);
    std::string name = m_model->data(nameIndex).toString().toUtf8().constData();

    // Read the temperature set by the user
    QModelIndex tempIndex = m_model->index(row, 1, parent);
    double tempPhys = m_model->data(tempIndex).toDouble();

    // Read the volumetric flow set by the user
    QModelIndex flowIndex = m_model->index(row, 2, parent);
    double flowPhys = m_model->data(flowIndex).toDouble();

    // Each named geometry can be composed of many quads
    std::unordered_set<VoxelQuad> quads = voxelGeometry->getQuadsByName(name);
    for (VoxelQuad quad : quads) {
      // Set boundary condition
      BoundaryCondition *bc = &(bcs->at(quad.m_bc.m_id));
      bc->setTemperature(*uc, tempPhys);
      bc->setFlow(*uc, flowPhys, quad.getAreaReal());
      std::cout << quad.getAreaReal() << ", " << quad.getAreaDiscrete(*uc)
                << std::endl;
    }
  }
}

int BCInputTableView::updateModel(std::shared_ptr<VoxelGeometry> voxelGeometry,
                                  std::shared_ptr<UnitConverter> uc) {
  std::vector<std::string> names = voxelGeometry->getGeometryNames();
  int row = 0;
  for (int i = 0; i < names.size(); i++) {
    std::string name = names.at(i);
    std::unordered_set<VoxelQuad> quads = voxelGeometry->getQuadsByName(name);

    // A geometry may consist of different boundary conditions, with different
    // temps and velocities set at the start. Using this table sets them all to
    // the same, scaled according to their area
    for (VoxelQuad quad : quads) {
      if (quad.m_bc.m_type == VoxelType::INLET_CONSTANT ||
          quad.m_bc.m_type == VoxelType::INLET_RELATIVE) {
        QStandardItem *nameItem =
            new QStandardItem(QString::fromStdString(name));
        m_model->setItem(row, 0, nameItem);

        // Set temperature cell
        real tempC = quad.m_bc.getTemperature(*uc);
        m_model->setItem(row, 1, new QStandardItem(QString::number(tempC)));

        // Set volumetric flow rate cell
        real flow = quad.m_bc.getFlow(*uc, quad.getNumVoxels());
        m_model->setItem(row, 2, new QStandardItem(QString::number(flow)));

        row++;
        break;
      }
    }
  }
  return row;
}

void BCInputTableView::buildModel(std::shared_ptr<VoxelGeometry> voxelGeometry,
                                  std::shared_ptr<UnitConverter> uc) {
  std::vector<std::string> names = voxelGeometry->getGeometryNames();

  m_model = new BCInputTableModel(names.size(), 3);
  m_model->setHeaderData(0, Qt::Horizontal, tr("Geometry"));
  m_model->setHeaderData(1, Qt::Horizontal, tr("Temp."));
  m_model->setHeaderData(2, Qt::Horizontal, tr("Vol.Flow"));

  int rowsUpdated = updateModel(voxelGeometry, uc);
  m_model->removeRows(rowsUpdated, names.size() - rowsUpdated);

  setModel(m_model);
  verticalHeader()->hide();
  resizeRowsToContents();
  resizeColumnsToContents();
}

void BCInputTableView::clear() {
  if (m_model && m_model->rowCount() > 0) m_model->clear();
}

void BCInputTableView::mousePressEvent(QMouseEvent *event) {
  if (event->button() == Qt::LeftButton) {
    QModelIndex index = indexAt(event->pos());
    if (index.column() == 1 ||
        index.column() == 2) {  // column you want to use for one click
      edit(index);
    }
  }
  QTableView::mousePressEvent(event);
}

BCInputTableView::BCInputTableView(QWidget *mainWindow)
    : QTableView(mainWindow), m_model(nullptr) {
  setAlternatingRowColors(true);
  setItemDelegate(new BCInputTableDelegate(mainWindow));
}

BCInputTableView::~BCInputTableView() {}
