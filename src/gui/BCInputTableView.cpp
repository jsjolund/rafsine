#include "BCInputTableView.hpp"

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

Qt::ItemFlags BCInputTableModel::flags(const QModelIndex &index) const {
  Qt::ItemFlags flags;

  flags = QStandardItemModel::flags(index);

  if (index.column() == 0) {
    flags &= ~Qt::ItemIsEditable;
    return flags;
  }
  return QStandardItemModel::flags(index);
}

BCInputTableView::BCInputTableView(QWidget *mainWindow)
    : QTableView(mainWindow), m_model(nullptr) {
  setAlternatingRowColors(true);
  setItemDelegate(new BCInputTableDelegate(mainWindow));
}

BCInputTableView::~BCInputTableView() {}

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
    real tempLu = uc->Temp_to_lu(tempPhys);

    // Read the volumetric flow set by the user
    QModelIndex flowIndex = m_model->index(row, 2, parent);
    double qPhys = m_model->data(flowIndex).toDouble();

    // Each named geometry can be composed of many quads
    std::unordered_set<VoxelQuad> quads = voxelGeometry->getQuadsByName(name);
    for (VoxelQuad quad : quads) {
      // Set temperature
      BoundaryCondition *bc = &(bcs->at(quad.m_bc.m_id));
      bc->m_temperature = tempLu;

      // Set velocity
      real velocityLu = max(0.0, uc->Q_to_Ulu(qPhys, quad.getAreaReal()));
      glm::vec3 nVelocity = glm::normalize(
          glm::vec3(bc->m_normal.x, bc->m_normal.y, bc->m_normal.z));
      if (bc->m_type == VoxelType::INLET_ZERO_GRADIENT) nVelocity = -nVelocity;
      bc->m_velocity.x = nVelocity.x * velocityLu;
      bc->m_velocity.y = nVelocity.y * velocityLu;
      bc->m_velocity.z = nVelocity.z * velocityLu;
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
        real tempC = uc->luTemp_to_Temp(quad.m_bc.m_temperature);
        QStandardItem *tempItem = new QStandardItem(QString::number(tempC));
        m_model->setItem(row, 1, tempItem);

        // Set volumetric flow rate cell
        real flow =
            uc->Ulu_to_Q(quad.m_bc.m_velocity.norm(), quad.getAreaVoxel());
        QStandardItem *flowItem = new QStandardItem(QString::number(flow));
        m_model->setItem(row, 2, flowItem);

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
