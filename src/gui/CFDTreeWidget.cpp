#include "CFDTreeWidget.hpp"

CFDTreeWidget::CFDTreeWidget(QWidget *parent) : QTreeWidget(parent)
{
  setAlternatingRowColors(true);
  QStringList headers;
  headers << "Geometry"
          << "Details";
  setHeaderLabels(headers);
}

CFDTreeWidget::~CFDTreeWidget() { clear(); }

QString CFDTreeWidget::vecToQStr(vec3<real> vec)
{
  return QStringLiteral("(%1, %2, %3)").arg(vec.x).arg(vec.y).arg(vec.z);
}

void CFDTreeWidget::buildModel(std::shared_ptr<VoxelGeometry> voxelGeometry)
{

  std::vector<std::string> names = voxelGeometry->getGeometryNames();

  QList<QTreeWidgetItem *> items;

  for (std::string name : names)
  {
    QTreeWidgetItem *nameItem = new QTreeWidgetItem((QTreeWidget *)0, QStringList(QString::fromStdString(name)));

    std::unordered_set<VoxelQuad> quads = voxelGeometry->getQuadsByName(name);

    int i = 1;
    for (VoxelQuad quad : quads)
    {
      QTreeWidgetItem *quadItem = new QTreeWidgetItem((QTreeWidget *)0, {tr("quad"), QStringLiteral("%1").arg(i++)});
      quadItem->addChild(new QTreeWidgetItem((QTreeWidget *)0, {tr("origin"), vecToQStr(quad.m_origin)}));
      quadItem->addChild(new QTreeWidgetItem((QTreeWidget *)0, {tr("direction1"), vecToQStr(quad.m_dir1)}));
      quadItem->addChild(new QTreeWidgetItem((QTreeWidget *)0, {tr("direction2"), vecToQStr(quad.m_dir2)}));

      std::vector<BoundaryCondition> bcs;
      bcs.push_back(quad.m_bc);
      for (BoundaryCondition bc : quad.m_intersectingBcs)
        bcs.push_back(bc);

      for (BoundaryCondition bc : bcs)
      {
        QTreeWidgetItem *bcItem = new QTreeWidgetItem((QTreeWidget *)0, {tr("boundary"), QStringLiteral("%1").arg(bc.m_id)});
        std::stringstream ss;
        ss << bc.m_type;
        bcItem->addChild(new QTreeWidgetItem((QTreeWidget *)0, {tr("type"), QString::fromStdString(ss.str())}));
        bcItem->addChild(new QTreeWidgetItem((QTreeWidget *)0, {tr("normal"), vecToQStr(bc.m_normal)}));
        if (bc.m_type == VoxelType::INLET_RELATIVE || bc.m_type == VoxelType::INLET_CONSTANT || bc.m_type == VoxelType::INLET_ZERO_GRADIENT)
        {
          if (bc.m_type == VoxelType::INLET_RELATIVE || bc.m_type == VoxelType::INLET_CONSTANT)
            bcItem->addChild(new QTreeWidgetItem((QTreeWidget *)0, {tr("temperature"), QString::number(bc.m_temperature)}));
          if (bc.m_type == VoxelType::INLET_RELATIVE)
            bcItem->addChild(new QTreeWidgetItem((QTreeWidget *)0, {tr("relative pos."), vecToQStr(bc.m_rel_pos)}));
          bcItem->addChild(new QTreeWidgetItem((QTreeWidget *)0, {tr("velocity"), vecToQStr(bc.m_velocity)}));
        }
        quadItem->addChild(bcItem);
      }
      nameItem->addChild(quadItem);
    }
    items.append(nameItem);
  }
  insertTopLevelItems(0, items);
}