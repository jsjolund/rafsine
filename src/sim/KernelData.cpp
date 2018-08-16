#include "KernelData.hpp"

KernelData::KernelData(UnitConverter *uc, SimConstants *sc,
                       UserConstants *c, VoxelGeometry *vox)
    : uc(uc), sc(sc), c(c), vox(vox), geo(new std::vector<VoxelGeometryGroup *>())
{
}

KernelData::KernelData()
    : uc(new UnitConverter()),
      c(new UserConstants()),
      geo(new std::vector<VoxelGeometryGroup *>())
{
  // // reference length in meters
  // uc->m_ref_L_phys = 6.95;
  // // reference length in number of nodes
  // uc->m_ref_L_lbm = 256;
  // // reference speed in meter/second
  // uc->m_ref_U_phys = 1.0;
  // // reference speed in lattice units
  // uc->m_ref_U_lbm = 0.03;
  // // temperature conversion factor
  // uc->m_C_Temp = 1;
  // // reference temperature for Boussinesq in degrees Celsius
  // uc->m_T0_phys = 0;
  // uc->m_T0_lbm = 0;

  // sc = new SimConstants(uc);
  // // Size of the lattice
  // sc->mx = 6.95;
  // sc->my = 6.4;
  // sc->mz = 3.1;
  // // Kinematic viscosity of air
  // sc->nu = 1.568e-5;
  // // Thermal diffusivity
  // sc->nuT = 1.0e-2;
  // // Smagorinsky constant
  // sc->C = 0.02;
  // // Thermal conductivity
  // sc->k = 2.624e-5;
  // // Prandtl number of air
  // sc->Pr = 0.707;
  // // Turbulent Prandtl number
  // sc->Pr_t = 0.9;
  // // Gravity * thermal expansion
  // sc->gBetta = 9.82 * 3.32e-3;
  // // Initial temperature
  // sc->Tinit = 30;
  // // Reference temperature
  // sc->Tref = sc->Tinit;

  // (*c)["cracX"] = "0.510";
  // (*c)["cracY"] = "1.225";
  // (*c)["cracZ"] = "2.55";
  // (*c)["cracOutletY"] = "1.00";
  // (*c)["cracOutletZoffset"] = "0.1";
  // (*c)["cracOutletZ"] = "1.875 - cracOutletZoffset";

  // vox = new VoxelGeometry(sc->nx(), sc->ny(), sc->nz(), uc);

  // VoxelGeometryGroup wallQuads("Walls");
  // VoxelGeometryQuad xmin = vox->addWallXmin();
  // wallQuads.objs->push_back(&xmin);
  // VoxelGeometryQuad ymin = vox->addWallYmin();
  // wallQuads.objs->push_back(&ymin);
  // VoxelGeometryQuad zmin = vox->addWallZmin();
  // wallQuads.objs->push_back(&zmin);
  // VoxelGeometryQuad xmax = vox->addWallXmax();
  // wallQuads.objs->push_back(&xmax);
  // VoxelGeometryQuad ymax = vox->addWallYmax();
  // wallQuads.objs->push_back(&ymax);
  // VoxelGeometryQuad zmax = vox->addWallZmax();
  // wallQuads.objs->push_back(&zmax);

  // // vec3<std::string> testPoint("mx", "my", "mz");
  // // std::cout << testPoint << std::endl;

  // VoxelGeometryGroup cracGeo("CRAC01");
  // VoxelGeometryBox box("TestBox", vec3<real>(1, 2, 0), vec3<real>(3, 4, 2));
  // vox->addSolidBox(&box);
  // cracGeo.objs->push_back(&box);
  // VoxelGeometryQuad quad("TestQuad",
  //                        NodeMode::Enum::OVERWRITE,
  //                        vec3<real>(1.5, 2, 0.5),
  //                        vec3<real>(1.2, 0, 0),
  //                        vec3<real>(0, 0, 1.2),
  //                        vec3<int>(0, 1, 0),
  //                        VoxelType::Enum::INLET_CONSTANT,
  //                        10,
  //                        vec3<int>(0, 1, 0));
  // vox->addQuadBC(&quad);
  // cracGeo.objs->push_back(&quad);

  // geo->push_back(&wallQuads);
  // geo->push_back(&cracGeo);
}