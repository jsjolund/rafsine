#include "Voxel.hpp"

VoxelArray &VoxelArray::operator=(const VoxelArray &other)
{
  if (this == &other)
    return *this; // handling of self assignment
  delete[] m_data; // freeing previously used memory
  m_data = new voxel[other.getFullSize()];
  m_sizeX = other.m_sizeX;
  m_sizeY = other.m_sizeY;
  m_sizeZ = other.m_sizeZ;
  memcpy(m_data, other.m_data, sizeof(voxel) * getFullSize());
  return *this;
}

void VoxelArray::saveToFile(std::string filename)
{
#ifdef VERBOSE
  cout << "saving to file " << filename << " ... " << endl;
#endif
  //saving
  std::ofstream fout(filename.c_str());
  //save the sizes
  fout << m_sizeX << "  " << m_sizeY << "  " << m_sizeZ << endl;
  //save the voxel data
  for (unsigned int k = 0; k < m_sizeZ; k++)
  {
    for (unsigned int j = 0; j < m_sizeY; j++)
    {
      for (unsigned int i = 0; i < m_sizeX; i++)
      {
        fout << int((*this)(i, j, k));
        fout << "  ";
      }
      fout << endl;
    }
    fout << endl;
  }
  /*
    for(unsigned int k=0; k<m_sizeZ; k++)
      for(unsigned int j=0; j<m_sizeY; j++)
        {
          fout << int( (*this)(8,j,k) );
          fout << "  ";
        }
    */
  fout.close();
#ifdef VERBOSE
  cout << "done. " << endl;
#endif
}

bool VoxelArray::isEmpty(unsigned int x, unsigned int y, unsigned int z) const
{
  bool outside = true;
  int tx = x;
  int ty = y;
  int tz = z;
  if (tx < 0)
    return outside;
  if (ty < 0)
    return outside;
  if (tz < 0)
    return outside;
  if (tx >= int(m_sizeX))
    return outside;
  if (ty >= int(m_sizeY))
    return outside;
  if (tz >= int(m_sizeZ))
    return outside;
  voxel data = m_data[tx + ty * m_sizeX + tz * m_sizeX * m_sizeY];
  return ((data == VoxelType::Enum::EMPTY) || (data == VoxelType::Enum::FLUID));
}

bool VoxelArray::isEmptyStrict(unsigned int x, unsigned int y, unsigned int z) const
{
  bool outside = true;
  int tx = x;
  int ty = y;
  int tz = z;
  if (tx < 0)
    return outside;
  if (ty < 0)
    return outside;
  if (tz < 0)
    return outside;
  if (tx >= int(m_sizeX))
    return outside;
  if (ty >= int(m_sizeY))
    return outside;
  if (tz >= int(m_sizeZ))
    return outside;
  voxel data = m_data[tx + ty * m_sizeX + tz * m_sizeX * m_sizeY];
  return (data == VoxelType::Enum::EMPTY);
}

void VoxelArray::saveAutocrop(std::string filename)
{
#ifdef VERBOSE
  cout << "saving autocrop to file " << filename << " ... " << endl;
#endif
  //find the minimums and maximums
  int xmin = 0, xmax = m_sizeX - 1, ymin = 0, ymax = m_sizeY - 1, zmin = 0, zmax = m_sizeZ - 1;
  for (unsigned int i = 0; i < m_sizeX; i++)
  {
    xmin = i;
    for (unsigned int j = 0; j < m_sizeY; j++)
      for (unsigned int k = 0; k < m_sizeZ; k++)
      {
        if ((*this)(i, j, k) != VoxelType::Enum::EMPTY)
        {
          //cout << i << "; " << j <<"; "<<k<<endl;
          goto label_xmin;
        }
      }
  }
label_xmin:
  for (int i = m_sizeX - 1; i >= 0; i--)
  {
    xmax = i;
    for (unsigned int j = 0; j < m_sizeY; j++)
      for (unsigned int k = 0; k < m_sizeZ; k++)
      {
        if ((*this)(i, j, k) != VoxelType::Enum::EMPTY)
        {
          //cout << i << "; " << j <<"; "<<k<<endl;
          goto label_xmax;
        }
      }
  }
label_xmax:
  for (unsigned int j = 0; j < m_sizeY; j++)
  {
    ymin = j;
    for (unsigned int i = 0; i < m_sizeY; i++)
      for (unsigned int k = 0; k < m_sizeZ; k++)
      {
        if ((*this)(i, j, k) != VoxelType::Enum::EMPTY)
        {
          //cout << i << "; " << j <<"; "<<k<<endl;
          goto label_ymin;
        }
      }
  }
label_ymin:
  for (int j = m_sizeY - 1; j >= 0; j--)
  {
    ymax = j;
    for (unsigned int i = 0; i < m_sizeX; i++)
      for (unsigned int k = 0; k < m_sizeZ; k++)
      {
        if ((*this)(i, j, k) != VoxelType::Enum::EMPTY)
        {
          //cout << i << "; " << j <<"; "<<k<<endl;
          goto label_ymax;
        }
      }
  }
label_ymax:
  for (unsigned int k = 0; k < m_sizeZ; k++)
  {
    zmin = k;
    for (unsigned int i = 0; i < m_sizeY; i++)
      for (unsigned int j = 0; j < m_sizeY; j++)
      {
        if ((*this)(i, j, k) != VoxelType::Enum::EMPTY)
        {
          //cout << i << "; " << j <<"; "<<k<<endl;
          goto label_zmin;
        }
      }
  }
label_zmin:
  for (int k = m_sizeZ - 1; k >= 0; k--)
  {
    zmax = k;
    for (unsigned int i = 0; i < m_sizeX; i++)
      for (unsigned int j = 0; j < m_sizeY; j++)
      {
        if ((*this)(i, j, k) != VoxelType::Enum::EMPTY)
        {
          //cout << i << "; " << j <<"; "<<k<<endl;
          goto label_zmax;
        }
      }
  }
label_zmax:
  /*
    cout << xmin << endl;
    cout << xmax << endl;
    cout << ymin << endl;
    cout << ymax << endl;
    cout << zmin << endl;
    cout << zmax << endl;
    */
  //saving
  std::ofstream fout(filename.c_str());
  //save the sizes
  fout << (xmax - xmin + 1) << "  " << (ymax - ymin + 1) << "  " << (zmax - zmin + 1) << endl;
  //save the voxel data
  for (int k = zmin; k <= zmax; k++)
  {
    for (int j = ymin; j <= ymax; j++)
    {
      for (int i = xmin; i <= xmax; i++)
      {
        fout << int((*this)(i, j, k));
        fout << "  ";
      }
      fout << endl;
    }
    fout << endl;
  }
  fout.close();
#ifdef VERBOSE
  cout << "done. " << endl;
#endif
}

void VoxelArray::loadFromFile(std::string filename)
{
#ifdef VERBOSE
//cout << "loading from file " << filename << " ..." << endl;
#endif
  //loading
  std::ifstream fin(filename.c_str());
  //load the size
  unsigned int nx, ny, nz;
  fin >> nx >> ny >> nz;
  if ((nx != m_sizeX) || (ny != m_sizeY) || (nz != m_sizeZ))
  {
    cout << "x=" << nx << "=" << m_sizeX << endl;
    cout << "y=" << ny << "=" << m_sizeY << endl;
    cout << "z=" << nz << "=" << m_sizeZ << endl;
    FATAL_ERROR("Lattice sizes do not match.")
  }
  //load the voxel data
  int temp_block;
  for (unsigned int k = 0; k < m_sizeZ; k++)
    for (unsigned int j = 0; j < m_sizeY; j++)
      for (unsigned int i = 0; i < m_sizeX; i++)
      {
        fin >> temp_block;
        (*this)(i, j, k) = (voxel)(temp_block);
      }
  fin.close();
#ifdef VERBOSE
  //cout << "done." << endl;
#endif
}
