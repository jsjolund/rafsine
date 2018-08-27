#include "Voxel.hpp"

VoxelArray &VoxelArray::operator=(const VoxelArray &other)
{
  if (this == &other)
    return *this; // handling of self assignment
  delete[] data_; // freeing previously used memory
  data_ = new voxel[other.getFullSize()];
  sizeX_ = other.sizeX_;
  sizeY_ = other.sizeY_;
  sizeZ_ = other.sizeZ_;
  memcpy(data_, other.data_, sizeof(voxel) * getFullSize());
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
  fout << sizeX_ << "  " << sizeY_ << "  " << sizeZ_ << endl;
  //save the voxel data
  for (unsigned int k = 0; k < sizeZ_; k++)
  {
    for (unsigned int j = 0; j < sizeY_; j++)
    {
      for (unsigned int i = 0; i < sizeX_; i++)
      {
        fout << int((*this)(i, j, k));
        fout << "  ";
      }
      fout << endl;
    }
    fout << endl;
  }
  /*
    for(unsigned int k=0; k<sizeZ_; k++)
      for(unsigned int j=0; j<sizeY_; j++)
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
  if (tx >= int(sizeX_))
    return outside;
  if (ty >= int(sizeY_))
    return outside;
  if (tz >= int(sizeZ_))
    return outside;
  voxel data = data_[tx + ty * sizeX_ + tz * sizeX_ * sizeY_];
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
  if (tx >= int(sizeX_))
    return outside;
  if (ty >= int(sizeY_))
    return outside;
  if (tz >= int(sizeZ_))
    return outside;
  voxel data = data_[tx + ty * sizeX_ + tz * sizeX_ * sizeY_];
  return (data == VoxelType::Enum::EMPTY);
}

void VoxelArray::saveAutocrop(std::string filename)
{
#ifdef VERBOSE
  cout << "saving autocrop to file " << filename << " ... " << endl;
#endif
  //find the minimums and maximums
  int xmin = 0, xmax = sizeX_ - 1, ymin = 0, ymax = sizeY_ - 1, zmin = 0, zmax = sizeZ_ - 1;
  for (unsigned int i = 0; i < sizeX_; i++)
  {
    xmin = i;
    for (unsigned int j = 0; j < sizeY_; j++)
      for (unsigned int k = 0; k < sizeZ_; k++)
      {
        if ((*this)(i, j, k) != VoxelType::Enum::EMPTY)
        {
          //cout << i << "; " << j <<"; "<<k<<endl;
          goto label_xmin;
        }
      }
  }
label_xmin:
  for (int i = sizeX_ - 1; i >= 0; i--)
  {
    xmax = i;
    for (unsigned int j = 0; j < sizeY_; j++)
      for (unsigned int k = 0; k < sizeZ_; k++)
      {
        if ((*this)(i, j, k) != VoxelType::Enum::EMPTY)
        {
          //cout << i << "; " << j <<"; "<<k<<endl;
          goto label_xmax;
        }
      }
  }
label_xmax:
  for (unsigned int j = 0; j < sizeY_; j++)
  {
    ymin = j;
    for (unsigned int i = 0; i < sizeY_; i++)
      for (unsigned int k = 0; k < sizeZ_; k++)
      {
        if ((*this)(i, j, k) != VoxelType::Enum::EMPTY)
        {
          //cout << i << "; " << j <<"; "<<k<<endl;
          goto label_ymin;
        }
      }
  }
label_ymin:
  for (int j = sizeY_ - 1; j >= 0; j--)
  {
    ymax = j;
    for (unsigned int i = 0; i < sizeX_; i++)
      for (unsigned int k = 0; k < sizeZ_; k++)
      {
        if ((*this)(i, j, k) != VoxelType::Enum::EMPTY)
        {
          //cout << i << "; " << j <<"; "<<k<<endl;
          goto label_ymax;
        }
      }
  }
label_ymax:
  for (unsigned int k = 0; k < sizeZ_; k++)
  {
    zmin = k;
    for (unsigned int i = 0; i < sizeY_; i++)
      for (unsigned int j = 0; j < sizeY_; j++)
      {
        if ((*this)(i, j, k) != VoxelType::Enum::EMPTY)
        {
          //cout << i << "; " << j <<"; "<<k<<endl;
          goto label_zmin;
        }
      }
  }
label_zmin:
  for (int k = sizeZ_ - 1; k >= 0; k--)
  {
    zmax = k;
    for (unsigned int i = 0; i < sizeX_; i++)
      for (unsigned int j = 0; j < sizeY_; j++)
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
  if ((nx != sizeX_) || (ny != sizeY_) || (nz != sizeZ_))
  {
    cout << "x=" << nx << "=" << sizeX_ << endl;
    cout << "y=" << ny << "=" << sizeY_ << endl;
    cout << "z=" << nz << "=" << sizeZ_ << endl;
    FATAL_ERROR("Lattice sizes do not match.")
  }
  //load the voxel data
  int temp_block;
  for (unsigned int k = 0; k < sizeZ_; k++)
    for (unsigned int j = 0; j < sizeY_; j++)
      for (unsigned int i = 0; i < sizeX_; i++)
      {
        fin >> temp_block;
        (*this)(i, j, k) = (voxel)(temp_block);
      }
  fin.close();
#ifdef VERBOSE
  //cout << "done." << endl;
#endif
}
