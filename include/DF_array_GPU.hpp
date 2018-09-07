/**
  \file
      ArrayGPU.h
  \brief
      Helper class to define array on the GPU.
	  Uses thrust vectors as a basis.
  \author
      Nicolas Delbosc, University of Leeds
*/

#pragma once

#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/copy.h>
#include <thrust/fill.h>

#include "CudaUtils.hpp"

//Define an group of array on the GPU
//designed for 3D arrays (but works in 2D as well)
//Useful to group all the distribution functions into a single array
//distribution functions (fi) are packed in memory based on their direction:
// memory: f1,f1,...,f1,f2,f2,...,f2,f3,f3,...
class DistributionFunctionsGroup
{
private:
  //number of arrays (or directions for distribution functions)
  const unsigned int Q_;
  //number of nodes along each axis
  const unsigned int sizeX_;
  const unsigned int sizeY_;
  const unsigned int sizeZ_;
  //distribution functions on the GPU
  thrust::device_vector<real> dfGPU_;
  //true if the arrays are also allocated and usable on the CPU
  bool useCPU_;
  //distribution functions on the CPU
  thrust::host_vector<real> dfCPU_;

public:
  //Constructor
  DistributionFunctionsGroup(unsigned int Q,
                             unsigned int sizeX,
                             unsigned int sizeY,
                             unsigned int sizeZ = 1,
                             bool useCPU = true)
      : Q_(Q), sizeX_(sizeX), sizeY_(sizeY), sizeZ_(sizeZ), useCPU_(useCPU),
        dfCPU_(useCPU ? Q * sizeX * sizeY * sizeZ : 1),
        dfGPU_(Q * sizeX * sizeY * sizeZ)
  {
  }

  //return the amount of memory used by the group of arrays
  //notes: same amount on both CPU and GPU
  inline unsigned long int memoryUse()
  {
    return sizeof(real) * Q_ * sizeX_ * sizeY_ * sizeZ_;
  }

  //return the number of arrays in the group
  //i.e. the number of distribution functions
  inline unsigned int Q() { return Q_; }

  //return size of the lattice
  inline unsigned int sizeX() { return sizeX_; }
  inline unsigned int sizeY() { return sizeY_; }
  inline unsigned int sizeZ() { return sizeZ_; }
  inline unsigned int fullSize() { return sizeX_ * sizeY_ * sizeZ_; }

  //1D access to distribution function on the CPU
  inline real &operator()(unsigned int df_idx, unsigned int idx)
  {
    if (useCPU_)
      return dfCPU_[idx + df_idx * sizeX_ * sizeY_ * sizeZ_];
    else
    {
      std::cerr << "Error in 'DistributionFunctionsGroup::operator()'  CPU memory is not allocated." << std::endl;
      return dfCPU_[0];
    }
  }
  //3D access to distribution function on the CPU
  inline real &operator()(unsigned int df_idx, unsigned int x, unsigned int y, unsigned int z = 0)
  {
    return (*this)(df_idx, x + y * sizeX_ + z * sizeX_ * sizeY_);
  }

  //upload the distributions functions from the CPU to the GPU
  inline DistributionFunctionsGroup &upload()
  {
    if (useCPU_)
      dfGPU_ = dfCPU_;
    else
      std::cerr << "Error in 'DistributionFunctionsGroup::upload()'  CPU memory is not allocated." << std::endl;
    return *this;
  }
  //download the distributions functions from the GPU to the CPU
  inline DistributionFunctionsGroup &download()
  {
    if (useCPU_)
      dfCPU_ = dfGPU_;
    else
      std::cerr << "Error in 'DistributionFunctionsGroup::upload()'  CPU memory is not allocated." << std::endl;
    return *this;
  }
  //return a pointer to the beggining of the GPU memory
  inline real *gpu_ptr(unsigned int df_idx = 0)
  {
    const int SIZE = sizeX_ * sizeY_ * sizeZ_;
    return thrust::raw_pointer_cast(&(dfGPU_)[df_idx * SIZE]);
  }

  //copy from another group of distribution functions
  //SAME SIZE IS REQUIRED
  inline DistributionFunctionsGroup &operator=(const DistributionFunctionsGroup &f)
  {
    if ((Q_ == f.Q_) && (sizeX_ == f.sizeX_) && (sizeY_ == f.sizeY_) && (sizeZ_ == f.sizeZ_))
    {
      if (useCPU_)
        thrust::copy(f.dfCPU_.begin(), f.dfCPU_.end(), dfCPU_.begin());
      thrust::copy(f.dfGPU_.begin(), f.dfGPU_.end(), dfGPU_.begin());
    }
    else
    {
      std::cerr << "Error in 'DistributionFunctionsGroup::operator ='  sizes do not match." << std::endl;
    }
    useCPU_ = f.useCPU_;
    return *this;
  }

  //static function to swap two DistributionFunctionsGroup
  static inline void swap(DistributionFunctionsGroup &f1, DistributionFunctionsGroup &f2)
  {
    f1.dfCPU_.swap(f2.dfCPU_);
    f1.dfGPU_.swap(f2.dfGPU_);
  }

  //fill the ith array, i.e. the ith distribution function
  //with a constant value for all nodes
  inline void fill(unsigned int df_idx, real value)
  {
    const int SIZE = sizeX_ * sizeY_ * sizeZ_;
    if (useCPU_)
      thrust::fill(dfCPU_.begin() + df_idx * SIZE, dfCPU_.begin() + (df_idx + 1) * SIZE, value);
    thrust::fill(dfGPU_.begin() + df_idx * SIZE, dfGPU_.begin() + (df_idx + 1) * SIZE, value);
  }

  //return an iterator to the beggining of the ith distribution function (GPU)
  inline thrust::device_vector<real>::iterator begin(unsigned int df_idx)
  {
    const int SIZE = sizeX_ * sizeY_ * sizeZ_;
    return dfGPU_.begin() + df_idx * SIZE;
  }

  //return an iterator to the end of the ith distribution function (GPU)
  inline thrust::device_vector<real>::iterator end(unsigned int df_idx)
  {
    const int SIZE = sizeX_ * sizeY_ * sizeZ_;
    return dfGPU_.begin() + (df_idx + 1) * SIZE;
  }
  //Clear the arrays (free the memory)
  inline void clear()
  {
    dfCPU_.clear();
    dfCPU_.shrink_to_fit();
    dfGPU_.clear();
    dfGPU_.shrink_to_fit();
  }
};
