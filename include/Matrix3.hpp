#include "CudaUtils.hpp"

/**
 * @brief Simple 3x3 integer matrix for coordinate transform
 */
class Matrix3 {
 private:
  int m_val[9];
  enum {
    M00 = 0,
    M01 = 3,
    M02 = 6,
    M10 = 1,
    M11 = 4,
    M12 = 7,
    M20 = 2,
    M21 = 5,
    M22 = 8
  };

 public:
  /**
   * @brief Set matrix to identity
   */
  void idt() {
    m_val[M00] = 1;
    m_val[M10] = 0;
    m_val[M20] = 0;
    m_val[M01] = 0;
    m_val[M11] = 1;
    m_val[M21] = 0;
    m_val[M02] = 0;
    m_val[M12] = 0;
    m_val[M22] = 1;
  }

  /**
   * @brief Swap two rows A and B
   */
  void swapRows(int a, int b) {
    int tmp[3];
    tmp[0] = m_val[b * 3];
    tmp[1] = m_val[b * 3 + 1];
    tmp[2] = m_val[b * 3 + 2];
    m_val[b * 3] = m_val[a * 3];
    m_val[b * 3 + 1] = m_val[a * 3 + 1];
    m_val[b * 3 + 2] = m_val[a * 3 + 2];
    m_val[a * 3] = tmp[0];
    m_val[a * 3 + 1] = tmp[1];
    m_val[a * 3 + 2] = tmp[2];
  }

  /**
   * @brief Right-multiply this matrix by a column vector and store in input
   * variables
   */
  CUDA_CALLABLE_MEMBER void mulVec(int *x, int *y, int *z) const {
    int tx = *x, ty = *y, tz = *z;
    *x = m_val[M00] * tx + m_val[M10] * ty + m_val[M20] * tz;
    *y = m_val[M01] * tx + m_val[M11] * ty + m_val[M21] * tz;
    *z = m_val[M02] * tx + m_val[M12] * ty + m_val[M22] * tz;
  }

  /**
   * @brief Transpose the matrix
   */
  void transpose() {
    int v01 = m_val[M10];
    int v02 = m_val[M20];
    int v10 = m_val[M01];
    int v12 = m_val[M21];
    int v20 = m_val[M02];
    int v21 = m_val[M12];
    m_val[M01] = v01;
    m_val[M02] = v02;
    m_val[M10] = v10;
    m_val[M12] = v12;
    m_val[M20] = v20;
    m_val[M21] = v21;
  }

  /**
   * @brief Construct a new identity matrix
   */
  Matrix3() { idt(); }
};
