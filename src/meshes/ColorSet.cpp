#include "ColorSet.hpp"

osg::Vec4 ColorSet::getRandomColor(unsigned int* seed) {
  return osg::Vec4((1.0 / RAND_MAX) * rand_r(seed),
                   (1.0 / RAND_MAX) * rand_r(seed),
                   (1.0 / RAND_MAX) * rand_r(seed), 1.0);
}

ColorSet::ColorSet() {
  // Default colors
  m_colorSet.clear();
  m_colorSet[0] = osg::Vec4(1., 1., 1., 1.);
  unsigned int seed = 0;
  for (int i = 1; i < MAX_COLORSET_SIZE; i++) {
    m_colorSet[i] = getRandomColor(&seed);
  }
}
