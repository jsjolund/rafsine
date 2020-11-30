#!/bin/bash

python lbm_gen_ddqq.py ../src/sim/DdQq.cu
python lbm_gen_ddqq_idx.py ../include/DdQqIndexing.hpp

python3 lbm_gen_double_bgk_les_boussinesq.py ../include/LBM_BGK.hpp
python3 lbm_gen_double_mrt_les_boussinesq.py ../include/LBM_MRT.hpp
