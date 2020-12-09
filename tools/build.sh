#!/bin/bash

python3 lbm_gen_ddqq.py ../src/sim/DdQq.cu
python3 lbm_gen_ddqq_idx.py ../include/DdQqIndexing.hpp

python3 lbm_gen_init.py ../src/kernel/InitKernel.cu
python3 lbm_gen_double_bgk_les_boussinesq.py ../include/LBM_BGK.hpp
python3 lbm_gen_double_mrt_les_boussinesq.py ../include/LBM_MRT.hpp
python3 lbm_gen_d3q27_mrt.py ../include/LBM_MRT27.hpp
