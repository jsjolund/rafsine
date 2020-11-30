"""
Generate direction vector definition for CUDA
"""
import sys

from ddqq import ei
from code_printer import AnyFile

src = AnyFile()
src.comment("Generated by lbm_gen_ddqq_indexing.py")

dfs = []
Tfs = []
for i in range(0, 19):
    x = 'x'
    y = 'y'
    z = 'z'
    if ei.row(i)[0] == 1:
        x = 'xm'
    if ei.row(i)[0] == -1:
        x = 'xp'
    if ei.row(i)[1] == 1:
        y = 'ym'
    if ei.row(i)[1] == -1:
        y = 'yp'
    if ei.row(i)[2] == 1:
        z = 'zm'
    if ei.row(i)[2] == -1:
        z = 'zp'
    dfs += [f'real_t f{i} = df3D({i},{x},{y},{z},nz,ny,nz);']
    if i < 7:
        Tfs += [f'real_t T{i} = Tdf3D({i},{x},{y},{z},nz,ny,nz);']
src.append('\n'.join(dfs) + '\n'.join(Tfs))

src.generate(sys.argv)

