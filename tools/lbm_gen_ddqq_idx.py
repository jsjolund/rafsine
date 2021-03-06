"""
Generate direction vector definition for CUDA
"""
import sys

from ddqq import ei
from code_printer import AnyFile

src = AnyFile()
src.header(f'// Generated by {sys.argv[0]}')
src.append(
"""
// Store streamed distribution functions in registers
// Modulo with wraparound for negative numbers
const int xp = ((x + 1) % nx + nx) % nx;
// x minus 1
const int xm = ((x - 1) % nx + nx) % nx;
// y plus 1
const int yp = ((y + 1) % ny + ny) % ny;
// y minus 1
const int ym = ((y - 1) % ny + ny) % ny;
// z plus 1
const int zp = ((z + 1) % nz + nz) % nz;
// z minus 1
const int zm = ((z - 1) % nz + nz) % nz;
""")

src.append('int index[27];')
dfs = []
for i in range(0, 27):
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
    dfs += [f'index[{i}] = I4D({i},{x},{y},{z},nx,ny,nz);']
src.append('\n'.join(dfs))


src.generate(sys.argv)

