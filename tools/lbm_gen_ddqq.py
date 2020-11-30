"""
Generate direction vector definition for CUDA
"""
from ddqq import ei, d3q19_weights, d3q7_weights

# Vectors for host usage
D3Q27vectorsStr = ['const Vector3<int> D3Q27vectors[27] = {']
for i in range(0, 27):
    if i == 0:
        D3Q27vectorsStr += ['// Origin']
    elif i == 1:
        D3Q27vectorsStr += ['// 6 faces']
    elif i == 7:
        D3Q27vectorsStr += ['// 12 edges']
    elif i == 19:
        D3Q27vectorsStr += ['// 8 corners']
    if i < 26:
        D3Q27vectorsStr += [
            f'Vector3<int>({ei.row(i)[0]}, {ei.row(i)[1]}, {ei.row(i)[2]}),  // {i}']
    else:
        D3Q27vectorsStr += [
            f'Vector3<int>({ei.row(i)[0]}, {ei.row(i)[1]}, {ei.row(i)[2]})  // {i}']
D3Q27vectorsStr += ['};']
print('\n'.join(D3Q27vectorsStr))

# Vectors for CUDA usage
D3Q27Str = ['__constant__ real_t D3Q27[81] = {']
for i in range(0, 27):
    if i == 0:
        D3Q27Str += ['// Origin']
    elif i == 1:
        D3Q27Str += ['// 6 faces']
    elif i == 7:
        D3Q27Str += ['// 12 edges']
    elif i == 19:
        D3Q27Str += ['// 8 corners']
    if i < 26:
        D3Q27Str += [f'{ei.row(i)[0]}, {ei.row(i)[1]}, {ei.row(i)[2]},  // {i}']
    else:
        D3Q27Str += [f'{ei.row(i)[0]}, {ei.row(i)[1]}, {ei.row(i)[2]}  // {i}']
D3Q27Str += ['};']

print('\n'.join(D3Q27Str))

# Indices sorted by positive/negative axis
D3Q27ranksStr = ['const unsigned int D3Q27ranks[7][9] = {']
D3Q27ranksStr += ['{0, 0, 0, 0, 0, 0, 0, 0, 0}, // padding']
px = []
nx = []
py = []
ny = []
pz = []
nz = []
for i in range(0, 27):
    if ei.row(i)[0] == 1:
        px += [str(i)]
    if ei.row(i)[0] == -1:
        nx += [str(i)]
    if ei.row(i)[1] == 1:
        py += [str(i)]
    if ei.row(i)[1] == -1:
        ny += [str(i)]
    if ei.row(i)[2] == 1:
        pz += [str(i)]
    if ei.row(i)[2] == -1:
        nz += [str(i)]
D3Q27ranksStr += ['{' + ','.join(px) + '}, // positive x-axis']
D3Q27ranksStr += ['{' + ','.join(nx) + '}, // negative x-axis']
D3Q27ranksStr += ['{' + ','.join(py) + '}, // positive y-axis']
D3Q27ranksStr += ['{' + ','.join(ny) + '}, // negative y-axis']
D3Q27ranksStr += ['{' + ','.join(pz) + '}, // positive z-axis']
D3Q27ranksStr += ['{' + ','.join(nz) + '} // negative z-axis']
D3Q27ranksStr += ['};']
print('\n'.join(D3Q27ranksStr))

# Opposing vectors
D3Q27OppositeStr = ['__constant__ unsigned int D3Q27Opposite[27] = {']
opp = []
for i in range(0, 27):
  eiv = ei.row(i)
  for j in range(0, 27):
    eiu = ei.row(i)
    if eiv == -eiu:
      