import re
import subprocess
from pathlib import Path
from sympy import Matrix, ccode
from sympy.printing.ccode import C99CodePrinter


class CodePrinter(C99CodePrinter):
    """Code printer for CUDA LBM kernel generator"""

    def __init__(self, name, fp=True):
        super().__init__()
        self.name = name
        self.parameters = []
        self.rows = []
        self.fp = fp

    def parameter(self, *var, type='real'):
        """Add a function parameter"""
        for v in var:
            if isinstance(v, Matrix):
                self.parameters += [
                    f'{type} {f", {type} ".join([str(v.row(i)[0]) for i in range(0, v.shape[0])])}']
            else:
                self.parameters += [f'{type} {v}']

    def define(self, *var, type='real'):
        """Define a variable"""
        for v in var:
            if isinstance(v, Matrix):
                self.rows += [
                    f'{type} {", ".join([str(v.row(i)[0]) for i in range(0, v.shape[0])])};']
            else:
                self.rows += [f'{type} {v};']

    def comment(self, expr):
        """Append a comment"""
        self.rows += [f'\n// {expr}']

    def append(self, expr):
        """Append an expression from string"""
        self.rows += [expr]

    def let(self, var, expr):
        """Assign a variable"""
        custom_functions = {"Pow": "powf"} if self.fp else {}
        if isinstance(var, Matrix):
            for i in range(0, var.shape[0]):
                self.rows += [ccode(expr.row(i)[0], assign_to=var.row(i)
                                    [0], user_functions=custom_functions)]
        else:
            self.rows += [ccode(expr, assign_to=var,
                                user_functions=custom_functions)]

    def __repr__(self):
        src = '#pragma once\n' \
            + '#include "CudaUtils.hpp"\n' \
            + '#include "PhysicalQuantity.hpp"\n' \
            + f'__device__ __forceinline__ void {self.name}(' \
            + ', '.join(self.parameters) + ') {\n' \
            + '\n'.join(self.rows) + '\n' + '}' + '\n'
        if self.fp:
            return re.sub(r"(\d+.\d+e*-*\d*)", r"\1f", src, flags=re.MULTILINE)
        else:
            return src

    def save(self, include):
        try:
            include_path = Path(include)
            if include_path.is_dir():
                raise FileNotFoundError('Error: Path is a directory')
            with open(include_path, 'w') as file_to_write:
                file_to_write.write(str(self))
                print(f'Wrote to {include_path}')
            try:
                subprocess.call(
                    ['clang-format', '-i', '-style=Chromium', include_path.absolute()])
            except FileNotFoundError as e:
                print('Clang-format not found')
        except Exception as e:
            print(f'{e}')
