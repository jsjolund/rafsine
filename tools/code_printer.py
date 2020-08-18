import re
import subprocess
from pathlib import Path
from sympy import Matrix
from sympy.codegen.ast import Assignment
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
        """Define a function parameter"""
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
        if isinstance(var, Matrix):
            for i in range(0, var.shape[0]):
                self.rows += [self.doprint(Assignment(var.row(i)
                                                      [0], expr.row(i)[0]))]
        else:
            self.rows += [self.doprint(Assignment(var, expr))]

    def to_include(self):
        inc = '#pragma once\n' \
            + '#include "CudaUtils.hpp"\n' \
            + '#include "PhysicalQuantity.hpp"\n' \
            + f'__device__ void {self.name}(' \
            + ', '.join(self.parameters) + ');\n'
        if self.fp:
            return re.sub(r"(\d+.\d+)", r"\1f", inc, flags=re.MULTILINE)
        else:
            return inc

    def to_source(self):
        src = f'__device__ void {self.name}(' \
            + ', '.join(self.parameters) + ') {\n' \
            + '\n'.join(self.rows) + '\n' + '}'
        if self.fp:
            return re.sub(r"(\d+.\d+)", r"\1f", src, flags=re.MULTILINE)
        else:
            return src

    def save(self, include, source):
        try:
            include_path = Path(include)
            source_path = Path(source)
            if source_path.is_dir() or include_path.is_dir():
                raise FileNotFoundError('Error: Path is a directory')
            with open(include_path, 'w') as file_to_write:
                file_to_write.write(str(self.to_include()))
                print(f'Wrote to {include_path}')
            with open(source_path, 'w') as file_to_write:
                file_to_write.write(
                    f'#include "{include_path.name}"\n' + str(self.to_source()))
                print(f'Wrote to {source_path}')
            try:
                subprocess.call(
                    ['clang-format', '-i', '-style=Chromium', include_path.absolute()])
                subprocess.call(
                    ['clang-format', '-i', '-style=Chromium', source_path.absolute()])
            except FileNotFoundError as e:
                print('Clang-format not found')
        except Exception as e:
            print(f'{e}')
