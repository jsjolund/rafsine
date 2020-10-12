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
        self.includes = []
        self.fp = fp

    def parameter(self, *var, type='real_t'):
        """Add a function parameter"""
        for v in var:
            if isinstance(v, Matrix):
                self.parameters += [
                    f'{type} {f", {type} ".join([str(v.row(i)[0]) for i in range(0, v.shape[0])])}']
            else:
                self.parameters += [f'{type} {v}']

    def define(self, *var, type='real_t'):
        """Define a variable"""
        for v in var:
            if isinstance(v, Matrix):
                tmp = []
                for i in range(0, v.shape[0]):
                    if v.row(i)[0] != 0:
                        tmp += [str(v.row(i)[0])]
                self.rows += [
                    f'{type} {", ".join(tmp)};']
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
        funcs = {"Pow": "powf"} if self.fp else {}
        if isinstance(var, Matrix):
            for i in range(0, var.shape[0]):
                if var.row(i)[0] != 0:
                    self.rows += [ccode(expr.row(i)[0], assign_to=var.row(i)
                                        [0], user_functions=funcs)]
        else:
            self.rows += [ccode(expr, assign_to=var, user_functions=funcs)]

    def __set_fp(self, content):
        return re.sub(r"(\d+.\d+(e-\d+)*)", r"\1f", content, flags=re.MULTILINE)

    def include(self, headername):
        self.includes += ['#include "'+headername+'"']

    def __repr__(self):
        src = '#pragma once\n' \
            + '\n'.join(self.includes) + '\n' \
            + f'__device__ __forceinline__ void {self.name}(' \
            + ', '.join(self.parameters) + ') {\n' \
            + '\n'.join(self.rows) + '\n}\n'
        if self.fp:
            return self.__set_fp(src)
        else:
            return src

    def usage(self, cmdname):
        print(f'USAGE: {cmdname} OUTPUTFILE.hpp')

    def handle(self, argv):
        if len(argv) == 1 or len(argv) > 2 or argv[1] in ['-h', '--help']:
            print(str(self))
            self.usage(argv[0])
            return
        try:
            path = Path(argv[1])
            if path.is_dir():
                raise FileNotFoundError('Error: Path is a directory')
            with open(path, 'w') as file:
                file.write(str(self))
                print(f'Wrote to {path}')
            try:
                subprocess.call(
                    ['clang-format', '-i', '-style=Chromium', path.absolute()])
            except FileNotFoundError as e:
                print('Clang-format not found')
        except Exception as e:
            print(f'{e}')
