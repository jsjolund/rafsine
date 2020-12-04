import re
import os
import subprocess
from pathlib import Path
from sympy import Matrix
from sympy.printing.c import C99CodePrinter, ccode
import tempfile


class CodePrinter(C99CodePrinter):
    """Code printer for CUDA LBM kernel generator"""

    def __init__(self, name, fp=True):
        self.name = name
        self.parameters = []
        self.rows = []
        self.includes = []
        self.fp = fp
        self.src = ''

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
                                        [0], user_functions=funcs, standard='C99')]
        else:
            self.rows += [ccode(expr, assign_to=var, user_functions=funcs, standard='C99')]

    def eval(self, expr):
        funcs = {"Pow": "powf"} if self.fp else {}
        return ccode(expr, user_functions=funcs, standard='C99')

    def format_fp(self, content):
        """ Format floating point numbers to end in 'f' """
        return re.sub(r"(\d+.\d+(e-\d+)*)", r"\1f", content, flags=re.MULTILINE)

    def include(self, headername):
        self.includes += ['#include "'+headername+'"']

    def __remove_powf(self, string):
        pattern = r"powf\(([a-zA-Z0-9._+\-*/\s]+),\s*([0-9])\)"
        m = re.search(pattern, string)
        if not m:
            return string
        expr = m.group(1)
        num = m.group(2)
        if expr and num:
            new_exp = '*'.join([f'({expr})' for i in range(int(num))])
            new_string = string[0:m.start()] + new_exp + string[m.end():]
            return self.__remove_powf(new_string)
        else:
            return string

    def __repr__(self):
        lines = self.src.splitlines(True)
        for i in range(0, len(lines)):
            lines[i] = self.__remove_powf(lines[i])
        self.src = ''.join(lines)
        if self.fp:
            self.src = self.format_fp(self.src)
        return self.format(self.src)

    def usage(self, cmdname):
        print(f'USAGE: {cmdname} OUTPUTFILE')

    def format(self, code_string):
        path = os.path.join(tempfile.gettempdir(), next(tempfile._get_candidate_names()))
        f = open(path, "a")
        f.write(code_string)
        f.close()
        try:
            subprocess.call(['clang-format', '-i', '-style=Chromium', path])
        except FileNotFoundError as e:
            print('Clang-format not found')
        f = open(path, "r")
        code_string = f.read()
        f.close()
        os.remove(path)
        return code_string

    def generate(self, argv):
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
        except Exception as e:
            print(f'{e}')


class HppFile(CodePrinter):
    """ Creates a .hpp file containing a single __device__ __forceinline__ function with provided parameters """

    def __init__(self, name):
        super().__init__(self, name)
        self.name = name
        self.prefix = '__device__ __forceinline__'

    def __repr__(self):
        self.src = '#pragma once\n' \
            + '\n'.join(self.includes) + '\n' \
            + f'{self.prefix} void {self.name}(' \
            + ', '.join(self.parameters) + ') {\n' \
            + '\n'.join(self.rows) + '\n}\n'
        return super().__repr__()


class CppFile(CodePrinter):
    """ Creates a .cpp/.cu file containing a single __global__ function with provided parameters """

    def __init__(self, name):
        super().__init__(self, name)
        self.name = name
        self.prefix = '__global__'

    def __repr__(self):
        self.src = '\n'.join(self.includes) + '\n' \
            + f'{self.prefix} void {self.name}(' \
            + ', '.join(self.parameters) + ') {\n' \
            + '\n'.join(self.rows) + '\n}\n'
        return super().__repr__()


class AnyFile(CodePrinter):
    """ Creates a C++ file without any function definition or parameters """

    def __init__(self):
        super().__init__(self, '')

    def __repr__(self):
        self.src = '\n'.join(self.includes) + '\n' \
            + '\n'.join(self.rows) + '\n'
        return super().__repr__()
