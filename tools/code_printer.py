import subprocess
from pathlib import Path
from sympy import Matrix
from sympy.codegen.ast import Assignment
from sympy.printing.ccode import C99CodePrinter


class CodePrinter(C99CodePrinter):
    """Code printer for CUDA LBM kernel generator"""

    def __init__(self):
        super().__init__()
        self.rows = []

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

    def __repr__(self):
        return '\n'.join(self.rows)

    def save(self, filepath):
        try:
            path = Path(filepath)
            if path.is_dir():
                raise FileNotFoundError('Error: Path is a directory')
            with open(path, 'w') as file_to_write:
                file_to_write.write(str(self) + '\n')
                print(f'Wrote to {path}')
            try:
                subprocess.call(
                    ['clang-format', '-i', '-style=Chromium', path.absolute()])
            except Exception as e:
                print(f'{e}')
        except Exception as e:
            print(f'{e}')
