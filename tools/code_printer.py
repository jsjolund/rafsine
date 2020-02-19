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
