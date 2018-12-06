#!/bin/bash

sed 's/P02HDZX/P02HDZ01/g' P02HDZX_T.tex | tee P02HDZ01_T.tex
sed 's/P02HDZX/P02HDZ02/g' P02HDZX_T.tex | tee P02HDZ02_T.tex
sed 's/P02HDZX/P02HDZ03/g' P02HDZX_T.tex | tee P02HDZ03_T.tex
sed 's/P02HDZX/P02HDZ04/g' P02HDZX_T.tex | tee P02HDZ04_T.tex

sed 's/P02RX/P02R01/g' P02R0X_T.tex | tee P02R01_T.tex
sed 's/P02RX/P02R02/g' P02R0X_T.tex | tee P02R02_T.tex
sed 's/P02RX/P02R03/g' P02R0X_T.tex | tee P02R03_T.tex
sed 's/P02RX/P02R04/g' P02R0X_T.tex | tee P02R04_T.tex
sed 's/P02RX/P02R05/g' P02R0X_T.tex | tee P02R05_T.tex
sed 's/P02RX/P02R06/g' P02R0X_T.tex | tee P02R06_T.tex
sed 's/P02RX/P02R07/g' P02R0X_T.tex | sed 's/,ylabel={Power (kW)}/,ylabel={Power (kW)},ymin=2.4,ymax=4.6,ytick distance=0.5/g' | tee P02R07_T.tex
sed 's/P02RX/P02R08/g' P02R0X_T.tex | tee P02R08_T.tex
sed 's/P02RX/P02R09/g' P02R0X_T.tex | tee P02R09_T.tex
sed 's/P02RX/P02R10/g' P02R0X_T.tex | sed 's/,ylabel={Power (kW)}/,ylabel={Power (kW)},ymin=0.4,ymax=1.6,ytick distance=0.5/g' | tee P02R10_T.tex

BUILD='latexmk -pdf -pdflatex="pdflatex -file-line-error -interaction=nonstopmode"'

eval $BUILD P02R01_T.tex
eval $BUILD P02R02_T.tex
eval $BUILD P02R03_T.tex
eval $BUILD P02R04_T.tex
eval $BUILD P02R05_T.tex
eval $BUILD P02R06_T.tex
eval $BUILD P02R07_T.tex
eval $BUILD P02R08_T.tex
eval $BUILD P02R09_T.tex
eval $BUILD P02R10_T.tex
eval $BUILD P02HDZ01_T.tex
eval $BUILD P02HDZ02_T.tex
eval $BUILD P02HDZ03_T.tex
eval $BUILD P02HDZ04_T.tex
eval $BUILD P02RX_T_mean.tex
eval $BUILD P02HDZX_T_mean.tex

#latexmk -pdf -pdflatex="pdflatex -file-line-error -interaction=nonstopmode" "%f" | grep "^.*:[0-9]*: .*$"
