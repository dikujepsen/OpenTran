##from rewriter import *
from os.path import basename
import copy
##from lan_parser import *
# from cgen import *
# from transformation import *
# from analysis import *
fileprefix = "../../test/C/"
import ply.lex as lex
import os
import sys
if "../.." not in sys.path: sys.path.insert(0,"../..")
from framework import cgen, rewriter, transformation, analysis
from framework.lan_parser import *

import ply.yacc as yacc

SetNoReadBack = False
DoOptimizations = True


def LexAndParse(name, createTemp):

    cparser = yacc.yacc()
    lex.lex()

    run = 1
    while run:
        filename = fileprefix + name + '/' + name + 'For.cpp'
        funcname = basename(os.path.splitext(filename)[0])
        try:
            f = open(filename, 'r')
            s = f.read()
            f.close()
            ## print s
        except EOFError:
            break


        lex.input(s)
        while 1:
            tok = lex.token()
            if not tok: break
            ## print tok

        ast = cparser.parse(s)
        ## ast.show()
        ## print ast
        ## print slist
        cprint = cgen.CGenerator()

        ## printres = cprint.visit(ast)
        ## print printres
        rw = rewriter.Rewriter()
        rw.initOriginal(ast)
        tempfilename = fileprefix + name + '/'+'temp' +name.lower() + '.cpp'
        if createTemp:
            rw.rewrite(ast, funcname, changeAST = True)
            cprint.createTemp(ast, filename = tempfilename)

        run = 0
        filename = tempfilename
        ## funcname = basename(os.path.splitext(filename)[0])
        try:
            f = open(filename, 'r')
            s = f.read()
            f.close()
        except EOFError:
            break

        ast = cparser.parse(s)
        ## ## ast.show()
        tempast = copy.deepcopy(ast)
        tempast2 = copy.deepcopy(ast)
        return (rw, ast, tempast, tempast2, funcname)

def CGen(name, funcname, an, tempast2, ast, kernelstringname = ''):
        cprint = cgen.CGenerator()
        rw = an.rw
        an.GenerateKernels(tempast2, name, fileprefix)
        ## rw.InSourceKernel(tempast2, filename = fileprefix + name + '/'+funcname + '.cl', kernelstringname = kernelstringname)
        boilerast = rw.generateBoilerplateCode(ast)
        cprint.createTemp(boilerast, filename = fileprefix + name + '/'+'boilerplate.cpp')




def matmul():
    name = 'MatMul'
    (rw, ast, tempast, tempast2, funcname) = LexAndParse(name, True)
    rw.initNewRepr(tempast)
    tf = transformation.Transformation(rw)

    an = analysis.Analysis(rw, tf)
    if DoOptimizations:
        an.Transpose()
        an.DefineArguments()
        an.PlaceInReg()
        an.PlaceInLocalMemory()
    if SetNoReadBack:
        tf.SetNoReadBack()

    ## rw.DataStructures()
    CGen(name, funcname, an, tempast2, ast)


if __name__ == "__main__":
    matmul()
