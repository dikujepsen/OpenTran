##from rewriter import *
from os.path import basename
import copy
##from lan_parser import *
# from cgen import *
# from transformation import *
# from analysis import *
import ply.lex as lex
import os
import sys
if "../.." not in sys.path:
    sys.path.insert(0, "../..")
# from framework.lan_parser import *
# from framework.lan import *
# import framework.lan
import framework
import representation
import rewriter
import transf_repr
import transformation
import analysis
import ply.yacc as yacc
import cgen
import lan
import boilerplategen


fileprefix = "../../test/C/"
SetNoReadBack = False
DoOptimizations = True


def LexAndParse(name, createTemp):

    cparser = yacc.yacc(module=lan)
    lex.lex(module=lan)

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
        # ast.show()
        # print ast
        ## print slist
        cprint = cgen.CGenerator()

        ## printres = cprint.visit(ast)
        ## print printres
        astrepr = representation.Representation()
        astrepr.initOriginal(ast)
        rw = rewriter.Rewriter(astrepr)
        tempfilename = fileprefix + name + '/'+'temp' + name.lower() + '.cpp'
        if createTemp:
            rw.rewrite(ast, funcname, changeAST=True)
            cprint.create_temp(ast, filename=tempfilename)

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
        boilerplate = boilerplategen.Boilerplate()
        boilerast = boilerplate.generate_code(rw)
        cprint.create_temp(boilerast, filename=fileprefix + name + '/' + 'boilerplate.cpp')




def matmul():
    name = 'MatMul'
    (rw, ast, tempast, tempast2, funcname) = LexAndParse(name, True)
    transf_rp = transf_repr.Transf_Repr(rw.astrepr)
    transf_rp.initNewRepr(tempast)
    tf = transformation.Transformation(transf_rp)

    an = analysis.Analysis(transf_rp, tf)
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

