
from os.path import basename
import copy
import ply.lex as lex
import os
import sys
import representation
import rewriter
import transf_repr
import transformation
import analysis
import ply.yacc as yacc
import cgen
import lan
import boilerplategen
if "../.." not in sys.path:
    sys.path.insert(0, "../..")


fileprefix = "../../test/C/"
SetNoReadBack = False
DoOptimizations = True


def LexAndParse(name, createTemp):

    cparser = yacc.yacc(module=lan)
    lex.lex(module=lan)

    filename = fileprefix + name + '/' + name + 'For.cpp'
    funcname = basename(os.path.splitext(filename)[0])
    try:
        f = open(filename, 'r')
        s = f.read()
        f.close()
        ## print s
    except EOFError:
        print('file %s wasn\'t found', filename)

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
    baseform_filename = fileprefix + name + '/baseform_' + name.lower() + '.cpp'
    if createTemp:
        rw.rewrite_to_baseform(ast, funcname, changeAST=True)
        cprint.write_ast_to_file(ast, filename=baseform_filename)

    filename = baseform_filename
    try:
        f = open(filename, 'r')
        s = f.read()
        f.close()
    except EOFError:
        print('file %s wasn\'t found', filename)

    ast = cparser.parse(s)
    ## ## ast.show()
    tempast = copy.deepcopy(ast)
    tempast2 = copy.deepcopy(ast)
    return rw, tempast, tempast2, funcname


def gen_full_code(name, an, tempast2):
        cprint = cgen.CGenerator()
        rw = an.rw
        an.GenerateKernels(tempast2, name, fileprefix)
        boilerplate = boilerplategen.Boilerplate()
        boilerast = boilerplate.generate_code(rw)
        cprint.write_ast_to_file(boilerast, filename=fileprefix + name + '/' + 'boilerplate.cpp')




def matmul():
    name = 'MatMul'
    (rw, tempast, tempast2, funcname) = LexAndParse(name, True)
    # astrepr = representation.Representation()
    # astrepr.initOriginal(ast)
    # rw = rewriter.Rewriter(astrepr)

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
    gen_full_code(name, an, tempast2)


if __name__ == "__main__":
    matmul()

