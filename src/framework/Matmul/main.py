
import copy
import ply.lex as lex
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


fileprefix = "../../test/C/"
SetNoReadBack = False
DoOptimizations = True

def __get_ast_from_file(name):
    cparser = yacc.yacc(module=lan)
    lex.lex(module=lan)

    filename = fileprefix + name + '/' + name + 'For.cpp'
    try:
        f = open(filename, 'r')
        s = f.read()
        f.close()
    except EOFError:
        print('file %s wasn\'t found', filename)

    lex.input(s)
    while 1:
        tok = lex.token()
        if not tok: break
        ## print tok

    ast = cparser.parse(s)
    return ast


def __get_baseform_name(name):
    return fileprefix + name + '/baseform_' + name.lower() + '.cpp'


def __get_init_rewriter(name):
    ast = __get_ast_from_file(name)
    astrepr = representation.Representation()
    astrepr.init_original(ast)
    return rewriter.Rewriter(astrepr)


def _create_baseform(name):
    ast = __get_ast_from_file(name)
    astrepr = representation.Representation()
    astrepr.normalize_subcript(ast)

    rw = __get_init_rewriter(name)
    baseform_filename = __get_baseform_name(name)
    rw.rewrite_to_baseform(ast, name + 'For', change_ast=True)
    cprint = cgen.CGenerator()
    cprint.write_ast_to_file(ast, filename=baseform_filename)


def lex_and_parse(name):

    rw = __get_init_rewriter(name)

    filename = __get_baseform_name(name)
    try:
        f = open(filename, 'r')
        s = f.read()
        f.close()
    except EOFError:
        print('file %s wasn\'t found', filename)
        raise EOFError

    cparser = yacc.yacc(module=lan)
    ast = cparser.parse(s)

    tempast = copy.deepcopy(ast)
    tempast2 = copy.deepcopy(ast)
    return rw, tempast, tempast2


def gen_full_code(name, an, tempast2):
        cprint = cgen.CGenerator()
        rw = an.rw
        an.GenerateKernels(tempast2, name, fileprefix)
        boilerplate = boilerplategen.Boilerplate()
        boilerast = boilerplate.generate_code(rw)
        cprint.write_ast_to_file(boilerast, filename=fileprefix + name + '/' + 'boilerplate.cpp')


def matmul():
    name = 'MatMul'
    if True:
        _create_baseform(name)
    (rw, tempast, tempast2) = lex_and_parse(name)

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

