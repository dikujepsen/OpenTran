import ply.lex as lex
import ply.yacc as yacc

import lan
import rewriter
from codegen import cgen
from host import boilerplategen, kernelgen
from processing import collect_id as ci
from transformation import define_arguments as darg
from transformation import stencil
from transformation import transpose as tp

fileprefix = "../test/C/"
SetNoReadBack = True
DoOptimizations = True


def __get_ast_from_file(foldername, filename):
    cparser = yacc.yacc(module=lan)
    lex.lex(module=lan)

    fullfilename = fileprefix + foldername + '/' + filename
    try:
        f = open(fullfilename, 'r')
        s = f.read()
        f.close()
    except EOFError:
        print('file %s wasn\'t found', fullfilename)

    lex.input(s)
    while 1:
        tok = lex.token()
        if not tok: break
        ## print tok

    ast = cparser.parse(s)
    return ast


def __get_baseform_name(name):
    return fileprefix + name + '/' + __get_baseform_filename(name)


def __get_baseform_filename(name):
    return 'baseform_' + name.lower() + '.cpp'


def _create_baseform(name):
    ast = __get_ast_from_init(name)
    cprint = cgen.CGenerator()
    baseform_filename = __get_baseform_name(name)
    cprint.write_ast_to_file(ast, filename=baseform_filename)


def __get_ast_from_init(name):
    ast = __get_ast_from_file(name, name + 'For.cpp')
    ast.ext.append(lan.ProgramName(name))

    rw = rewriter.Rewriter()
    rw.rewrite_to_baseform(ast, name + 'For')

    return ast


def __get_ast_from_base(name):
    ast = __get_ast_from_file(name, __get_baseform_filename(name))
    return ast


def gen_full_code(ast):
    kgen = kernelgen.KernelGen(ast, fileprefix)
    cprint = cgen.CGenerator()

    kgen.generate_kernels()

    boilerplate = boilerplategen.Boilerplate(ast, SetNoReadBack)
    boilerast = boilerplate.generate_code()

    name = ci.get_program_name(ast)
    cprint.write_ast_to_file(boilerast, filename=fileprefix + name + '/' + 'boilerplate.cpp')


def matmul():
    name = 'MatMul'
    if True:
        ast = __get_ast_from_init(name)
    else:
        ast = __get_ast_from_base(name)

    __optimize(ast)


def __optimize(ast, par_dim=None):
    ast.ext.append(lan.ParDim(par_dim))
    name = ci.get_program_name(ast)
    if DoOptimizations:
        __main_transpose(ast)
        if name == 'Jacobi':
            __main_stencil(ast)
        __main_definearg(ast)

    gen_full_code(ast)


def knearest():
    name = 'KNearest'
    if True:
        ast = __get_ast_from_init(name)
    else:
        ast = __get_ast_from_base(name)
    __optimize(ast, par_dim=1)


def jacobi():
    name = 'Jacobi'
    if True:
        ast = __get_ast_from_init(name)
    else:
        ast = __get_ast_from_base(name)

    __optimize(ast)


def nbody():
    name = 'NBody'
    if True:
        ast = __get_ast_from_init(name)
    else:
        ast = __get_ast_from_base(name)

    __optimize(ast)


def laplace():
    name = 'Laplace'
    if True:
        ast = __get_ast_from_init(name)
    else:
        ast = __get_ast_from_base(name)
    __optimize(ast, par_dim=1)


def gaussian():
    name = 'GaussianDerivates'
    if True:
        ast = __get_ast_from_init(name)
    else:
        ast = __get_ast_from_base(name)
    __optimize(ast)


def __main_transpose(ast):
    tps = tp.Transpose(ast)
    tps.transpose()


def __main_definearg(ast):
    dargs = darg.DefineArguments(ast)
    dargs.define_arguments()


def __main_stencil(ast):
    sten = stencil.Stencil(ast)
    sten.stencil(['X1'], west=1, north=1, east=1, south=1, middle=0)


if __name__ == "__main__":
    matmul()
    knearest()
    jacobi()
    nbody()
    laplace()
    gaussian()
