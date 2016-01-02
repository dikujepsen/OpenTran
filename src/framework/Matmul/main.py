import copy
import ply.lex as lex
import ply.yacc as yacc
import boilerplategen
import cgen
import define_arguments as darg
import kernelgen
import lan
import place_in_local as local
import place_in_reg as reg
import rewriter
import stencil
import struct
import transpose as tp

fileprefix = "../../test/C/"
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

    rw = rewriter.Rewriter()
    rw.rewrite_array_ref(ast)
    rw.rewrite_to_baseform(ast, name + 'For')

    return ast


def __get_ast_from_base(name):
    ast = __get_ast_from_file(name, __get_baseform_filename(name))
    return ast


def gen_full_code(name, ks, tempast3):
    kgen = kernelgen.KernelGen(ks.ParDim)
    cprint = cgen.CGenerator()

    kgen.generate_kernels(tempast3, name, fileprefix)

    boilerplate = boilerplategen.Boilerplate()
    boilerplate.set_struct(ks.ParDim, kgen.kgen_strt, tempast3, SetNoReadBack)
    boilerast = boilerplate.generate_code()

    cprint.write_ast_to_file(boilerast, filename=fileprefix + name + '/' + 'boilerplate.cpp')


def matmul():
    name = 'MatMul'
    if True:
        ast = __get_ast_from_init(name)
    else:
        ast = __get_ast_from_base(name)

    __optimize(ast, name)


def __optimize(ast, name, par_dim=None):
    tempast = copy.deepcopy(ast)
    tempast2 = copy.deepcopy(ast)
    tempast3 = copy.deepcopy(ast)

    ks = struct.KernelStruct()
    if par_dim is not None:
        ks.ParDim = par_dim
    ks.set_datastructure(tempast3)
    if DoOptimizations:
        __main_transpose(tempast3, par_dim=ks.ParDim)
        __main_placeinreg(tempast3, par_dim=ks.ParDim)
        __main_placeinlocal(tempast3, par_dim=ks.ParDim)

        __main_definearg(tempast3, par_dim=ks.ParDim)

    gen_full_code(name, ks, tempast3)


def knearest():
    name = 'KNearest'
    if True:
        ast = __get_ast_from_init(name)
    else:
        ast = __get_ast_from_base(name)
    __optimize(ast, name, par_dim=1)


def jacobi():
    name = 'Jacobi'
    if True:
        ast = __get_ast_from_init(name)
    else:
        ast = __get_ast_from_base(name)
    tempast = copy.deepcopy(ast)
    tempast2 = copy.deepcopy(ast)
    tempast3 = copy.deepcopy(ast)

    ks = struct.KernelStruct()
    ks.set_datastructure(tempast3)

    if DoOptimizations:
        __main_transpose(tempast3)
        __main_placeinreg(tempast3)
        # tf.localMemory(['X1'], west=1, north=1, east=1, south=1, middle=0)
        __main_stencil(tempast3)
        __main_placeinlocal(tempast3)
        __main_definearg(tempast3)

    gen_full_code(name, ks, tempast3)


def nbody():
    name = 'NBody'
    if True:
        ast = __get_ast_from_init(name)
    else:
        ast = __get_ast_from_base(name)

    __optimize(ast, name)


def laplace():
    name = 'Laplace'
    if True:
        ast = __get_ast_from_init(name)
    else:
        ast = __get_ast_from_base(name)
    __optimize(ast, name, par_dim=1)


def gaussian():
    name = 'GaussianDerivates'
    if True:
        ast = __get_ast_from_init(name)
    else:
        ast = __get_ast_from_base(name)
    __optimize(ast, name)


def __main_transpose(tempast3, par_dim=None):
    tps = tp.Transpose()
    tps.set_datastructures(tempast3, par_dim)
    tps.transpose(tempast3)


def __main_definearg(tempast3, par_dim=None):
    dargs = darg.DefineArguments()
    if par_dim is not None:
        dargs.ParDim = par_dim
    dargs.set_datastructures(tempast3)
    dargs.define_arguments()


def __main_placeinreg(tempast3, par_dim=None):
    pass


def __main_placeinlocal(tempast3, par_dim=None):
    pass


def __main_stencil(tempast3):
    sten = stencil.Stencil()
    sten.set_datastructures(tempast3)
    sten.stencil(['X1'], west=1, north=1, east=1, south=1, middle=0)


if __name__ == "__main__":
    matmul()
    knearest()
    jacobi()
    nbody()
    laplace()
    gaussian()
