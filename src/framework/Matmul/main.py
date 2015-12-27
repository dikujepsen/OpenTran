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
    rw, ast = __get_ast_from_init(name)
    cprint = cgen.CGenerator()
    baseform_filename = __get_baseform_name(name)
    cprint.write_ast_to_file(ast, filename=baseform_filename)


def __get_ast_from_init(name):
    ast = __get_ast_from_file(name, name + 'For.cpp')

    rw = rewriter.Rewriter()
    rw.rewrite_array_ref(ast)
    rw.rewrite_to_baseform(ast, name + 'For')

    return rw, ast


def __get_ast_from_base(name):
    ast = __get_ast_from_file(name, __get_baseform_filename(name))
    rw, _ = __get_ast_from_init(name)
    return rw, ast


def gen_full_code(name, ks, bps, tempast2):

    kgen = kernelgen.KernelGen(ks)

    kgen.generate_kernels(tempast2, name, fileprefix)

    boilerplate = boilerplategen.Boilerplate()
    boilerplate.set_struct(ks, bps, kgen.kgen_strt)
    boilerast = boilerplate.generate_code()

    cprint = cgen.CGenerator()
    cprint.write_ast_to_file(boilerast, filename=fileprefix + name + '/' + 'boilerplate.cpp')


def matmul():
    name = 'MatMul'
    if True:
        rw, ast = __get_ast_from_init(name)
    else:
        rw, ast = __get_ast_from_base(name)

    __optimize(rw, ast, name)


def __optimize(rw, ast, name, par_dim=None):
    tempast = copy.deepcopy(ast)
    tempast2 = copy.deepcopy(ast)
    tempast3 = copy.deepcopy(ast)

    ks = struct.KernelStruct()
    if par_dim is not None:
        ks.ParDim = par_dim
    ks.set_datastructure(rw, tempast)
    bps = struct.BoilerPlateStruct()
    bps.set_datastructure(tempast, par_dim)
    if DoOptimizations:
        __main_transpose(ks, bps, tempast3, par_dim=ks.ParDim)
        # an.Transpose()

        __main_definearg(ks, bps, tempast3, par_dim=ks.ParDim)
        # an.DefineArguments()
        __main_placeinreg(ks, bps, tempast3, par_dim=ks.ParDim)
        __main_placeinlocal(ks, bps, tempast3, par_dim=ks.ParDim)
        # an.PlaceInLocalMemory()
    if SetNoReadBack:
        bps.set_no_read_back()

    gen_full_code(name, ks, bps, tempast3)


def knearest():
    name = 'KNearest'
    if True:
        rw, ast = __get_ast_from_init(name)
    else:
        rw, ast = __get_ast_from_base(name)
    __optimize(rw, ast, name, par_dim=1)


def jacobi():
    name = 'Jacobi'
    if True:
        rw, ast = __get_ast_from_init(name)
    else:
        rw, ast = __get_ast_from_base(name)
    tempast = copy.deepcopy(ast)
    tempast2 = copy.deepcopy(ast)
    tempast3 = copy.deepcopy(ast)

    ks = struct.KernelStruct()
    ks.set_datastructure(rw, tempast3)
    bps = struct.BoilerPlateStruct()
    bps.set_datastructure(tempast)
    if DoOptimizations:
        __main_transpose(ks, bps, tempast3)
        __main_definearg(ks, bps, tempast3)
        __main_placeinreg(ks, bps, tempast3)
        # tf.localMemory(['X1'], west=1, north=1, east=1, south=1, middle=0)
        __main_stencil(ks, bps, tempast3)
        __main_placeinlocal(ks, bps, tempast3)
    if SetNoReadBack:
        bps.set_no_read_back()

    gen_full_code(name, ks, bps, tempast3)


def nbody():
    name = 'NBody'
    if True:
        rw, ast = __get_ast_from_init(name)
    else:
        rw, ast = __get_ast_from_base(name)

    __optimize(rw, ast, name)


def laplace():
    name = 'Laplace'
    if True:
        rw, ast = __get_ast_from_init(name)
    else:
        rw, ast = __get_ast_from_base(name)
    __optimize(rw, ast, name, par_dim=1)


def gaussian():
    name = 'GaussianDerivates'
    if True:
        rw, ast = __get_ast_from_init(name)
    else:
        rw, ast = __get_ast_from_base(name)
    __optimize(rw, ast, name)


def __main_transpose(ks, bps, tempast3, par_dim=None):
    tps = tp.Transpose()
    if par_dim is not None:
        tps.ParDim = par_dim
    tps.set_datastructures(tempast3)
    tps.transpose()

    for (arr_name, idx_list_list) in tps.Subscript.items():

        idx_list_list2 = ks.Subscript[arr_name]

        for i, idx_list in enumerate(idx_list_list):
            for j, idx in enumerate(idx_list):
                if isinstance(idx, lan.Id) and isinstance(idx_list_list2[i][j], lan.Id):
                    idx_list_list2[i][j].name = idx.name

    bps.WriteTranspose = tps.WriteTranspose
    bps.Transposition = tps.Transposition
    bps.NameSwap = tps.NameSwap
    bps.HstId = tps.HstId
    bps.GlobalVars = tps.GlobalVars

    ks.Type = tps.Type


def __main_definearg(ks, bps, tempast3, par_dim=None):
    dargs = darg.DefineArguments()
    if par_dim is not None:
        dargs.ParDim = par_dim
    dargs.set_datastructures(tempast3)
    dargs.define_arguments(bps.NameSwap)

    bps.define_compound = dargs.define_compound

    ks.KernelArgs = dargs.kernel_args


def __main_placeinreg(ks, bps, tempast3, par_dim=None):
    pireg = reg.PlaceInReg()
    if par_dim is not None:
        pireg.ParDim = par_dim
    pireg.set_datastructures(tempast3)
    pireg.place_in_reg()

    ks.PlaceInRegArgs = pireg.PlaceInRegArgs
    ks.PlaceInRegCond = pireg.PlaceInRegCond


def __main_placeinlocal(ks, bps, tempast3, par_dim=None):
    pilocal = local.PlaceInLocal()
    if par_dim is not None:
        pilocal.ParDim = par_dim
    pilocal.set_datastructures(tempast3)
    pilocal.place_in_local()

    ks.PlaceInLocalArgs = pilocal.PlaceInLocalArgs
    ks.PlaceInLocalCond = pilocal.PlaceInLocalCond


def __main_stencil(ks, bps, tempast3):
    sten = stencil.Stencil()
    sten.set_datastructures(tempast3)
    sten.stencil(['X1'], west=1, north=1, east=1, south=1, middle=0)

    ks.LocalSwap = sten.LocalSwap
    ks.num_array_dims = sten.num_array_dims
    ks.ArrayIdToDimName = sten.ArrayIdToDimName
    ks.LoopArrays = sten.LoopArrays
    ks.Add = sten.Add


if __name__ == "__main__":
    # matmul()
    # knearest()
    jacobi()
    # nbody()
    # laplace()
    # gaussian()
