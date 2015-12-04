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
import define_arguments as darg
import transpose as tp
import place_in_reg as reg
import place_in_local as local
import stencil

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
    astrepr = representation.Representation()
    astrepr.init_original(ast)
    rw = rewriter.Rewriter(astrepr)
    rw.rewrite_to_baseform(ast, name + 'For')
    return rw, ast


def __get_ast_from_base(name):
    ast = __get_ast_from_file(name, __get_baseform_filename(name))
    rw, _ = __get_ast_from_init(name)
    return rw, ast


def gen_full_code(name, an, tempast2):
    cprint = cgen.CGenerator()
    rw = an.rw
    an.GenerateKernels(tempast2, name, fileprefix)
    boilerplate = boilerplategen.Boilerplate()
    boilerast = boilerplate.generate_code(rw)
    cprint.write_ast_to_file(boilerast, filename=fileprefix + name + '/' + 'boilerplate.cpp')


def matmul():
    name = 'MatMul'
    if False:
        rw, ast = __get_ast_from_init(name)
    else:
        rw, ast = __get_ast_from_base(name)
    tempast = copy.deepcopy(ast)
    tempast2 = copy.deepcopy(ast)
    tempast3 = copy.deepcopy(ast)

    transf_rp = transf_repr.TransfRepr(rw.astrepr)
    transf_rp.init_rew_repr(tempast, dev='CPU')
    tf = transformation.Transformation(transf_rp)

    an = analysis.Analysis(transf_rp, tf)
    if DoOptimizations:
        __main_transpose(transf_rp, tempast3)
        # an.Transpose()

        __main_definearg(transf_rp, tempast3)
        # an.DefineArguments()
        __main_placeinreg(an, tempast3)
        __main_placeinlocal(an, tempast3)
        # an.PlaceInLocalMemory()
    if SetNoReadBack:
        tf.SetNoReadBack()

    ## rw.DataStructures()
    gen_full_code(name, an, tempast2)


def knearest():
    name = 'KNearest'
    if True:
        rw, ast = __get_ast_from_init(name)
    else:
        rw, ast = __get_ast_from_base(name)
    tempast = copy.deepcopy(ast)
    tempast2 = copy.deepcopy(ast)
    tempast3 = copy.deepcopy(ast)

    transf_rp = transf_repr.TransfRepr(rw.astrepr)
    transf_rp.ParDim = 1
    transf_rp.init_rew_repr(tempast, dev='CPU')
    tf = transformation.Transformation(transf_rp)
    # tf.SetParDim(1)
    an = analysis.Analysis(transf_rp, tf)
    if DoOptimizations:
        __main_transpose(transf_rp, tempast3, par_dim=1)
        __main_definearg(transf_rp, tempast3, par_dim=1)
        __main_placeinreg(an, tempast3, par_dim=1)
        __main_placeinlocal(an, tempast3, par_dim=1)

    if SetNoReadBack:
        tf.SetNoReadBack()

    gen_full_code(name, an, tempast3)


def jacobi():
    name = 'Jacobi'
    if True:
        rw, ast = __get_ast_from_init(name)
    else:
        rw, ast = __get_ast_from_base(name)
    tempast = copy.deepcopy(ast)
    tempast2 = copy.deepcopy(ast)
    tempast3 = copy.deepcopy(ast)

    transf_rp = transf_repr.TransfRepr(rw.astrepr)
    transf_rp.init_rew_repr(tempast, dev='CPU')
    tf = transformation.Transformation(transf_rp)

    an = analysis.Analysis(transf_rp, tf)
    if DoOptimizations:
        __main_transpose(transf_rp, tempast3)
        __main_definearg(transf_rp, tempast3)
        __main_placeinreg(an, tempast3)
        # tf.localMemory(['X1'], west=1, north=1, east=1, south=1, middle=0)
        __main_stencil(an, tempast3)
        __main_placeinlocal(an, tempast3)
    if SetNoReadBack:
        tf.SetNoReadBack()

    gen_full_code(name, an, tempast3)


def nbody():
    name = 'NBody'
    if True:
        rw, ast = __get_ast_from_init(name)
    else:
        rw, ast = __get_ast_from_base(name)
    tempast = copy.deepcopy(ast)
    tempast2 = copy.deepcopy(ast)
    tempast3 = copy.deepcopy(ast)

    transf_rp = transf_repr.TransfRepr(rw.astrepr)
    transf_rp.init_rew_repr(tempast, dev='CPU')
    tf = transformation.Transformation(transf_rp)

    an = analysis.Analysis(transf_rp, tf)
    if DoOptimizations:
        __main_transpose(transf_rp, tempast3)
        __main_definearg(transf_rp, tempast3)

        __main_placeinreg(an, tempast3)
        # an.PlaceInReg()
        # an.PlaceInLocalMemory()
        __main_placeinlocal(an, tempast3)
    if SetNoReadBack:
        tf.SetNoReadBack()
    ## rw.Unroll2({'j': 32})
    gen_full_code(name, an, tempast3)


def laplace():
    name = 'Laplace'
    if True:
        rw, ast = __get_ast_from_init(name)
    else:
        rw, ast = __get_ast_from_base(name)
    tempast = copy.deepcopy(ast)
    tempast2 = copy.deepcopy(ast)
    tempast3 = copy.deepcopy(ast)

    transf_rp = transf_repr.TransfRepr(rw.astrepr)
    transf_rp.ParDim = 1
    transf_rp.init_rew_repr(tempast, dev='CPU')
    tf = transformation.Transformation(transf_rp)

    an = analysis.Analysis(transf_rp, tf)
    if DoOptimizations:
        __main_transpose(transf_rp, tempast3, par_dim=1)
        __main_definearg(transf_rp, tempast3, par_dim=1)
        __main_placeinreg(an, tempast3, par_dim=1)
        __main_placeinlocal(an, tempast3, par_dim=1)
    else:
        tf.SetDefine(['dim'])

    if SetNoReadBack:
        tf.SetNoReadBack()

    ## rw.DataStructures()

    ## tf.Unroll2({'d' : 0, 'd_outer' : 0, 'd_inner' : 0})
    gen_full_code(name, an, tempast3)


def gaussian():
    name = 'GaussianDerivates'
    if True:
        rw, ast = __get_ast_from_init(name)
    else:
        rw, ast = __get_ast_from_base(name)
    tempast = copy.deepcopy(ast)
    tempast2 = copy.deepcopy(ast)
    tempast3 = copy.deepcopy(ast)

    transf_rp = transf_repr.TransfRepr(rw.astrepr)
    transf_rp.init_rew_repr(tempast, dev='CPU')
    tf = transformation.Transformation(transf_rp)

    an = analysis.Analysis(transf_rp, tf)
    if DoOptimizations:
        # an.Transpose()
        __main_transpose(transf_rp, tempast3)
        # an.DefineArguments()
        __main_definearg(transf_rp, tempast3)
        # an.PlaceInReg()
        __main_placeinreg(an, tempast3)
        __main_placeinlocal(an, tempast3)

        # an.PlaceInLocalMemory()
    #         ## tf.Unroll2({'k' : 0, 'd' : 0, 'g' : 0, 'b' : 0})
    ## rw.DataStructures()
    if SetNoReadBack:
        tf.SetNoReadBack()
    gen_full_code(name, an, tempast3)


def __main_transpose(transf_rp, tempast3, par_dim=None):
    tps = tp.Transpose()
    if par_dim is not None:
        tps.ParDim = par_dim
    tps.set_datastructures(tempast3)
    tps.transpose()
    # transf_rp.Subscript = tps.Subscript
    # print tps.Subscript
    # print transf_rp.Subscript
    for (arr_name, idx_list_list) in tps.Subscript.items():

        idx_list_list2 = transf_rp.Subscript[arr_name]

        for i, idx_list in enumerate(idx_list_list):
            for j, idx in enumerate(idx_list):
                if isinstance(idx, lan.Id) and isinstance(idx_list_list2[i][j], lan.Id):
                    idx_list_list2[i][j].name = idx.name

    transf_rp.WriteTranspose = tps.WriteTranspose
    transf_rp.Transposition = tps.Transposition
    transf_rp.NameSwap = tps.NameSwap
    transf_rp.Type = tps.Type
    transf_rp.HstId = tps.HstId
    transf_rp.GlobalVars = tps.GlobalVars


def __main_definearg(transf_rp, tempast3, par_dim=None):
    dargs = darg.DefineArguments()
    if par_dim is not None:
        dargs.ParDim = par_dim
    dargs.set_datastructures(tempast3)
    dargs.define_arguments(transf_rp.NameSwap)
    transf_rp.KernelArgs = dargs.kernel_args

    transf_rp.Define = dargs.define_compound


def __main_placeinreg(an, tempast3, par_dim=None):
    pireg = reg.PlaceInReg()
    if par_dim is not None:
        pireg.ParDim = par_dim
    pireg.set_datastructures(tempast3)
    pireg.place_in_reg()
    an.PlaceInRegArgs = pireg.PlaceInRegArgs
    an.PlaceInRegCond = pireg.PlaceInRegCond

def __main_placeinlocal(an, tempast3, par_dim=None):
    pilocal = local.PlaceInLocal()
    if par_dim is not None:
        pilocal.ParDim = par_dim
    pilocal.set_datastructures(tempast3)
    pilocal.place_in_local()
    an.PlaceInLocalArgs = pilocal.PlaceInLocalArgs
    an.PlaceInLocalCond = pilocal.PlaceInLocalCond
    an.rw.Local = pilocal.Local

def __main_stencil(an, tempast3):
    sten = stencil.Stencil()
    sten.set_datastructures(tempast3)
    sten.stencil(['X1'], west=1, north=1, east=1, south=1, middle=0)
    an.rw.astrepr.num_array_dims = sten.num_array_dims
    an.rw.LocalSwap = sten.LocalSwap
    an.rw.ArrayIdToDimName = sten.ArrayIdToDimName
    an.rw.Kernel = sten.Kernel
    an.rw.astrepr.LoopArrays = sten.LoopArrays

if __name__ == "__main__":
    matmul()
    knearest()
    jacobi()
    nbody()
    laplace()
    gaussian()
