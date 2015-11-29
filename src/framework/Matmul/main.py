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
    transf_rp.init_rew_repr(tempast)
    tf = transformation.Transformation(transf_rp)

    an = analysis.Analysis(transf_rp, tf)
    if DoOptimizations:
        tps = tp.Transpose()
        tps.set_datastructues(tempast3)
        tps.transpose()
        transf_rp.Subscript = tps.Subscript
        transf_rp.WriteTranspose = tps.WriteTranspose
        transf_rp.Transposition = tps.Transposition
        transf_rp.NameSwap = tps.NameSwap
        transf_rp.Type = tps.Type
        transf_rp.HstId = tps.HstId
        transf_rp.GlobalVars = tps.GlobalVars

        # an.Transpose()
        dargs = darg.DefineArguments()
        dargs.set_datastructures(tempast3)
        dargs.define_arguments(transf_rp.NameSwap)
        transf_rp.KernelArgs = dargs.kernel_args
        transf_rp.Define = dargs.define_compound
        # an.DefineArguments()
        an.PlaceInReg()
        an.PlaceInLocalMemory()
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
    transf_rp.init_rew_repr(tempast)
    tf = transformation.Transformation(transf_rp)
    # tf.SetParDim(1)
    an = analysis.Analysis(transf_rp, tf)
    if DoOptimizations:
        # tps = tp.Transpose()
        # tps.set_datastructues(tempast3)
        # tps.transpose()
        # transf_rp.Subscript = tps.Subscript
        # transf_rp.WriteTranspose = tps.WriteTranspose
        # transf_rp.Transposition = tps.Transposition
        # transf_rp.NameSwap = tps.NameSwap
        # transf_rp.Type = tps.Type
        # transf_rp.HstId = tps.HstId
        # transf_rp.GlobalVars = tps.GlobalVars
        an.Transpose()
        # dargs = darg.DefineArguments()
        # dargs.set_datastructures(tempast3)
        # dargs.define_arguments(transf_rp.NameSwap)
        # transf_rp.KernelArgs = dargs.kernel_args
        # transf_rp.Define = dargs.define_compound
        an.DefineArguments()
        an.PlaceInReg()
        an.PlaceInLocalMemory()
    if SetNoReadBack:
        tf.SetNoReadBack()

    ## rw.DataStructures()
    gen_full_code(name, an, tempast2)


if __name__ == "__main__":
    matmul()
    knearest()
