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
from framework import cgen, transformation, analysis, lan
# from framework.lan_parser import *
# from framework.lan import *
# import framework.lan
import framework
import rewriter

import ply.yacc as yacc

SetNoReadBack = False
DoOptimizations = True


def LexAndParse(name, createTemp):

    cparser = yacc.yacc(module=framework.lan)
    lex.lex(module=framework.lan)

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
    # rw.initNewRepr(tempast)
    # tf = transformation.Transformation(rw)
    #
    # an = analysis.Analysis(rw, tf)
    # if DoOptimizations:
    #     an.Transpose()
    #     an.DefineArguments()
    #     an.PlaceInReg()
    #     an.PlaceInLocalMemory()
    # if SetNoReadBack:
    #     tf.SetNoReadBack()

    ## rw.DataStructures()
    # CGen(name, funcname, an, tempast2, ast)


if __name__ == "__main__":
    matmul()

# FileAST <top>:
#   TypeId <ext[0]>: type=['unsigned']
#     Id <name>: name=hA
#   TypeId <ext[1]>: type=['unsigned']
#     Id <name>: name=wB
#   TypeId <ext[2]>: type=['unsigned']
#     Id <name>: name=wA
#   TypeId <ext[3]>: type=['float', '*']
#     Id <name>: name=A
#   TypeId <ext[4]>: type=['float', '*']
#     Id <name>: name=B
#   TypeId <ext[5]>: type=['float', '*']
#     Id <name>: name=C
#   ForLoop <ext[6]>:
#     Assignment <init>: op==
#       TypeId <lval>: type=['unsigned']
#         Id <name>: name=i
#       Constant <rval>: value=0
#     BinOp <cond>: op=<
#       Id <lval>: name=i
#       Id <rval>: name=hA
#     Increment <inc>: op=++
#       Id <name>: name=i
#     Compound <compound>:
#       ForLoop <stmt[0]>:
#         Assignment <init>: op==
#           TypeId <lval>: type=['unsigned']
#             Id <name>: name=j
#           Constant <rval>: value=0
#         BinOp <cond>: op=<
#           Id <lval>: name=j
#           Id <rval>: name=wB
#         Increment <inc>: op=++
#           Id <name>: name=j
#         Compound <compound>:
#           Assignment <stmt[0]>: op==
#             TypeId <lval>: type=['float']
#               Id <name>: name=sum
#             Constant <rval>: value=0
#           ForLoop <stmt[1]>:
#             Assignment <init>: op==
#               TypeId <lval>: type=['unsigned']
#                 Id <name>: name=k
#               Constant <rval>: value=0
#             BinOp <cond>: op=<
#               Id <lval>: name=k
#               Id <rval>: name=wA
#             Increment <inc>: op=++
#               Id <name>: name=k
#             Compound <compound>:
#               Assignment <stmt[0]>: op=+=
#                 Id <lval>: name=sum
#                 BinOp <rval>: op=*
#                   ArrayRef <lval>:
#                     Id <name>: name=A
#                     BinOp <subscript 0>: op=+
#                       BinOp <lval>: op=*
#                         Id <lval>: name=i
#                         Id <rval>: name=wA
#                       Id <rval>: name=k
#                   ArrayRef <rval>:
#                     Id <name>: name=B
#                     BinOp <subscript 0>: op=+
#                       Id <lval>: name=j
#                       BinOp <rval>: op=*
#                         Id <lval>: name=k
#                         Id <rval>: name=wB
#           Assignment <stmt[2]>: op==
#             ArrayRef <lval>:
#               Id <name>: name=C
#               BinOp <subscript 0>: op=+
#                 BinOp <lval>: op=*
#                   Id <lval>: name=wB
#                   Id <rval>: name=i
#                 Id <rval>: name=j
#             Id <rval>: name=sum
