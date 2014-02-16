from rewriter import *
from os.path import basename
import copy
from lan_parser import *
from cgen import *
from transformation import *
from analysis import *
fileprefix = "../test/C/"

SetNoReadBack = False

def LexAndParse(name, createTemp):
    import ply.yacc as yacc
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
        cprint = CGenerator()
        ## printres = cprint.visit(ast)
        ## print printres
        rw = Rewriter()
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
        cprint = CGenerator()
        rw = an.rw
        an.GenerateKernels(tempast2, name, fileprefix)
        ## rw.InSourceKernel(tempast2, filename = fileprefix + name + '/'+funcname + '.cl', kernelstringname = kernelstringname)
        boilerast = rw.generateBoilerplateCode(ast)
        cprint.createTemp(boilerast, filename = fileprefix + name + '/'+'boilerplate.cpp')


def jacobi():
    name = 'Jacobi'
    (rw, ast, tempast, tempast2, funcname) = LexAndParse(name, True)
    rw.initNewRepr(tempast)
    tf = Transformation(rw)
    
    an = Analysis(rw, tf)
    an.Transpose()
    an.DefineArguments()
    an.PlaceInReg()

    tf.localMemory(['X1'], west = 1, north = 1, east = 1, south = 1, middle = 0)
    if SetNoReadBack:
        tf.SetNoReadBack()

    an.PlaceInLocalMemory()
    CGen(name, funcname, an, tempast2, ast)

def matmul():
    name = 'MatMul'
    (rw, ast, tempast, tempast2, funcname) = LexAndParse(name, True)
    rw.initNewRepr(tempast)
    tf = Transformation(rw)

    an = Analysis(rw, tf)
    an.Transpose()
    an.DefineArguments()
    an.PlaceInReg()
    if SetNoReadBack:    
        tf.SetNoReadBack()
        
    ## rw.DataStructures()
    an.PlaceInLocalMemory()
    CGen(name, funcname, an, tempast2, ast)
    

def nbody():
    name = 'NBody'
    (rw, ast, tempast, tempast2, funcname) = LexAndParse(name, True)

    rw.initNewRepr(tempast)
    tf = Transformation(rw)
    an = Analysis(rw, tf)
    an.Transpose()
    an.DefineArguments()
    an.PlaceInReg()

    if SetNoReadBack:
        tf.SetNoReadBack()
    ## rw.Unroll2({'j': 32})
    an.PlaceInLocalMemory()
    CGen(name, funcname, an, tempast2, ast)

def knearest():
    name = 'KNearest'
    (rw, ast, tempast, tempast2, funcname) = LexAndParse(name, True)
    tf = Transformation(rw)
    tf.SetParDim(1)
    rw.initNewRepr(tempast)
    
    an = Analysis(rw, tf)
    an.Transpose()
    an.DefineArguments()
    an.PlaceInReg()
    if SetNoReadBack:
        tf.SetNoReadBack()
    ## rw.DataStructures()
    ## rw.Unroll2({'k' : 0})
    
    an.PlaceInLocalMemory()
    CGen(name, funcname, an, tempast2, ast)

def gaussian():
    name = 'GaussianDerivates'
    (rw, ast, tempast, tempast2, funcname) = LexAndParse(name, True)
    ## rw.SetParDim(1)
    rw.initNewRepr(tempast)
    tf = Transformation(rw)

    
    an = Analysis(rw, tf)
    an.Transpose()
    an.DefineArguments()
    an.PlaceInReg()
    ## tf.Unroll2({'k' : 0, 'd' : 0, 'g' : 0, 'b' : 0})
    ## rw.DataStructures()
    if SetNoReadBack:
        tf.SetNoReadBack()
    an.PlaceInLocalMemory()
    CGen(name, funcname, an, tempast2, ast)

def laplace():
    name = 'Laplace'
    (rw, ast, tempast, tempast2, funcname) = LexAndParse(name, True)
    tf = Transformation(rw)
    tf.SetParDim(1)
    rw.initNewRepr(tempast)
    an = Analysis(rw, tf)
    an.Transpose()
    an.DefineArguments()
    an.PlaceInReg()
    
    if SetNoReadBack:
        tf.SetNoReadBack()
    
    rw.DataStructures()
    
    ## tf.Unroll2({'d' : 0, 'd_outer' : 0, 'd_inner' : 0})
    an.PlaceInLocalMemory()
    CGen(name, funcname, an, tempast2, ast)
    

if __name__ == "__main__":
    jacobi()
    matmul()
    nbody()
    laplace()
    knearest()
    gaussian()
