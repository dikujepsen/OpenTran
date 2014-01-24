from rewriter import *
from os.path import basename
import copy
from lan_parser import *
from cgen import *
from transformation import *
fileprefix = "../test/C/"



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

def CGen(name, funcname, rw, tempast2, ast):
        cprint = CGenerator()
        rw.InSourceKernel(tempast2, filename = fileprefix + name + '/'+funcname + '.cl')
        boilerast = rw.generateBoilerplateCode(ast)
        cprint.createTemp(boilerast, filename = fileprefix + name + '/'+'boilerplate.cpp')


def jacobi():
    name = 'Jacobi'
    (rw, ast, tempast, tempast2, funcname) = LexAndParse(name, False)
    rw.initNewRepr(tempast)
    tf = Transformation(rw)
    
    ## rw.transpose('A')
    ## rw.transpose('B')
    ## rw.transpose('C')
    tf.localMemory(['X1'], west = 1, north = 1, east = 1, south = 1, middle = 0)
    tf.SetDefine(['hst_ptrB_dim1', 'hst_ptrX2_dim1', 'wA'])
    ## rw.localMemory('A')
    ## rw.dataStructures()

    CGen(name, funcname, rw, tempast2, ast)

def matmul():
    name = 'MatMul'
    (rw, ast, tempast, tempast2, funcname) = LexAndParse(name, False)
    rw.initNewRepr(tempast)
    tf = Transformation(rw)
    ## rw.rewriteToSequentialC(ast)
    ## cprint.createTemp(ast, filename = 'ctemp.cpp')
    ## rw.rewriteToDeviceCTemp(tempast, False)
    ## cprint.createTemp(tempast, filename = 'devtemp.cpp')


    ## rw.transpose('A')
    ## rw.transpose('B')
    ## rw.transpose('C')
    ## rw.localMemory(['A','B'])
    tf.localMemory3({'A' : [0], 'B' : [0]})
    ## rw.dataStructures()
    tf.SetDefine(['hst_ptrB_dim1', 'hst_ptrA_dim1', 'wA', 'hst_ptrC_dim1'])
    tf.SetNoReadBack()
        
    CGen(name, funcname, rw, tempast2, ast)
    

def nbody():
    name = 'NBody'
    (rw, ast, tempast, tempast2, funcname) = LexAndParse(name, False)

    rw.initNewRepr(tempast)
    tf = Transformation(rw)
    ## rw.SetLSIZE(['256'])
    tf.SetDefine(['hst_ptrForces_dim1', 'N', 'hst_ptrPos_dim1'])
    ## rw.dataStructures()
    ## rw.localMemory2(['Mas', 'Pos'])
    ## rw.localMemory3({'Mas' : [1] , 'Pos' : [2,3]})
    ## rw.SetNoReadBack()
    ## rw.Unroll2({'j': 32})
    CGen(name, funcname, rw, tempast2, ast)

def knearest():
    name = 'KNearest'
    (rw, ast, tempast, tempast2, funcname) = LexAndParse(name, False)
    tf = Transformation(rw)
    tf.SetParDim(1)
    rw.initNewRepr(tempast)
    
    ## rw.constantMemory(['Pos']) 
    ## rw.transpose('train_patterns')
    tf.transpose('test_patterns')
    
    ## rw.localMemory(['train_patterns'])
    
        
    ## rw.localMemory(['test_patterns','train_patterns'])
    ## rw.transpose('dist_matrix')
    tf.SetDefine(['dim', 'hst_ptrtest_patterns_dim1',
                  'hst_ptrtrain_patterns_dim1', 'hst_ptrdist_matrix_dim1',
                  'NTRAIN'])
    tf.placeInReg3({'test_patterns': [0]})
    tf.SetNoReadBack()
    ## rw.dataStructures()
    ## rw.Unroll2({'k' : 0})
    
    CGen(name, funcname, rw, tempast2, ast)

def gaussian():
    name = 'GaussianDerivates'
    (rw, ast, tempast, tempast2, funcname) = LexAndParse(name, False)
    ## rw.SetParDim(1)
    rw.initNewRepr(tempast)
    tf = Transformation(rw)

    tf.SetLSIZE(['16', '16'])
    ## rw.rewriteToSequentialC(ast)
    ## cprint.createTemp(ast, filename = 'ctemp.cpp')
    ## rw.rewriteToDeviceCTemp(tempast, False)
    ## cprint.createTemp(tempast, filename = 'devtemp.cpp')
    

    tf.transpose('p_a_i_x')
    tf.transpose('q_a_i_x')
    tf.SetDefine(['dim', 'scaleweight2_x', 'hst_ptrp_a_i_x_dim1',
                  'hst_ptrK__ij_x_dim1', 'scales2_x',
                  'hst_ptrq_a_i_x_dim1'])
    tf.Unroll2({'k' : 0, 'd' : 0, 'g' : 0, 'b' : 0})
    ## rw.transpose('C')
    ## rw.localMemory(['A','B'])
    #rw.dataStructures()
    #rw.SetNoReadBack()
    CGen(name, funcname, rw, tempast2, ast)

def laplace():
    name = 'Laplace'
    (rw, ast, tempast, tempast2, funcname) = LexAndParse(name, False)
    rw.initNewRepr(tempast)
    tf = Transformation(rw)
    tf.SetLSIZE(['256'])
    ## rw.dataStructures()
    tf.transpose('level')
    tf.transpose('level_int')
    tf.transpose('index')
    tf.SetDefine(['dim', 'hst_ptrlevel_dim1', 'hst_ptrindex_dim1',
                  'storagesize', 'hst_ptrlevel_int_dim1'])
    
    tf.placeInReg3({'level': [0], 'level_int' : [0], 'index' : [0]})
    tf.SetNoReadBack()

    rw.DataStructures()
    # rw.Unroll2({'d' : 0, 'd_outer' : 0, 'd_inner' : 0})
    CGen(name, funcname, rw, tempast2, ast)
    

if __name__ == "__main__":
    jacobi()
    matmul()
    nbody()
    laplace()
    knearest()
    gaussian()
