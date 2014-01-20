from lan_lex import *
from lan_ast import *
from rewriter import *
from os.path import basename
import copy
precedence = (
    ('left','PLUS','MINUS'),
    ('left','TIMES','DIVIDE'),
    )

    
  
def p_first(p):
    """ first : top_level
    """
    p[0] =  FileAST([]) if p[1] is None else FileAST(p[1])



def p_top_level(p):
    """ top_level :  top_level comment
    | top_level function_declaration
    | top_level declaration
    | top_level compound
    | top_level assignment_expression_semi
    | top_level expr
    | top_level for_loop
    | top_level include
    | empty
    """
    tmp1 =  [] if p[1] is None else p[1]
    if len(p) == 3:
        tmp2 = p[2] if isinstance(p[2], list) else [p[2]] 
    p[0] =  tmp1 + tmp2 if len(p) == 3 else p[1]



def p_comment(p):
    """comment : COMMENT"""
    p[0] = Comment(p[1],p.lineno(1))



def p_arg_params(p):
    """arg_params : term COMMA arg_params
    | typeid COMMA arg_params
    | binop
    | typeid
    | empty
    """
    if len(p) == 4:
        p[0] = [p[1]] + p[3]
    else:
        p[0] = [] if p[1] is None else [p[1]]

def p_arglist(p):
    """arglist : LPAREN arg_params RPAREN """
    p[0] = ArgList(p[2],p.lineno(1))

def p_assignment_operator(p):
    """assignment_operator :      EQUALS
    				| PLUSEQUALS
    				| MINUSEQUALS
    				| TIMESEQUALS
                                """
    p[0] = p[1]

def p_assignment_expression(p):
    """assignment_expression :    typeid assignment_operator expr
    				| identifier assignment_operator expr
    				| array_reference assignment_operator expr
                                """
    p[0] = Assignment(p[1], p[3], p[2], p.lineno(1))

def p_assignment_expression_semi(p):
    """assignment_expression_semi : assignment_expression SEMI """
    p[0] = p[1]


def p_constant(p):
    """constant : INT_CONST
    | FLOAT_CONST
    | STRING_LITERAL
    """
    p[0] = Constant(p[1],p.lineno(1))
    
def p_increment(p):
    """ increment : term unary_token_after  """    
    p[0] = Increment(p[1],p[2],p.lineno(1))



def p_binop(p):
    """ binop : LPAREN binop_expression RPAREN
    | binop_expression
    """
    if len(p) == 2:
        p[0] = p[1]
    else:
        p[0] = p[2]


def p_binop_expression(p):
    """ binop_expression : term
    | binop DIVIDE binop
    | binop TIMES binop
    | binop PLUS  binop
    | binop MINUS binop
    | binop MOD binop
    | binop OR binop
    | binop AND binop
    | binop LSHIFT binop
    | binop RSHIFT binop
    | binop LOGOR binop
    | binop LOGAND binop
    | binop LT binop
    | binop GT binop
    | binop LE binop
    | binop GE binop
    | binop EQ binop
    | binop NE binop


    """
    if len(p) == 2:
        p[0] = p[1]
    else:
        p[0] = BinOp(p[1], p[2], p[3], p.lineno(1))



def p_subscript(p):
    """ subscript : LBRACKET expr RBRACKET"""
    p[0] = p[2]

def p_subscript_list(p):
    """ subscript_list : subscript
    | subscript subscript_list"""
    if len(p) == 3:
        p[0] = [p[1]] + p[2]
    else:
        p[0] = [] if p[1] is None else [p[1]]


def p_array_reference(p):
    """ array_reference : identifier subscript_list
    """
    p[0] = ArrayRef(p[1], p[2], p.lineno(1))


def p_for_loop(p):
    """ for_loop : FOR LPAREN assignment_expression SEMI binop SEMI increment RPAREN compound
    """
    p[0] = ForLoop(p[3], p[5], p[7], p[9], p.lineno(1))

def p_unary_token_before(p):
    """unary_token_before :  MINUS 
    | LOGNOT
    """
    p[0] = p[1]

def p_unary_token_after(p):
    """unary_token_after :   PLUSPLUS
    | MINUSMINUS
    """
    p[0] = p[1]

def p_unary_expression(p):
    """ unary_expression : unary_token_before term"""    
    p[0] = UnaryBefore(p[1],p[2],p.lineno(1))



def p_term(p):
    """term : identifier
    | constant
    | array_reference
    | function_call
    | unary_expression
    """
    p[0] = p[1]


def p_compound(p):
    """compound : LBRACE top_level RBRACE """
    p[0] = Compound([] if p[2] is None else p[2],p.lineno(1))

def p_func_call(p):
    """ function_call : identifier arglist """
    p[0] = FuncDecl(p[1], p[2], Compound([]), p.lineno(1))

def p_func_decl_1(p):
    """function_declaration : typeid arglist SEMI"""
    p[0] = FuncDecl(p[1], p[2], Compound([],p.lineno(1)),p.lineno(1))

def p_func_decl_2(p):
    """function_declaration : typeid arglist compound """
    p[0] = FuncDecl(p[1], p[2], p[3], p.lineno(1))

def p_func_decl_3(p):
    """function_declaration : function_call SEMI """
    p[0] = p[1]

    


def p_decl_1(p):
    """declaration : typeid SEMI"""
    p[0] = p[1]

def p_decl_2(p):
    """declaration : array_typeid SEMI"""
    p[0] = p[1]
    
def p_typeid(p):
    """ typeid : type identifier"""
    p[0] = TypeId(p[1], p[2])

def p_array_typeid(p):
    """ array_typeid : type identifier subscript_list"""
    p[0] = ArrayTypeId(p[1], p[2], p[3])


def p_native_type(p):
    """native_type : VOID	
    | SIZE_T
    | UNKNOWN
    | CHAR
    | SHORT
    | INT
    | LONG
    | FLOAT
    | DOUBLE
    | SIGNED
    | UNSIGNED
    """
    p[0] = p[1]

def p_expr(p):
    """ expr : binop
    """
    p[0] = p[1]


def p_type(p):
    """type : native_type
	| native_type TIMES """
    
    p[0] = [p[1]] if len(p) == 2 else \
           [p[1]] + [p[2]]
    

def p_identifier(p):
    """ identifier : ID """
    p[0] = Id(p[1], p.lineno(1))


def p_include(p):
    """ include : PPHASH INCLUDE STRING_LITERAL """
    p[0] = Include(p[3])
    


def p_empty(p):
    'empty : '
    p[0] = None

    
def p_error(p):
    print("Syntax error at '%s'" % p.value)


from cgen import *

fileprefix = "../test/C/"

def jacobi():
    import ply.yacc as yacc
    cparser = yacc.yacc()
    # Build the lexer
    lex.lex()

    run = 1
    while run:
        filename = '../test/Jacobi/JacobiFor.cpp'
        ## filename = '../test/matmulfunc4.cpp'
        funcname = basename(os.path.splitext(filename)[0])
        try:
            ## f = open('../test/matmulfunc2.cpp', 'r')
            ## f = open('../test/matrixMul.cpp', 'r')
            ## f = open('../test/matmulfunc3.cpp', 'r')
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
        ## rw.rewrite(ast, funcname, changeAST = True)
        ## cprint.createTemp(ast, filename = 'tempjacobi.cpp')

        run = 0
        filename = '../src/tempjacobi.cpp'
        ## filename = '../src/temp.cpp'
        funcname = basename(os.path.splitext(filename)[0])
        try:
            f = open(filename, 'r')
            s = f.read()
            f.close()
            ## print s
        except EOFError:
            break
 
        ast = cparser.parse(s)
        ## ## ast.show()
        tempast = copy.deepcopy(ast)
        tempast2 = copy.deepcopy(ast)
        rw.initNewRepr(tempast)

        rw.rewriteToSequentialC(ast)
        cprint.createTemp(ast, filename = 'cmattemp.cpp')
        ## rw.rewriteToDeviceCTemp(tempast, False)
        ## cprint.createTemp(tempast, filename = 'devtemp.cpp')


        ## rw.transpose('A')
        ## rw.transpose('B')
        ## rw.transpose('C')
        rw.localMemory(['X1'], west = 1, north = 1, east = 1, south = 1, middle = 0)
        ## rw.localMemory('A')
        rw.dataStructures()
        rw.rewriteToDeviceCRelease(tempast2)
        ## cprint.createTemp(tempast2, filename = 'matmulfunc4.cl')
        cprint.createTemp(tempast2, filename = '../test/Jacobi/Jacobi.cl')
        boilerast = rw.generateBoilerplateCode(ast)
        cprint.createTemp(boilerast, filename = 'boilerplate.cpp')

def matmul():
    import ply.yacc as yacc
    cparser = yacc.yacc()
    # Build the lexer
    lex.lex()

    run = 1
    while run:
        filename = fileprefix + 'Matmul/matmulfunc4.cpp'
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
        ast.show()
        ## print ast
        cprint = CGenerator()
        rw = Rewriter()
        rw.initOriginal(ast)
        ## rw.rewrite(ast, funcname, changeAST = True)
        ## cprint.createTemp(ast, filename = 'tempmatmul.cpp')

        run = 0
        filename = '../src/tempmatmul.cpp'
        funcname = basename(os.path.splitext(filename)[0])
        try:
            f = open(filename, 'r')
            s = f.read()
            f.close()
            ## print s
        except EOFError:
            break
 
        ast = cparser.parse(s)
        ## ## ast.show()
        tempast = copy.deepcopy(ast)
        tempast2 = copy.deepcopy(ast)
        rw.initNewRepr(tempast)

        ## rw.rewriteToSequentialC(ast)
        ## cprint.createTemp(ast, filename = 'ctemp.cpp')
        ## rw.rewriteToDeviceCTemp(tempast, False)
        ## cprint.createTemp(tempast, filename = 'devtemp.cpp')


        ## rw.transpose('A')
        ## rw.transpose('B')
        ## rw.transpose('C')
        rw.localMemory(['A','B'])
        rw.dataStructures()
        rw.SetDefine(['hst_ptrB_dim1', 'hst_ptrA_dim1', 'wA', 'hst_ptrC_dim1'])
        
        rw.InSourceKernel(tempast2, filename = fileprefix + 'Matmul/matmulfunc4.cl')
        boilerast = rw.generateBoilerplateCode(ast)
        cprint.createTemp(boilerast, filename = fileprefix + 'Matmul/boilerplate.cpp')

def nbody():
    import ply.yacc as yacc
    cparser = yacc.yacc()
    lex.lex()

    run = 1
    while run:
        filename = fileprefix + 'NBody/NBodyFor.cpp'
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
        tempfilename = fileprefix + 'NBody/tempnbody.cpp'
        ## rw.rewrite(ast, funcname, changeAST = True)
        ## cprint.createTemp(ast, filename = tempfilename)

        run = 0
        filename = tempfilename
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
        rw.initNewRepr(tempast)

        ## rw.rewriteToSequentialC(ast)
        ## cprint.createTemp(ast, filename = 'ctemp.cpp')
        ## rw.rewriteToDeviceCTemp(tempast, False)
        ## cprint.createTemp(tempast, filename = 'devtemp.cpp')

        rw.dataStructures()

        rw.SetLSIZE(['256'])
        rw.SetDefine(['hst_ptrForces_dim1', 'hst_ptrPos_dim1', 'N'])
        rw.Unroll2({'j' : 32})
        
        rw.InSourceKernel(tempast2, filename = fileprefix + 'NBody/'+funcname + '.cl')
        ## rw.rewriteToDeviceCRelease(tempast2)
        ## cprint.createTemp(tempast2, filename = fileprefix + 'NBody/'+funcname + '.cl')
        boilerast = rw.generateBoilerplateCode(ast)
        cprint.createTemp(boilerast, filename = fileprefix + 'NBody/'+'boilerplate.cpp')

def nbody2():
    import ply.yacc as yacc
    cparser = yacc.yacc()
    lex.lex()

    run = 1
    while run:
        filename = fileprefix + 'NBody2/NBody2For.cpp'
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
        tempfilename = fileprefix + 'NBody2/'+'tempnbody2.cpp'
        ## rw.rewrite(ast, funcname, changeAST = True)
        ## cprint.createTemp(ast, filename = tempfilename)

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
        rw.initNewRepr(tempast)


        rw.SetLSIZE(['256','1'])
        ## rw.localMemory(['Pos'], south = 1, middle = 1)
        rw.SetDefine(['hst_ptrForces_x_dim1', 'hst_ptrForces_y_dim1', 'hst_ptrPos_dim1'])
        rw.dataStructures()
        ## rw.localMemory2(['Mas', 'Pos'])
        ## rw.constantMemory(['Pos'])
        ## rw.SetNoReadBack()
        ## rw.constantMemory2({'Pos' : [2,3], 'Mas' : [1]})
        ## rw.placeInReg2({ 'Pos' : [0, 1], 'Mas' : [0]})
        ## rw.Unroll(['k', 'kk'])
        
        rw.InSourceKernel(tempast2, filename = fileprefix + 'NBody2/'+funcname + '.cl')
        ## rw.rewriteToDeviceCRelease(tempast)
        ## cprint.createTemp(tempast, filename = fileprefix + 'NBody2/'+funcname + '.cl')
        boilerast = rw.generateBoilerplateCode(ast)
        cprint.createTemp(boilerast, filename = fileprefix + 'NBody2/'+'boilerplate.cpp')

def knearest():
    import ply.yacc as yacc
    cparser = yacc.yacc()
    lex.lex()

    run = 1
    while run:
        filename = fileprefix + 'KNearest/KNearestFor.cpp'
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
        tempfilename = fileprefix + 'KNearest/'+'tempknearest.cpp'
        ## rw.rewrite(ast, funcname, changeAST = True)
        ## cprint.createTemp(ast, filename = tempfilename)

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
        rw.SetParDim(1)
        rw.initNewRepr(tempast)

        
        
        ## rw.constantMemory(['Pos']) 
        #rw.transpose('train_patterns')
        rw.transpose('test_patterns')

        #rw.localMemory(['train_patterns'])
        #rw.localMemory2(['train_patterns'])

        
        ## rw.localMemory(['test_patterns','train_patterns'])
        ## rw.transpose('dist_matrix')
        rw.SetDefine(['dim', 'hst_ptrtest_patterns_dim1',
                   'hst_ptrtrain_patterns_dim1', 'hst_ptrdist_matrix_dim1',
                   'NTRAIN'])
        rw.placeInReg3({'test_patterns': [0]})
        rw.SetNoReadBack()
        ## rw.dataStructures()
        ## rw.Unroll2({'k' : 0})
        
        rw.InSourceKernel(tempast2, filename = fileprefix + 'KNearest/'+funcname + '.cl')
        boilerast = rw.generateBoilerplateCode(ast)
        cprint.createTemp(boilerast, filename = fileprefix + 'KNearest/'+'boilerplate.cpp')

def gaussian():
    import ply.yacc as yacc
    cparser = yacc.yacc()
    # Build the lexer
    lex.lex()

    run = 1
    while run:
        filename = fileprefix + 'GaussianDerivates/GaussianDerivatesFor.cpp'
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
        cprint = CGenerator()
        rw = Rewriter()
        rw.initOriginal(ast)
        tempfilename = fileprefix + 'GaussianDerivates/'+'tempgaussian.cpp'
        ## rw.rewrite(ast, funcname, changeAST = True)
        ## cprint.createTemp(ast, filename = tempfilename)

        run = 0
        filename = tempfilename
        ## funcname = basename(os.path.splitext(filename)[0])

        try:
            f = open(filename, 'r')
            s = f.read()
            f.close()
            ## print s
        except EOFError:
            break

        ast = cparser.parse(s)
        ## ## ast.show()
        tempast = copy.deepcopy(ast)
        tempast2 = copy.deepcopy(ast)

        ## rw.SetParDim(1)
        rw.initNewRepr(tempast)

        rw.SetLSIZE(['16', '16'])
        ## rw.rewriteToSequentialC(ast)
        ## cprint.createTemp(ast, filename = 'ctemp.cpp')
        ## rw.rewriteToDeviceCTemp(tempast, False)
        ## cprint.createTemp(tempast, filename = 'devtemp.cpp')


        rw.transpose('p_a_i_x')
        rw.transpose('q_a_i_x')
        rw.SetDefine(['dim', 'scaleweight2_x', 'hst_ptrp_a_i_x_dim1',
                      'hst_ptrK__ij_x_dim1', 'scales2_x',
                      'hst_ptrq_a_i_x_dim1'])
        rw.Unroll2({'k' : 0, 'd' : 0, 'g' : 0, 'b' : 0})
        ## rw.transpose('C')
        ## rw.localMemory(['A','B'])
        #rw.dataStructures()
        rw.SetNoReadBack()
        ## rw.rewriteToDeviceCRelease(tempast2)
        # fileprefix + 'GaussianDerivates/
        rw.InSourceKernel(tempast2, filename = fileprefix + 'GaussianDerivates/'+funcname + '.cl')
        
        boilerast = rw.generateBoilerplateCode(ast)
        cprint.createTemp(boilerast, filename = fileprefix + 'GaussianDerivates/'+'boilerplate.cpp')
## 0.00325902 -0 0.032143 -0.0251829 -0 0 0.0489179 -0.00325902 0.00212156 -0.0879844 
## -0.0118921 -0 -0.0118921 -0 -0.00325902 -0 0 0.00741233 -0.00453474 0.00325902 
## -0.015038 -0.0016064 -0.0286717 -0.0141325 -0.00690135 -0.0141325 0.00741234 -0.00741234 0.00741234 -0.00449265 
## 0.0101567 -0.00212156 0.0141325 0.0504732 -0.0633 -0 -0.0204797 -0.0186647 0.0229105 0.0489179 
## -0 -0 -0.0124129 -0.019482 -0.0124129 -0.0607155 -0.0296494 -0.169504 0.0113864 -0.00221446 
## 0.00556629 -0.142482 0 0.031201 0.0095137 0 0.0141325 

def laplace():
    import ply.yacc as yacc
    cparser = yacc.yacc()
    lex.lex()

    run = 1
    while run:
        filename = fileprefix + 'Laplace/LaplaceFor.cpp'
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
        tempfilename = fileprefix + 'Laplace/'+'templaplace.cpp'
        ## rw.rewrite(ast, funcname, changeAST = True)
        ## cprint.createTemp(ast, filename = tempfilename)

        run = 0
        funcname = basename(os.path.splitext(filename)[0])
        filename = tempfilename
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
        rw.initNewRepr(tempast)


        rw.SetLSIZE(['256'])
        ## rw.localMemory(['Pos'], south = 1, middle = 1)
        rw.dataStructures()
        ## rw.placeInReg2({'level': [0], 'level_int' : [0], 'index' : [0]})
        rw.transpose('level')
        rw.transpose('level_int')
        rw.transpose('index')
        rw.SetDefine(['dim', 'hst_ptrlevel_dim1', 'hst_ptrindex_dim1',
                      'storagesize', 'hst_ptrlevel_int_dim1'])
        ## rw.SetDefine(['dim'])

        rw.placeInReg3({'level': [0], 'level_int' : [0], 'index' : [0]})
        ## rw.localMemory(['level'])
        # rw.localMemory2(['alpha'])
        ## rw.placeInReg2({ 'alpha_local' : [1]})
        ## rw.constantMemory(['Pos'])
        # rw.SetNoReadBack()
        ## rw.constantMemory2({'Pos' : [2,3], 'Mas' : [1]})
        ## rw.placeInReg2({ 'Pos' : [0, 1], 'Mas' : [0]})
        # rw.Unroll2({'d' : 0, 'd_outer' : 0, 'd_inner' : 0})
        
        rw.InSourceKernel(tempast2, filename = fileprefix + 'Laplace/'+funcname + '.cl')
        boilerast = rw.generateBoilerplateCode(ast)
        cprint.createTemp(boilerast, filename = fileprefix + 'Laplace/'+'boilerplate.cpp')

if __name__ == "__main__":
    ## jacobi()
    ## matmul()
    ## nbody2()
    ## laplace()
    knearest()
    ## gaussian()
