from lan_ast import *

precedence = (
    ('left', 'PLUS', 'MINUS'),
    ('left', 'TIMES', 'DIVIDE'),
)


def p_first(p):
    """ first : top_level
    """
    p[0] = FileAST([]) if p[1] is None else FileAST(p[1])


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
    tmp1 = [] if p[1] is None else p[1]
    if len(p) == 3:
        tmp2 = p[2] if isinstance(p[2], list) else [p[2]]
    p[0] = tmp1 + tmp2 if len(p) == 3 else p[1]


def p_comment(p):
    """comment : COMMENT"""
    p[0] = Comment(p[1], p.lineno(1))


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
    p[0] = ArgList(p[2], p.lineno(1))


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
    p[0] = Constant(p[1], p.lineno(1))


def p_increment(p):
    """ increment : term unary_token_after  """
    p[0] = Increment(p[1], p[2], p.lineno(1))


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
    p[0] = UnaryBefore(p[1], p[2], p.lineno(1))


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
    p[0] = Compound([] if p[2] is None else p[2], p.lineno(1))


def p_func_call(p):
    """ function_call : identifier arglist """
    p[0] = FuncCall(p[1], p[2], p.lineno(1))


# def p_func_decl_1(p):
#     """function_declaration : typeid arglist SEMI"""
#     p[0] = FuncDecl(p[1], p[2], Compound([], p.lineno(1)), p.lineno(1))


def p_func_decl_1(p):
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
    | native_type TIMES
    """

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
