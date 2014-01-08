import sys

import ply.lex as lex
from ply.lex import TOKEN
##
## Reserved keywords
##
keywords = (
    # Types
    'CHAR', 'DOUBLE', 'FLOAT', 'INT', 'LONG', 'UNKNOWN',
    'SHORT', 'SIGNED', 'UNSIGNED', 'VOID', 'SIZE_T',
    # Control flow
    'FOR', 'RETURN',
    
    # C internals
    'SIZEOF', 'INCLUDE'
    )

tokens = keywords + (
    # Identifier
    'ID',
    # Type of variables/functions, not needed, covered by keywords
    # 'TYPE', 

    # Different types of constants
    'FLOAT_CONST', 'INT_CONST',

    # Strings, e.g. "here is a string"
    'STRING_LITERAL',

    # Operators
    # arithmetic
    'PLUS','MINUS','TIMES','DIVIDE', 'MOD',
    'OR', 'AND', 'LSHIFT', 'RSHIFT',
    # comparison
    'LOGOR', 'LOGAND', 'LOGNOT',
    'LT', 'LE', 'GT', 'GE', 'EQ', 'NE',

    # Assignments
    'EQUALS', 'PLUSEQUALS', 'MINUSEQUALS', 'TIMESEQUALS',

    # Increments/decrements
    'PLUSPLUS', 'MINUSMINUS',
    
    # Delimeters 
    'LPAREN', 'RPAREN',         # ( )
    'LBRACKET', 'RBRACKET',     # [ ]
    'LBRACE', 'RBRACE',         # { } 
    'COMMA', 'PERIOD',          # , .
    'SEMI', 'COLON',            # ; :

    # pre-processor 
    'PPHASH',      # '#'

    # Save comments also
    'COMMENT',

    )


# valid C identifiers (K&R2: A.2.3)
floating_point = r"""([0-9]*\.[0-9]+)|([0-9]+\.)"""
integer = r"""0|([1-9][0-9]*)"""

identifier = r"""[a-zA-Z_][a-zA-Z0-9_]*"""

string_characters = r"""[a-zA-Z0-9_\+\.,: \t;!=<>\"\#\-@$%&/\{\}\(\)\[\]\?\*]*"""
comment_literal = '//' + string_characters #+ r'\n'
@TOKEN(comment_literal)
def t_COMMENT(t):
    t.lexer.lineno += 1
    return t


@TOKEN(identifier)
def t_ID(t):
    tmp = t.value.upper()
    if tmp in keywords:
        t.type = tmp
    return t

@TOKEN(floating_point)
def t_FLOAT_CONST(t):
    return t

@TOKEN(integer)
def t_INT_CONST(t):
    return t


t_STRING_LITERAL = '\"' + string_characters + '\"'


# Operators
t_PLUS    	= r'\+'
t_MINUS   	= r'-'
t_TIMES   	= r'\*'
t_DIVIDE  	= r'/'
t_MOD           = r'%'
t_OR            = r'\|'
t_AND           = r'&'
t_LSHIFT        = r'<<'
t_RSHIFT        = r'>>'
t_LOGOR         = r'\|\|'
t_LOGAND        = r'&&'
t_LOGNOT        = r'!'
t_LT            = r'<'
t_GT            = r'>'
t_LE            = r'<='
t_GE            = r'>='
t_EQ            = r'=='
t_NE            = r'!='

# Assignments
t_EQUALS  	= r'='
t_TIMESEQUALS    = r'\*='
t_PLUSEQUALS     = r'\+='
t_MINUSEQUALS    = r'-='

# Increment/decrement
t_PLUSPLUS      = r'\+\+'
t_MINUSMINUS    = r'--'

# Delimiters
t_LPAREN  	= r'\('
t_RPAREN  	= r'\)'
t_LBRACKET      = r'\['
t_RBRACKET      = r'\]'
t_LBRACE        = r'\{'
t_RBRACE        = r'\}'
t_COMMA         = r','
t_PERIOD        = r'\.'
t_SEMI          = r';'
t_COLON         = r':'
t_PPHASH        = r'\#'

# Ignored characters
t_ignore = " \t"


def t_newline(t):
    r'\n+'
    t.lexer.lineno += t.value.count("\n")

def t_returnnewline(t):
    r'\r\n+'
    t.lexer.lineno += t.value.count("\n")

def t_error(t):
    print ' Illegal character \'%s\'' % t.value[0]
    print t
    t.lexer.skip(1)
    
if __name__ == "__main__":
    # Build the lexer
    lex.lex()

    run = 1
    while run:
        try:
            f = open('../test/matmulfunc.cpp', 'r')
            ##f = open('../test/matrixMul.cpp', 'r')
            s = f.read()
            f.close()
            print s
            ## s = raw_input('calc > ')   # use input() on Python 3
        except EOFError:
            break
        lex.input(s)
        while 1:
            tok = lex.token()
            if not tok: break
            print tok
        run = 0
