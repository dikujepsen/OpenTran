import os
from lan_ast import *

class CGenerator(object):
    """ Uses the same visitor pattern as the NodeVisitor, but modified to
        return a value from each visit method, using string accumulation in 
        generic_visit.
    """
    def __init__(self):
        self.output = ''
        
        # Statements start with indentation of self.indent_level spaces, using
        # the _make_indent method
        #
        self.indent_level = 0
        self.inside_ArgList = False
        self.inside_Assignment = False
        
    def createTemp(self, ast, filename = 'temp.cpp'):
        code = self.visit(ast)
        currentdir = os.getcwd()
        fullFilename = currentdir + '/' + filename
        try:
            os.remove(fullFilename)
        except OSError:
            pass
        try:
            fileobj =  open(fullFilename,'w')
            fileobj.write(code)
            fileobj.close()
        except IOError:
            print "Unable to write file"
       
    def not_simple_node(self, n):
        """ Returns True for nodes that are "simple"
        """
        return not isinstance(n, (Constant, Id, ArrayRef))
    
    def parenthesize_if(self, n, condition):
        """ Visits 'n' and returns its string representation, parenthesized
            if the condition function applied to the node returns True.
        """
        s = self.visit(n)
        if condition(n):
            return '(' + s + ')'
        else:
            return s


    def _make_indent(self):
        return ' ' * self.indent_level
    
    def visit(self, node):
        method = 'visit_' + node.__class__.__name__
        return getattr(self, method, self.generic_visit)(node)
    
    def generic_visit(self, node):
        if node is None:
            return ''
        else:
            return ''.join(self.visit(c[1]) if len(c) == 2 \
                           else self.visit(c) for c in node.children())
    
    def visit_FileAST(self, n):
        s = ''
        for ext in n.ext:
            if isinstance(ext, Compound):
                s += self.visit_GlobalCompound(ext)
            else:
                s += self.visit(ext) + '\n'
        return s

    
    def visit_GlobalCompound(self, n):
        s =  ''
        for stat in n.statements:
           s += self.visit(stat)
        s += '\n'
        return s


    def visit_GroupCompound(self, n):
        s =  ''
        for stat in n.statements:
           s += self.visit(stat) + '\n' + self._make_indent() 
        return s

    def visit_Comment(self, n):
        return n.value + '\n'

    def visit_Increment(self, n):
        s = self.visit(n.name)
        return s + n.op

    def visit_UnaryBefore(self, n):
        s = self.visit(n.expr)
        return n.op + s

    def visit_TypeId(self, n):
        s = self.visit(n.name)
        s1 = ' '.join(n.type)
        s1 += ' ' + s
        if not self.inside_ArgList:
            s1 += ';'
        return s1

    def visit_Assignment(self, n):
        self.inside_ArgList = True
        lval = self.visit(n.lval)
        self.inside_ArgList = False
        self.inside_Assignment = True
        rval = self.visit(n.rval)
        self.inside_Assignment = False
        return lval + ' ' + n.op + ' ' + rval + ';'

    def visit_Compound(self, n):
        s = '\n' + self._make_indent() + '{\n'
        self.indent_level += 2
        for stat in n.statements:
            s += self._make_indent() + self.visit(stat) + '\n'
        self.indent_level -= 2
        s += self._make_indent() + '}\n'
        return s

    def visit_ArgList(self, n):
        s = '('
        count = 1
        for arg in n.arglist:
            if count == 1:
                s += '\n\t'
            s += self.visit(arg) + ', '
            if count % 3 == 0:
                s += '\n\t'
            count += 1
        if n.arglist:
            s = s[:-2]
        if (count-1) % 3 == 0 and count != 1:
            s = s[:-2]
        return s + ')'

    def visit_ArrayRef(self, n):
        s = self.visit(n.name)
        for arg in n.subscript:
            s += '[' + self.visit(arg) + ']'
        return s

    def visit_BinOp(self, n):
        lval = self.parenthesize_if(n.lval,self.not_simple_node)
        rval = self.parenthesize_if(n.rval,self.not_simple_node)
        return lval + ' ' + n.op + ' ' + rval

    def visit_FuncDecl(self, n):
        self.inside_ArgList = True
        typeid = self.visit(n.typeid) 
        arglist = self.visit(n.arglist) 
        self.inside_ArgList = False
        if self.inside_Assignment:
            compound = ''
            end = ''
        elif n.compound.statements:
            typeid = '\n' + typeid
            compound = self.visit(n.compound)
        else:
            compound = ';'
        
        return typeid + arglist + compound

    def visit_ForLoop(self, n):
        init = self.visit(n.init) # already has a semi and \n at end
        init = init[:-1] # remove newline
        cond = self.visit(n.cond)
        inc = self.visit(n.inc)
        self.indent_level += 2
        compound = self.visit(n.compound)
        self.indent_level -= 2
        return 'for (' + init + ' ' + cond + '; ' + inc + ')' + compound
        
    def visit_Id(self, n):
        return n.name
        
    def visit_Constant(self, n):
        if isinstance(n.value,str):
            return '\"'+n.value+'\"'
        else:
            return str(n.value)
        
