import os

import lan
import transf_repr
import transf_visitor as tvisitor
import cgen

debug = False


class SSGenerator(object):
    """ Uses the same visitor pattern as the NodeVisitor, but modified to
        return a stringstream from each visit method, using string accumulation in 
        generic_visit.
    """
    def __init__(self):
        self.output = ''
        self.quotes = '\\"'
        self.newline = '" << endl;\n'
        self.semi = ';'
        self.start = 'str << "'
        self.UnrollLoops = []
        self.statements = []
    
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
            print "createTemp: Unable to write file"
            
    def createKernelStringStream(self, ast, newast, UnrollLoops, kernelstringname, filename = 'temp.cpp'):

        self.UnrollLoops = UnrollLoops
        
        # Swap the IDs that contain the loop indexes before
        # generating the code
        swapUnrollID = tvisitor.SwapUnrollID(self.UnrollLoops)
        swapUnrollID.visit(ast)
        oldcode = self.visit(ast)
        
        
        split = oldcode.split('\n')
        for s in split:
            self.statements.append(lan.Id(s))

        
            
        # Create the function where we generate the code for the
        # kernel function
        kernelfunc = transf_repr.EmptyFuncDecl(kernelstringname, type = ['std::string'])
        # insert the stringstream typeid
        self.statements.insert(0, lan.TypeId(['std::stringstream'], lan.Id('str')))
        kernelfunc.compound.statements = self.statements
        # insert conversion to std::string
        self.statements.append(lan.Id('return str.str();'))
        
        newast.ext = [kernelfunc]
        cprint = cgen.CGenerator()
        cprint.print_ast(newast, filename = filename)
            
            

    def simple_node(self, n):
        """ Returns True for nodes that are "simple"
        """
        return not isinstance(n, (lan.Constant, lan.Id, lan.ArrayRef))
    
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
        newline = self.newline
        start = self.start
        if debug:
            newline = n.__class__.__name__ +  newline
            start = n.__class__.__name__ + start
        s = ''
        for ext in n.ext:
            if isinstance(ext, lan.Compound):
                s += self.visit_GlobalCompound(ext)
            else:
                s += self.visit(ext)
        return s

    
    def visit_GlobalCompound(self, n):
        newline = self.newline
        start = self.start
        if debug:
            newline = n.__class__.__name__ +  newline
            start = n.__class__.__name__ + start
        s = ''
        for stat in n.statements:
           s += self.visit(stat)
        s = start + s + self.newline
        return s


    def visit_GroupCompound(self, n):
        newline = self.newline
        start = self.start
        
        if debug:
            newline = n.__class__.__name__ +  newline
            start = n.__class__.__name__ + start
        s = ''
        for i,stat in enumerate(n.statements):
            start1 = ''
            ## if i != 0:
            newline1 = ''
            if not isinstance(stat, lan.ForLoop):
                newline1 = newline
                start1 = start+ self._make_indent()
            s += start1  + self.visit(stat) + newline1
        return s

    def visit_Comment(self, n):
        return n.value 

    def visit_Increment(self, n):
        s = self.visit(n.name)
        return s + n.op

    def visit_UnaryBefore(self, n):
        s = self.visit(n.expr)
        return n.op + s

    def visit_TypeId(self, n):
        s1 = self.visit(n.name)
        if n.type:
            s = ' '.join(n.type)
            s1 = s + ' ' + s1
        if not self.inside_ArgList:
            s1 += self.semi
        return s1

    def visit_ArrayTypeId(self, n):
        s = self.visit(n.name)
        s1 = ' '.join(n.type)
        s1 += ' ' + s
        for arg in n.subscript:
            s1 += '[' + self.visit(arg) + ']'
        if not self.inside_ArgList:
            s1 += self.semi
        return s1

    def visit_Assignment(self, n):
        self.inside_ArgList = True
        lval = self.visit(n.lval)
        self.inside_ArgList = False
        self.inside_Assignment = True
        rval = self.visit(n.rval)
        self.inside_Assignment = False
        return lval + ' ' + n.op + ' ' + rval + self.semi

    def visit_ArrayInit(self, n):
        s = '{'
        for stat in n.values:
            s += self.visit(stat) + ', '
        s = s[:-2]
        s += '}'
        return s

    def visit_Compound(self, n):
        start = self.start
        newline = self.newline
        if debug:
            newline = n.__class__.__name__ +  newline
            start = n.__class__.__name__ +  start
            
        s = '' #start + self._make_indent() + '{' + newline
        self.indent_level += 2
        for stat in n.statements:
            if isinstance(stat, lan.ForLoop) or isinstance(stat, lan.GroupCompound):
                s += self.visit(stat)
            else:
                s += start + self._make_indent() + self.visit(stat) + newline 
        self.indent_level -= 2
        ## s += start + self._make_indent() + '}'  
        return s

    def visit_ArgList(self, n):
        newline = self.newline
        start = self.start
        if debug:
            newline = n.__class__.__name__ +  newline
            start = n.__class__.__name__ +  start

        s = '('
        count = 1
        if len(n.arglist) == 1:
            retval = '(' + self.visit(n.arglist[0]) + ')'
            return retval
            
        for arg in n.arglist:
            if count == 1:
                s += newline + start  + '\t' 
            s += self.visit(arg)
            if count != (len(n.arglist)):
                s += ', '
            if count % 3 == 0:
                s += newline + start + '\t' 
            count += 1
        ## if n.arglist:
        ##     s = s[:-2]
        ## if (count-1) % 3 == 0 and count != 1:
        ##     s = s[:-2]
        return s + ')'

    def visit_ArrayRef(self, n):
        s = self.visit(n.name)
        for arg in n.subscript:
            s += '[' + self.visit(arg) + ']'
        return s

    def visit_BinOp(self, n):
        lval = self.parenthesize_if(n.lval,self.simple_node)
        rval = self.parenthesize_if(n.rval,self.simple_node)
        return lval + ' ' + n.op + ' ' + rval

    def visit_FuncDecl(self, n):
        newline = self.newline
        start = self.start
        if debug:
            newline = n.__class__.__name__ +  newline
            start = n.__class__.__name__ +  start
            
        self.inside_ArgList = True
        typeid = self.visit(n.typeid)
        
        arglist = self.visit(n.arglist)

        self.inside_ArgList = False
        if self.inside_Assignment:
            compound = ''
            end = ''
        elif n.compound.statements:
            typeid = start + typeid
            arglist += ' {' + newline
            compound = self.visit(n.compound) + start + '}' + newline
        else:
            compound = self.semi
        retval = typeid + arglist + compound  
        return retval

    def visit_ForLoop(self, n):
        newline = self.newline
        start = self.start
        if debug:
            newline = n.__class__.__name__ +  newline
            start = n.__class__.__name__ +  start

        name = n.init.lval.name.name
        init = self.visit(n.init) # already has a semi at the end
        cond = self.visit(n.cond)
        inc = self.visit(n.inc)

        self.indent_level += 2
        compound = self.visit(n.compound)
        self.indent_level -= 2
        if name in self.UnrollLoops:
            start = ''
            newline = '\n'
        return start + self._make_indent() + 'for (' + init + ' ' + cond + self.semi + ' ' + inc + ') {'\
          + newline  + compound + start + self._make_indent() + '}' + newline

    def visit_IfThen(self, n):
        newline = self.newline
        start = self.start
        if debug:
            newline = n.__class__.__name__ +  newline
            start = n.__class__.__name__ + start

        cond = self.visit(n.cond)
        self.indent_level += 2
        compound = self.visit(n.compound)
        self.indent_level -= 2
        return 'if (' + cond + ')' + newline + compound

        
    def visit_Id(self, n):
        return n.name

    def visit_Include(self, n):
        newline = self.newline
        start = self.start
        if debug:
            newline = n.__class__.__name__ +  newline
            start = n.__class__.__name__ + start
        return start + "#include " + '\\"'+ n.name[1:-1] + '\\"' + newline
 
    
    def visit_Constant(self, n):
        try:
            s = float(n.value)
        except ValueError:
            if len(n.value) == 0:
                return '\"\"'
            if n.value[0] == '"':
                ## if self.extraquotes and False:
                ##     return self.quotes + n.value[1:-1] + self.quotes
                
                return n.value
            else:
                return self.quotes + n.value + self.quotes
        else:
            return str(n.value)
        
