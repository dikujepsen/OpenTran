import os
import lan

debug = False


class CGenerator(object):
    """ Uses the same visitor pattern as the NodeVisitor, but modified to
        return a value from each visit method, using string accumulation in 
        generic_visit.
    """

    def __init__(self):
        self.output = ''
        self.quotes = '\"'
        self.newline = '\n'
        self.semi = ';'
        self.start = ''

        # Statements start with indentation of self.indent_level spaces, using
        # the _make_indent method
        #
        self.indent_level = 0
        self.inside_ArgList = False
        self.inside_ArgList2 = list()
        self.arg_list_level = 0
        self.inside_Assignment = False

    def write_ast_to_file(self, ast, filename='temp.cpp'):
        code = self.visit(ast)
        currentdir = os.getcwd()
        full_file_name = currentdir + '/' + filename
        try:
            os.remove(full_file_name)
        except OSError:
            pass
        try:
            fileobj = open(full_file_name, 'w')
            fileobj.write(code)
            fileobj.close()
        except IOError:
            print "Unable to write file"

    def simple_node(self, n):
        """ Returns True for nodes that are "simple"
        """
        return not isinstance(n, (lan.Constant, lan.Id, lan.ArrayRef, lan.FuncDecl, lan.FuncCall))

    def parenthesize_if(self, n, condition):
        """ Visits 'n' and returns its string representation, parenthesized
            if the condition function applied to the node returns True.
            :param n:
            :param condition:
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
            newline = n.__class__.__name__ + newline
            start = n.__class__.__name__ + start
        s = ''

        for ext in n.ext:
            if isinstance(ext, lan.Compound):
                s += self.visit_GlobalCompound(ext)
            else:
                s += start + self.visit(ext) + newline
        return s

    def visit_GlobalCompound(self, n):
        s = ''
        for stat in n.statements:
            s += self.visit(stat)
        s += n.__class__.__name__ + self.newline
        return s

    def visit_GroupCompound(self, n):
        newline = self.newline
        start = self.start
        if debug:
            newline = n.__class__.__name__ + newline
            start = n.__class__.__name__ + start
        s = ''
        for i, stat in enumerate(n.statements):
            start1 = ''
            if i != 0:
                start1 = start
            s += start1 + self.visit(stat) + newline + self._make_indent()
        s += start
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
        s = self.visit(n.name)
        if n.type:
            s1 = ' '.join(n.type)
            s1 += ' ' + s
        else:
            s1 = s
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
            newline = n.__class__.__name__ + newline
            start = n.__class__.__name__ + start

        s = start + self._make_indent() + '{' + newline
        self.indent_level += 2
        for stat in n.statements:
            s += start + self._make_indent() + self.visit(stat) + newline
        self.indent_level -= 2
        s += start + self._make_indent() + '}'
        return s

    def visit_ArgList(self, n):
        newline = self.newline
        start = self.start
        if debug:
            newline = n.__class__.__name__ + newline
            start = n.__class__.__name__ + start

        s = '('
        count = 1
        if len(n.arglist) == 1:
            return '(' + self.visit(n.arglist[0]) + ')'

        for arg in n.arglist:
            if count == 1:
                s += newline + '\t' + start
            s += self.visit(arg)
            if count != (len(n.arglist)):
                s += ', '
            if count % 3 == 0:
                s += newline + '\t' + start
            count += 1
        return s + ')'

    def visit_ArrayRef(self, n):
        s = self.visit(n.name)
        for arg in n.subscript:
            s += '[' + self.visit(arg) + ']'
        return s

    def visit_BinOp(self, n):
        lval = self.parenthesize_if(n.lval, self.simple_node)
        rval = self.parenthesize_if(n.rval, self.simple_node)
        return lval + ' ' + n.op + ' ' + rval

    def visit_FuncDecl(self, n):
        newline = self.newline
        if debug:
            newline = n.__class__.__name__ + newline

        my_inside_arg_list = self.inside_ArgList
        self.inside_ArgList = True

        typeid = self.visit(n.typeid)

        arglist = self.visit(n.arglist)

        self.inside_ArgList = False or my_inside_arg_list
        if self.inside_Assignment or self.inside_ArgList:
            compound = ''
        elif n.compound.statements:
            typeid = self.start + typeid
            arglist += newline
            compound = self.visit(n.compound) + newline
        else:
            compound = self.semi
        return typeid + arglist + compound

    def visit_FuncCall(self, n):

        my_inside_arg_list = self.inside_ArgList
        self.inside_ArgList = True

        id = self.visit(n.id)
        arglist = self.visit(n.arglist)

        self.inside_ArgList = False or my_inside_arg_list
        if self.inside_Assignment or self.inside_ArgList:
            end = ''
        else:
            end = self.semi
        return id + arglist + end



    def visit_ForLoop(self, n):
        newline = self.newline
        if debug:
            newline = n.__class__.__name__ + newline

        init = self.visit(n.init)  # already has a semi at the end
        cond = self.visit(n.cond)
        inc = self.visit(n.inc)
        self.indent_level += 2
        compound = self.visit(n.compound)
        self.indent_level -= 2
        return 'for (' + init + ' ' + cond + self.semi + ' ' + inc + ')' \
               + newline + compound

    def visit_IfThen(self, n):
        newline = self.newline
        start = self.start
        if debug:
            newline = n.__class__.__name__ + newline
            start = n.__class__.__name__ + start

        cond = self.visit(n.cond)
        self.indent_level += 2
        compound = self.visit(n.compound)
        self.indent_level -= 2
        return 'if (' + cond + ')' + newline + compound

    def visit_IfThenElse(self, n):
        newline = self.newline
        start = self.start
        if debug:
            newline = n.__class__.__name__ + newline
            start = n.__class__.__name__ + start

        cond = self.visit(n.cond)
        self.indent_level += 2
        compound1 = self.visit(n.compound1)
        compound2 = self.visit(n.compound2)
        self.indent_level -= 2
        return 'if (' + cond + ')' + newline + compound1 \
               + newline + self._make_indent() + 'else' + newline + compound2

    def visit_Id(self, n):
        return n.name

    def visit_Include(self, n):
        return "#include " + n.name

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

    def visit_Return(self, n):
        expr = self.visit(n.expr)
        return 'return ' + expr

    def visit_RawCpp(self, n):
        return n.code

    def visit_Type(self, n):
        return n.type

    def visit_Ref(self, n):
        expr = self.visit(n.expr)
        return '&' + expr

    def visit_Cout(self, n):
        s = ''
        for arg in n.print_args:
            s += ' << ' + self.visit(arg)
        return 'cout' + s + ' << endl;'

    def visit_RunOCLArg(self, n):
        s = self.visit(n.ocl_arg)
        return s

    def visit_CppClass(self, n):
        s = ''
        name = self.visit(n.name)
        s += 'class ' + name + '\n {\n'
        for var in n.var_list:
            s += self.visit(var)
