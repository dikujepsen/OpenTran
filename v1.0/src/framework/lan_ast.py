import sys
class Node(object): 

    def children(self):
        """ A sequence of all children that are Nodes
        """
        pass
    
    def show(self, buf=sys.stdout, offset=0, _my_node_name=None):
        """ Pretty print the Node and all its attributes and
            children (recursively) to a buffer.
            
            buf:   
                Open IO buffer into which the Node is printed.
            
            offset: 
                Initial offset (amount of leading spaces)
        """

        lead = ' ' * offset
        if _my_node_name is not None:
            buf.write(lead + self.__class__.__name__+ ' <' + _my_node_name + '>: ')
        else:
            buf.write(lead + self.__class__.__name__+ ' <top>: ')

#        if self.__class__ == Number:
#            print ": " + self.value
        nvlist = [(n, getattr(self,n)) for n in self.attr_names]
        attrstr = ', '.join('%s=%s' % nv for nv in nvlist)
        buf.write(attrstr)
        
        buf.write('\n')
        for (child_name, child) in self.children():
            ## print child
            child.show(
                buf,
                offset=offset + 2,
                _my_node_name=child_name)

class NodeVisitor(object):
    current_parent = None
    def visit(self, node):
        """ Visit a node. 
        """
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        #print self.current_parent
        return visitor(node)
        
    def generic_visit(self, node):
        """ Called if no explicit visitor function exists for a 
            node. Implements preorder visiting of the node.
        """
        oldparent = self.current_parent
        self.current_parent = node
        for c_name, c in node.children():
            self.visit(c)
        self.current_parent = oldparent


class FileAST(Node):
    def __init__(self, ext, coord=None):
        self.ext = ext
        self.coord = coord

    def __repr__(self):
        return "FileAST(%r)" % ( self.ext)

    def children(self):
        nodelist = []
        for i, child in enumerate(self.ext or []):
            nodelist.append(("ext[%d]" % i, child))
        return tuple(nodelist)

    attr_names = ()


class Comment(Node):
    def __init__(self,value,coord = None):
        self.value = value
        self.coord = coord
    def __repr__(self):
        return "Comment(%r)" % ( self.value)
    def children(self):
        nodelist = []
        return tuple(nodelist)
    attr_names = ('value',)

class ArrayInit(Node):
    def __init__(self, values, coord = None):
        self.values = values
        self.coord = coord
    def __repr__(self):
        return "ArrayInit(%r)" % ( self.values)
    def children(self):
        nodelist = []
        for n in self.values:
            nodelist.append(n)
        return tuple(nodelist)
    attr_names = ()


class Constant(Node):
    def __init__(self,value,coord = None):
        self.value = value
        self.coord = coord
    def __repr__(self):
        return "Constant(%r)" % ( self.value)
    def children(self):
        nodelist = []
        return tuple(nodelist)
    attr_names = ('value',)

class Increment(Node):
    def __init__(self, name, op, coord = None):
        self.name = name
        self.op = op
        self.coord = coord
    def __repr__(self):
        return "Increment(%r%r)" % ( self.name, self.op)
    def children(self):
        nodelist = []
        nodelist.append(("name", self.name))
        return tuple(nodelist)
    attr_names = ("op",)
    

class UnaryBefore(Node):
    def __init__(self, op, expr,coord = None):
        self.op = op
        self.expr = expr
        self.coord = coord
    def __repr__(self):
        return "UnaryBefore(%r %r)" % ( self.op , self.expr)
    def children(self):
        nodelist = []
        nodelist.append(("expr", self.expr))
        return tuple(nodelist)
    attr_names = ("op",)


class Id(Node):
    def __init__(self,name,coord = None):
        self.name = name
        self.coord = coord
    def __repr__(self):
        return "Id(%r)" % ( self.name)
    def children(self):
        nodelist = []
        return tuple(nodelist)
    attr_names = ('name',)

class Include(Node):
    def __init__(self, name, coord = None):
        self.name = name
        self.coord = coord
    def __repr__(self):
        return "Include(%r)" % ( self.name)
    def children(self):
        nodelist = []
        return tuple(nodelist)
    attr_names = ('name',)

class TypeId(Node):
    def __init__(self,type,name,coord = None):
        self.type = type
        self.name = name
        self.coord = coord
    def __repr__(self):
        return "TypeId(%r %r)" % ( self.type, self.name)
    def children(self):
        nodelist = [("name", self.name)]
        return tuple(nodelist)
    attr_names = ('type',)

class ArrayTypeId(Node):
    def __init__(self,type, name, subscript, coord = None):
        self.type = type
        self.name = name
        self.subscript = subscript
        self.coord = coord
    def __repr__(self):
        return "ArrayTypeId(%r %r % r)" % ( self.type, self.name, self.subscript)
    def children(self):
        nodelist = [("name", self.name)]
        for count, i in enumerate(self.subscript):
            nodelist.append(("subscript %r" % count, i))

        return tuple(nodelist)
    attr_names = ('type',)

    
class Assignment(Node):
    def __init__(self, lval, rval, op = '=', coord = None):
        self.lval = lval
        self.op = op
        self.rval = rval
        self.coord = coord
    def __repr__(self):
        return "Assignment(%r %r %r)" % ( self.lval, self.op, self.rval)
    def children(self):
        nodelist = []
        nodelist.append(("lval", self.lval))
        nodelist.append(("rval", self.rval))
        return tuple(nodelist)
    attr_names = ("op",)

# Special class for grouping statements (no newlines in between)
class GroupCompound(Node):
    def __init__(self,statements,coord = None):
        self.statements = statements
        self.coord = coord
    def __repr__(self):
        return "GroupCompound({%r})" % ( self.statements )
    def children(self):
        nodelist = []
        count = 0
        for i in self.statements:
            nodelist.append(("stmt[%r]" % count, i))
            count += 1
        return tuple(nodelist)
    attr_names = ()


class Compound(Node):
    def __init__(self,statements,coord = None):
        self.statements = statements
        self.coord = coord
    def __repr__(self):
        return "Compound({%r})" % ( self.statements )
    def children(self):
        nodelist = []
        count = 0
        for i in self.statements:
            nodelist.append(("stmt[%r]" % count, i))
            count += 1
        return tuple(nodelist)
    attr_names = ()

class ArgList(Node):
    def __init__(self, arglist,coord  = None):
        self.arglist = arglist
        self.coord = coord
    def __repr__(self):
        return "ArgList(%r)" % ( self.arglist )
    def children(self):
        nodelist = []
        count = 0
        for i in self.arglist:
            nodelist.append(("arglist[%r]" % count, i))
            count += 1
        return tuple(nodelist)
    attr_names = ()

class ArrayRef(Node):
    def __init__(self, name, subscript, coord  = None, extra = dict()):
        self.name = name
        self.subscript = subscript
        self.coord = coord
        self.extra = extra
        
    def __repr__(self):
        return "ArrayRef(%r%r)" % ( self.name , self.subscript )
    def children(self):
        nodelist = []
        nodelist.append(("name", self.name))
        count = 0
        for i in self.subscript:
            nodelist.append(("subscript %r" % count, i))
            count += 1
        return tuple(nodelist)
    attr_names = ()


class BinOp(Node):
    def __init__(self,lval,op,rval,coord = None):
        self.lval = lval
        self.rval = rval
        self.op = op
        self.coord = coord
    def __repr__(self):
        return "BinOp(%r %r %r)" % ( self.lval, self.op, self.rval)
    def children(self):
        nodelist = []
        nodelist.append(("lval", self.lval))
        nodelist.append(("rval", self.rval))
        return tuple(nodelist)
    attr_names = ("op",)

class FuncDecl(Node):
    def __init__(self,typeid,arglist,compound,coord = None):
        self.typeid = typeid
        self.arglist = arglist
        self.compound = compound
        self.coord = coord
    def __repr__(self):
        return "FuncDecl(%r %r %r)" % ( self.typeid, \
                                        self.arglist, \
                                        self.compound )
    def children(self):
        nodelist = []
        nodelist.append(("typeid", self.typeid))
        nodelist.append(("arglist", self.arglist))
        nodelist.append(("compound", self.compound))
        return tuple(nodelist)
    attr_names = ()

class ForLoop(Node):
    def __init__(self,init,cond,inc,compound,coord = None):
        self.init = init
        self.cond = cond
        self.inc = inc
        self.compound = compound
    def __repr__(self):
        return "\nForLoop(%r, %r, %r, %r) " % ( self.init.lval.name, \
                                            self.cond, \
                                            self.inc, \
                                            self.compound
                                        )
    def children(self):
        nodelist = []
        nodelist.append(("init", self.init))
        nodelist.append(("cond", self.cond))
        nodelist.append(("inc", self.inc))
        nodelist.append(("compound", self.compound))
        return tuple(nodelist)
    attr_names = ()

class IfThen(Node):
    def __init__(self, cond, compound, coord = None):
        self.cond = cond
        self.compound = compound
    def __repr__(self):
        return "If(%r) then {%r}" % ( self.cond, \
                                      self.compound )
    def children(self):
        nodelist = []
        nodelist.append(("cond", self.cond))
        nodelist.append(("compound", self.compound))
        return tuple(nodelist)
    attr_names = ()

class IfThenElse(Node):
    def __init__(self, cond, compound1, compound2, coord = None):
        self.cond = cond
        self.compound1 = compound1
        self.compound2 = compound2
    def __repr__(self):
        return "If(%r) then {%r} else {%r}" % ( self.cond, \
                                      self.compound1,\
            						  self.compound2)
    def children(self):
        nodelist = []
        nodelist.append(("cond", self.cond))
        nodelist.append(("compoundthen", self.compound1))
        nodelist.append(("compoundelse", self.compound2))
        return tuple(nodelist)
    attr_names = ()

