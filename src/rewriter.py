import os
from lan_ast import *

globalindices = list()

class Rewriter(NodeVisitor):
    """ Rewrites a few things in the AST to increase the
    	abstraction level.
    """
    def __init__(self):
        self.index = list()

    def rewrite(self, ast, functionname = 'FunctionName'):
        loops = ForLoops()
        loops.visit(ast)
        ## loops.reset()
        ## loops.visit(loops.ast.compound)
        forLoopAst = loops.ast
        loopIndices = LoopIndices()
        loopIndices.visit(forLoopAst)
        self.index = loopIndices.index
        globalindices = loopIndices.index
        loopIndices.end.reverse()
        print loopIndices.end
        ## subs = Subscripts()
        ## subs.visit(forLoopAst)
        ## tmp = subs.subscript[0]
        ## subs.subscript[0] = subs.subscript[1]
        ## subs.subscript[1] = tmp
        ## for i in subs.subscript.values():
        ##     i.show()
        norm = Norm(self.index)
        norm.visit(forLoopAst)
        arrays = Arrays(self.index)
        arrays.visit(ast)
        print arrays.indices

        typeIds = TypeIds()
        typeIds.visit(ast)
        print typeIds.ids

        ids = Ids()
        ids.visit(ast)
        print ids.ids
        otherIds = ids.ids - arrays.ids - typeIds.ids
        print otherIds
        typeid = TypeId(['int'], Id(functionname),ast.coord)
        arraysArg = list()
        for arrayid in arrays.ids:
            arraysArg.append(TypeId(['unknown','*'], Id(arrayid,ast.coord),ast.coord))
            for iarg in xrange(arrays.indices[arrayid]):
                arraysArg.append(TypeId(['size_t'], Id('hst_ptr'+arrayid+'_dim'+str(iarg+1),ast.coord),ast.coord))
                
        for arrayid in otherIds:
             arraysArg.append(TypeId(['unknown'], Id(arrayid,ast.coord),ast.coord))
            
        arglist = ArgList([] + arraysArg,ast.coord)
        compound = Compound(ast.ext,ast.coord)
        ## ast.ext.insert(0,FuncDecl(typeid,arglist,compound,ast.coord))
        ast.ext = list()
        ast.ext.append(FuncDecl(typeid,arglist,compound,ast.coord))
        


class InitIds(NodeVisitor):
    """ Finds Id's
    """
    def __init__(self):
        self.index = list()
    
    def visit_Id(self, node):
        self.index.append(node.name)

class Ids(NodeVisitor):
    """ Finds all Ids """
    def __init__(self):
        self.ids = set()
    def visit_Id(self, node):
        self.ids.add(node.name)


class Ends(NodeVisitor):
    """ Finds Id's
    """
    def __init__(self):
        self.index = list()
    
    def visit_Id(self, node):
        self.index.append(node.name)


class LoopIndices(NodeVisitor):
    """ Finds loop indices
    """
    def __init__(self):
        self.index = list()
        self.end = list()
    def visit_ForLoop(self, node):
        IdVis = InitIds()
        IdVis.visit(node.init)
        self.index.extend(IdVis.index)
        self.visit(node.compound)
        try:
            self.end.append(node.cond.rval.name)
        except AttributeError:
            self.end.append('Unknown')
            

class ForLoops(NodeVisitor):
    """ Returns first loop it encounters 
    """
    def __init__(self):
        self.isFirst = True

    def reset(self):
        self.isFirst = True
        
    def visit_ForLoop(self, node):
        if self.isFirst:
            self.ast = node
            self.isFirst = False
            return node

class NumIndices(NodeVisitor):
    """ Finds if there is two distinct loop indices
    	in an 1D array reference
    """
    def __init__(self, numIndices, indices):
        self.numIndices = numIndices
        self.num = 0
        self.indices = indices
        self.found = set()
        self.yes = False
    def visit_Id(self, node):
        if node.name in self.indices \
        and node.name not in self.found \
        and self.num < self.numIndices:
            self.found.add(node.name)
            self.num += 1
            if self.num >= self.numIndices:
                self.yes = True
                
    def reset(self):
        self.firstFound = False

    
class Subscripts(NodeVisitor):
    """ Finds loop indices
    """
    def __init__(self):
        self.subscript = dict()
        self.count = 0
    def visit_ArrayRef(self, node):
        if len(node.subscript) == 1:
            self.subscript[self.count] = node
            self.count += 1

class Arrays(NodeVisitor):
    """ Finds array Ids """
    def __init__(self, loopindices):
        self.ids = set()
        self.indices = dict()
        self.loopindices = loopindices
    def visit_ArrayRef(self, node):
        name = node.name.name
        self.ids.add(name)
        numIndices = NumIndices(99, self.loopindices)
        for s in node.subscript:
            numIndices.visit(s)
        if name not in self.indices:
            self.indices[name] = numIndices.num

class TypeIds(NodeVisitor):
    """ Finds type Ids """
    def __init__(self):
        self.ids = set()
    def visit_TypeId(self, node):
        self.ids.add(node.name.name)


class NumBinOps(NodeVisitor):
    """ Finds the number of BinOp in an 1D array subscript
    """
    def __init__(self):
        self.ops = list()
    def visit_BinOp(self, node):
        self.ops.append(node.op)
        self.visit(node.lval)
        self.visit(node.rval)


class Norm(NodeVisitor):
    """ Normalizes subscripts to the form i * (width of j) + j
    """
    def __init__(self, indices):
        self.subscript = dict()
        self.count = 0
        self.indices = indices
    def visit_ArrayRef(self, node):
        if len(node.subscript) == 1:
            numBinOps = NumBinOps()
            binop = node.subscript[0]
            numBinOps.visit(binop)
            if len(numBinOps.ops) == 2:
                if '+' in numBinOps.ops and '*' in numBinOps.ops:
                    if not isinstance(binop.lval, BinOp):
                        (binop.lval, binop.rval) = (binop.rval, binop.lval)
                    twoIndices = NumIndices(2, self.indices)
                    ## twoIndices.visit(binop.lval)
                    ## twoIndices.reset()
                    ## twoIndices.visit(binop.rval)
                    twoIndices.visit(binop)
                    if twoIndices.yes:
                        if binop.lval.lval.name not in self.indices:
                            (binop.lval.lval.name, binop.lval.rval.name) = \
                            (binop.lval.rval.name, binop.lval.lval.name)
                        # convert to 2D
                        node.subscript = [Id(binop.lval.lval.name,node.coord),\
                                          binop.rval]
                
