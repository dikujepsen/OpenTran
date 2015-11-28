from lan.lan_ast import *
import copy


class Ids(NodeVisitor):
    """ Finds all unique IDs, excluding function IDs"""

    def __init__(self):
        self.ids = set()

    def visit_FuncDecl(self, node):
        if node.compound.statements == []:
            self.visit(node.arglist)

    def visit_Id(self, node):
        self.ids.add(node.name)


class Ids2(NodeVisitor):
    """ Finds all unique IDs, excluding function IDs"""

    def __init__(self):
        self.ids = set()

    def visit_FuncDecl(self, node):
        if node.compound.statements != []:
            self.visit(node.compound)

    def visit_Id(self, node):
        self.ids.add(node.name)


class LoopIndices(NodeVisitor):
    """ Finds loop indices, the start and end values of the
        indices and creates a mapping from a loop index to
        the ForLoop AST node that is indexes.
    """

    def __init__(self):
        self.index = list()
        self.end = dict()
        self.start = dict()
        self.Loops = dict()

    def visit_ForLoop(self, node):
        self.Loops[node.init.lval.name.name] = node
        IdVis = Ids()
        IdVis.visit(node.init)
        ids = list(IdVis.ids)
        self.index.extend(ids)
        self.visit(node.compound)
        try:
            self.end[ids[0]] = (node.cond.rval.name)
            self.start[ids[0]] = (node.init.rval.value)
        except AttributeError:
            self.end[ids[0]] = 'Unknown'
            self.start[ids[0]] = 'Unknown'


class ForLoops(NodeVisitor):
    """ Returns first loop it encounters 
    """

    def __init__(self):
        self.isFirst = True
        self.ast = None

    def reset(self):
        self.isFirst = True

    def visit_ForLoop(self, node):
        if self.isFirst:
            self.ast = node
            self.isFirst = False
            return node


class _NumIndices(NodeVisitor):
    """ Finds if there is two distinct loop indices
    	in an 1D array reference
    """

    def __init__(self, numIndices, indices):
        self.numIndices = numIndices
        self.num = 0
        self.indices = indices
        self.found = set()
        self.subIdx = set()
        self.yes = False

    def reset(self):
        self.firstFound = False
        self.subIdx = set()

    def visit_Id(self, node):
        if node.name in self.indices \
                and node.name not in self.found \
                and self.num < self.numIndices:
            self.found.add(node.name)
            self.subIdx.add(node.name)
            self.num += 1
            if self.num >= self.numIndices:
                self.yes = True


class Arrays(NodeVisitor):
    """ Finds array Ids """

    def __init__(self, loopindices):
        self.ids = set()
        self.numIndices = dict()
        self.indexIds = dict()
        self.loopindices = loopindices
        self.numSubscripts = dict()
        self.Subscript = dict()
        self.LoopArrays = dict()
        self.SubIdx = dict()

    def visit_ArrayRef(self, node):
        name = node.name.name
        self.ids.add(name)
        numIndcs = _NumIndices(99, self.loopindices)
        if name in self.Subscript:
            self.Subscript[name].append(node.subscript)
            self.LoopArrays[name].append(node)
        else:
            self.Subscript[name] = [node.subscript]
            self.LoopArrays[name] = [node]

        listidx = []
        for s in node.subscript:
            numIndcs.visit(s)
            if numIndcs.subIdx:
                listidx.extend(list(numIndcs.subIdx))
            else:
                listidx.append(None)
            numIndcs.reset()

        if name in self.SubIdx:
            self.SubIdx[name].append(listidx)
        else:
            self.SubIdx[name] = [listidx]

        if name not in self.numIndices:
            self.numIndices[name] = numIndcs.num
            self.numSubscripts[name] = numIndcs.num
            self.indexIds[name] = (numIndcs.found)
        else:
            self.indexIds[name].update((numIndcs.found))

        ## self.numSubscripts[name] = max(len(node.subscript),self.numIndices[name])
        self.numSubscripts[name] = len(node.subscript)
        for n in node.subscript:
            self.visit(n)


class TypeIds(NodeVisitor):
    """ Finds type Ids """

    def __init__(self):
        self.ids = set()
        self.dictIds = dict()

    def visit_TypeId(self, node):
        name = node.name.name
        self.ids.add(name)
        self.dictIds[name] = node.type

    def visit_ArrayTypeId(self, node):
        name = node.name.name
        self.ids.add(name)
        self.dictIds[name] = copy.deepcopy(node.type)
        if len(node.type) != 2:
            # "ArrayTypeId: Need to check, type of array is ", node.type
            ## self.dictIds[name].append('*')
            pass


class _NumBinOps(NodeVisitor):
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
            num_binops = _NumBinOps()
            binop = node.subscript[0]
            num_binops.visit(binop)
            if len(num_binops.ops) == 2:
                if '+' in num_binops.ops and '*' in num_binops.ops:

                    if not isinstance(binop.lval, BinOp):
                        (binop.lval, binop.rval) = (binop.rval, binop.lval)
                    two_indices = _NumIndices(2, self.indices)
                    two_indices.visit(binop)
                    if two_indices.yes:
                        if binop.lval.lval.name not in self.indices:
                            (binop.lval.lval.name, binop.lval.rval.name) = \
                                (binop.lval.rval.name, binop.lval.lval.name)
                        # convert to 2D
                        node.subscript = [Id(binop.lval.lval.name, node.coord), binop.rval]
