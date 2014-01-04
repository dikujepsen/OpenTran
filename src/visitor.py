from lan_ast import *

class AddToId(NodeVisitor):
    """ Finds the Id and replaces it with a binop that
    adds another variable to the id.
    """
    def __init__(self, id, variable):
        self.id = id
        self.variable = variable


    def changeIdNode(self, node):
        if isinstance(node, Id):
            if node.name == self.id:
                return BinOp(node, '+', Id(self.variable))
            else:
                return node
        else:
            return node
    def visit_BinOp(self, node):
        self.visit(node.lval)
        node.lval = self.changeIdNode(node.lval)
        self.visit(node.rval)
        node.rval = self.changeIdNode(node.rval)

    def visit_Assignment(self, node):
        self.visit(node.lval)
        node.lval = self.changeIdNode(node.lval)
        self.visit(node.rval)
        node.rval = self.changeIdNode(node.rval)

    def visit_ArrayRef(self, node):
        for i, n in enumerate(node.subscript):
            addToId = Ids()
            addToId.visit(n)
            if self.id in addToId.ids:
                sub = node.subscript[i]
                try:
                    if node.extra['localMemory']:
                        return
                except KeyError:
                    pass 
                node.subscript[i] = BinOp(sub, '+', Id(self.variable))


class FindReadWrite(NodeVisitor):
    """ Returns a mapping of array to either
    'read'-only, 'write'-only, or 'readwrite'
    """
    def __init__(self, ArrayIds):
        self.ReadWrite = dict()
        self.ArrayIds = ArrayIds
        self.left = True
        for n in self.ArrayIds:
            self.ReadWrite[n] = set()
        
    def visit_Assignment(self, node):
        self.left = True
        self.visit(node.lval)
        self.left = False
        self.visit(node.rval)

    def visit_Id(self, node):
        name = node.name
        if name in self.ArrayIds:
            if self.left:
                self.ReadWrite[name].add('write')
            else:
                self.ReadWrite[name].add('read')

                

class ExchangeId(NodeVisitor):
    """ Exchanges the Ids that we parallelize with the threadids,
    (or whatever is given in idMap)
    """
    def __init__(self, idMap):
        self.idMap = idMap

    def visit_Id(self, node):
        if node.name in self.idMap:
            node.name = self.idMap[node.name]


class ExchangeIndices(NodeVisitor):
    """ Exchanges the indices that we parallelize with the threadids,
    (or whatever is given in idMap)
    ARGS: idMap: a dictionary of Id changes
    	  arrays: A list/set of array names that we change
    """
    def __init__(self, idMap, arrays):
        self.idMap = idMap
        self.arrays = arrays

    def visit_ArrayRef(self, node):
        if node.name.name in self.arrays:
            exchangeId = ExchangeId(self.idMap)
            for n in node.subscript:
                exchangeId.visit(n)

class ExchangeTypes(NodeVisitor):
    """ Exchanges the size_t to unsigned for every TypeId
    """
    def __init__(self):
        pass
    
    def visit_TypeId(self, node):
        if node.type:
            if node.type[0] == 'size_t':
                node.type[0] = 'unsigned'


class ExchangeArrayId(NodeVisitor):
    """ Exchanges the id of arrays in ArrayRefs with
    what is given in idMap)
    """
    def __init__(self, idMap):
        self.idMap = idMap

    def visit_ArrayRef(self, node):
        try:
            if node.extra['localMemory']:
                return
        except KeyError:
            pass

        try:
            node.name.name = self.idMap[node.name.name]
        except KeyError:
            pass
 

class FindFunction(NodeVisitor):
    """ Finds the typeid of the kernel function """
    def __init__(self):
        self.typeid = None
        
    def visit_FuncDecl(self, node):
        self.visit_TypeId(node.typeid)

    def visit_TypeId(self, node):
        self.typeid = node

        
class FindDeviceArgs(NodeVisitor):
    """ Finds the argument that we transfer from the C code
    to the device. 
    """
    def __init__(self, argIds):
        self.argIds = argIds
        self.arglist = list()
    
    def visit_ArgList(self, node):
        for typeid in node.arglist:
            if isinstance(typeid, TypeId):
                if typeid.name.name in self.argIds:
                    self.argIds.remove(typeid.name.name)
                    if len(typeid.type) == 2:
                        if typeid.type[1] == '*':
                            typeid.type.insert(0,'__global')
                    self.arglist.append(typeid)

class PerfectForLoop(NodeVisitor):
    """ Performs simple checks to decide if we have 1D or 2D
    parallelism, i.e. if we have a perfect loops nest of size one
    or two. 
    """
    def __init__(self):
        self.depth = 0
        self.ast = None
        self.inner = None

    def visit_FuncDecl(self, node):
        funcstats = node.compound.statements
        if len(funcstats) == 1: # 
            if isinstance(funcstats[0], ForLoop):
                self.ast = funcstats[0]
                self.inner = funcstats[0]
                self.depth += 1
                loopstats = funcstats[0].compound.statements
                if len(loopstats) == 1:
                    if isinstance(loopstats[0], ForLoop):
                        self.depth += 1
                        self.inner = loopstats[0]

        ## stats = node.compound.statements
        ## if len(stats) == 1:
        ##     if isinstance(stats[0], ForLoop):
                


class RewriteArrayRef(NodeVisitor):
    """ Rewrites the arrays references of form A[i][j] to
    A[i * JDIMSIZE + j]
    """
    def __init__(self, NumDims, ArrayIdToDimName, data):
        self.data = data
        self.NumDims = NumDims
        self.ArrayIdToDimName = ArrayIdToDimName
    
    def visit_ArrayRef(self, node):
        n = node.name.name
        try:
            if self.data.NumDims[n] == 2 and len(node.subscript) == 2:
                try:
                    if self.data.SubSwap[n]:
                        (node.subscript[0], node.subscript[1]) = \
                        (node.subscript[1], node.subscript[0])
                except KeyError:
                    pass
                leftbinop = BinOp(node.subscript[0],'*', \
                # Id on first dimension

                Id(self.ArrayIdToDimName[n][0]))
                topbinop = BinOp(leftbinop,'+', \
                node.subscript[1])
                node.subscript = [topbinop]
        except KeyError:
            pass

class FindDim(NodeVisitor):
    """ Finds the size of the dimNum dimension.
    """
    def __init__(self, arrayIds):
        self.arrayIds = arrayIds
        self.dimNames = dict()
    
    def visit_ArgList(self, node):
        for arrayname in self.arrayIds:
            findSpecificArrayId = FindSpecificArrayId(arrayname)
            count = 0
            for typeid in node.arglist:            
                findSpecificArrayId.reset(arrayname)
                findSpecificArrayId.visit(typeid)
                if findSpecificArrayId.Found:
                    self.dimNames[arrayname] = list()
                    for n in xrange(self.arrayIds[arrayname]):
                        self.dimNames[arrayname].append(
                        node.arglist[count + 1 + n].name.name)
                count += 1


class FindSpecificArrayId(NodeVisitor):
    """ Finds a specific arrayId
    """
    def __init__(self, arrayId):
        self.arrayId = arrayId
        self.Found = False
    
    def visit_TypeId(self, node):
        if node.name.name == self.arrayId:
            self.Found = True

    def reset(self, arrayId):
        self.Found = False
        self.arrayId = arrayId

class InitIds(NodeVisitor):
    """ Finds Id's in a for loop initialization.
    More generally: Finds all Ids and adds them to a list.    
    """
    def __init__(self):
        self.index = list()
    
    def visit_Id(self, node):
        self.index.append(node.name)

class FindUpperLimit(NodeVisitor):
    """ Finds Id's in an for loop initialization.
    More generally: Finds all Ids and adds them to a list.    
    """
    def __init__(self):
        self.index = list()
    
    def visit_Id(self, node):
        self.index.append(node.name)
        
class Ids(NodeVisitor):
    """ Finds all unique IDs, excluding function IDs"""
    def __init__(self):
        self.ids = set()

    def visit_FuncDecl(self, node):
        pass
        
    def visit_Id(self, node):
        self.ids.add(node.name)

class LoopIds(NodeVisitor):
    """ Finds all unique LoopIndices
        -- Used in localMemory2
    """
    def __init__(self, LoopIds):
        self.LoopIds = LoopIds
        self.ids = set()

    def reset(self):
        self.ids = set()

        
    def visit_Id(self, node):
        name = node.name
        if name in self.LoopIds:
            self.ids.add(name)


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

class NumIndices(NodeVisitor):
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
        numIndcs = NumIndices(99, self.loopindices)
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
            
        self.numSubscripts[name] = max(len(node.subscript),self.numIndices[name])

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
                
def EmptyFuncDecl(name, type = ['void']):
    """ Returns a FuncDecl with no arguments or body """
    allocateBufferTypeId = TypeId(type, Id(name))
    allocateBufferArgList = ArgList([])
    allocateBufferCompound = Compound([])
    allocateBuffer = FuncDecl(allocateBufferTypeId,\
                              allocateBufferArgList,\
                              allocateBufferCompound)


    return allocateBuffer

def ConstantAssignment(name, constant = 0, type = ['unsigned']):
    lval = TypeId(type, Id(name))
    rval = Constant(constant)
    return Assignment(lval,rval)
