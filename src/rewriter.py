import os
from lan_ast import *


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
        print "36 " , arrays.numIndices
        print "37 " , arrays.ids
        print "38 " , arrays.indexIds

        typeIds = TypeIds()
        typeIds.visit(ast)
        print typeIds.ids

        ids = Ids()
        ids.visit(ast)
        print ids.ids
        otherIds = ids.ids - arrays.ids - typeIds.ids
        print otherIds
        typeid = TypeId(['void'], Id(functionname),ast.coord)
        arraysArg = list()
        for arrayid in arrays.ids:
            arraysArg.append(TypeId(['unknown','*'], Id(arrayid,ast.coord),ast.coord))
            for iarg in xrange(arrays.numIndices[arrayid]):
                arraysArg.append(TypeId(['size_t'], Id('hst_ptr'+arrayid+'_dim'+str(iarg+1),ast.coord),ast.coord))
                
        for arrayid in otherIds:
             arraysArg.append(TypeId(['unknown'], Id(arrayid,ast.coord),ast.coord))
            
        arglist = ArgList([] + arraysArg,ast.coord)
        compound = Compound(ast.ext,ast.coord)
        ## ast.ext.insert(0,FuncDecl(typeid,arglist,compound,ast.coord))
        ast.ext = list()
        ast.ext.append(FuncDecl(typeid,arglist,compound,ast.coord))


        
    def rewriteToSequentialC(self, ast): 
        loops = ForLoops()
        loops.visit(ast)
        forLoopAst = loops.ast
        loopIndices = LoopIndices()
        loopIndices.visit(forLoopAst)
        self.index = loopIndices.index

        arrays2 = Arrays(self.index)
        arrays2.visit(ast)
        print "36 " , arrays2.numIndices
        print "37 " , arrays2.ids
        print "38 " , arrays2.indexIds
        findDim = FindDim(arrays2.numIndices)
        findDim.visit(ast)
        print "75 " , findDim.dimNames
        rewriteArrayRef = RewriteArrayRef(findDim.dimNames)
        rewriteArrayRef.visit(ast)

    def rewriteToDeviceCTemp(self, ast):

        perfectForLoop = PerfectForLoop()
        perfectForLoop.visit(ast)
        ## print perfectForLoop.depth
        ## print perfectForLoop.ast
        initIds = InitIds()
        initIds.visit(perfectForLoop.ast.init)
        gridIds = list()
        idMap = dict()
        idMap[initIds.index[0]] = 'get_global_id(0)'
        gridIds.extend(initIds.index)
        kernel = perfectForLoop.ast.compound
        if perfectForLoop.depth == 2:
            initIds = InitIds()
            initIds.visit(kernel.statements[0].init)
            kernel = kernel.statements[0].compound
            idMap[initIds.index[0]] = 'get_global_id(1)'
            gridIds.extend(initIds.index)
            (idMap[gridIds[0]], idMap[gridIds[1]]) = (idMap[gridIds[1]], idMap[gridIds[0]])
        ## print "idmap " , idMap

        ## print kernel
        arrays = Arrays([])
        arrays.visit(kernel)
        typeIds = TypeIds()
        typeIds.visit(kernel)
        ## print arrays.ids
        ## print typeIds.ids
        ids = Ids()
        ids.visit(kernel)
        ## print ids.ids
        otherIds = ids.ids  - typeIds.ids - set(gridIds)
        ## print otherIds
        findDeviceArgs = FindDeviceArgs(otherIds)
        findDeviceArgs.visit(ast)
        ## print findDeviceArgs.arglist
        findFunction = FindFunction()
        findFunction.visit(ast)
        ## print findFunction.typeid

        # add OpenCL keywords to indicate the kernel function.
        findFunction.typeid.type.insert(0, '__kernel')
        
        exchangeIndices = ExchangeIndices(idMap)
        exchangeIndices.visit(kernel)
        newast =  FuncDecl(findFunction.typeid, ArgList(findDeviceArgs.arglist,ast.coord), kernel, ast.coord)
        ast.ext = list()
        ast.ext.append(newast)

    def rewriteToDeviceCRelease(self, ast):

        perfectForLoop = PerfectForLoop()
        perfectForLoop.visit(ast)
        ## print perfectForLoop.depth
        ## print perfectForLoop.ast
        initIds = InitIds()
        initIds.visit(perfectForLoop.ast.init)
        gridIds = list()
        idMap = dict()
        idMap[initIds.index[0]] = 'get_global_id(0)'
        gridIds.extend(initIds.index)
        kernel = perfectForLoop.ast.compound
        if perfectForLoop.depth == 2:
            initIds = InitIds()
            initIds.visit(kernel.statements[0].init)
            kernel = kernel.statements[0].compound
            idMap[initIds.index[0]] = 'get_global_id(1)'
            gridIds.extend(initIds.index)
            (idMap[gridIds[0]], idMap[gridIds[1]]) = (idMap[gridIds[1]], idMap[gridIds[0]])
        print "idmap " , idMap

        self.rewriteToSequentialC(ast)

        ## print kernel
        arrays = Arrays([])
        arrays.visit(kernel)
        typeIds = TypeIds()
        typeIds.visit(kernel)
        ## print arrays.ids
        ## print typeIds.ids
        ids = Ids()
        ids.visit(kernel)
        ## print ids.ids
        otherIds = ids.ids  - typeIds.ids - set(gridIds)
        print otherIds
        findDeviceArgs = FindDeviceArgs(otherIds)
        findDeviceArgs.visit(ast)
        print findDeviceArgs.arglist
        findFunction = FindFunction()
        findFunction.visit(ast)
        print findFunction.typeid

        # add OpenCL keywords to indicate the kernel function.
        findFunction.typeid.type.insert(0, '__kernel')
        
        exchangeIndices = ExchangeIndices(idMap)
        exchangeIndices.visit(kernel)

        newast =  FuncDecl(findFunction.typeid, ArgList(findDeviceArgs.arglist,ast.coord), kernel, ast.coord)
        ast.ext = list()
        ast.ext.append(newast)


    def generateBoilerplateCode(self, ast):

        loops = ForLoops()
        loops.visit(ast)
        forLoopAst = loops.ast
        loopIndices = LoopIndices()
        loopIndices.visit(forLoopAst)
        self.index = loopIndices.index

        arrays2 = Arrays(self.index)
        arrays2.visit(ast)
        print "36 " , arrays2.numIndices
        print "36s " , arrays2.numSubscripts
        print "37 " , arrays2.ids
        print "38 " , arrays2.indexIds
        findDim = FindDim(arrays2.numIndices)
        findDim.visit(ast)
        print "75 " , findDim.dimNames


        perfectForLoop = PerfectForLoop()
        perfectForLoop.visit(ast)
        ## print perfectForLoop.depth
        ## print perfectForLoop.ast
        initIds = InitIds()
        initIds.visit(perfectForLoop.ast.init)
        gridIds = list()
        idMap = dict()
        idMap[initIds.index[0]] = 'get_global_id(0)'
        gridIds.extend(initIds.index)
        kernel = perfectForLoop.ast.compound
        if perfectForLoop.depth == 2:
            initIds = InitIds()
            initIds.visit(kernel.statements[0].init)
            kernel = kernel.statements[0].compound
            idMap[initIds.index[0]] = 'get_global_id(1)'
            gridIds.extend(initIds.index)
            (idMap[gridIds[0]], idMap[gridIds[1]]) = (idMap[gridIds[1]], idMap[gridIds[0]])
        print "GENidmap " , idMap

        ## print kernel
        arrays = Arrays([])
        arrays.visit(kernel)
        typeIds = TypeIds()
        typeIds.visit(kernel)
        ## print arrays.ids
        ## print typeIds.ids
        ids = Ids()
        ids.visit(kernel)
        ## print ids.ids
        otherIds = ids.ids  - typeIds.ids - set(gridIds)
        print otherIds
        findDeviceArgs = FindDeviceArgs(otherIds)
        findDeviceArgs.visit(ast)
        print "findDeviceArgs " , findDeviceArgs.arglist
        findFunction = FindFunction()
        findFunction.visit(ast)
        print findFunction.typeid
        kernelId = findFunction.typeid.name
        kernelId.name = kernelId.name + 'Kernel'
        kernelTypeid = TypeId(['cl_kernel'], kernelId, 0)
        print kernelTypeid

        fileAST = FileAST([])
        fileAST.ext.append(kernelTypeid)
        fileAST.show()

        listDevBuffers = []
        for n in arrays2.ids:
            listDevBuffers.append(TypeId(['cl_mem'], Id('dev_ptr' + n), 0))

        listDevBuffers = GroupCompound(listDevBuffers)
        ## print listDevBuffers

        fileAST.ext.append(listDevBuffers)

        listHostPtrs = []
        for n in findDeviceArgs.arglist:
            name = n.name.name
            type = n.type[-2:]
            prefix = 'hst_ptr' if len(type) == 2 else ''                
            listHostPtrs.append(TypeId(type, Id(prefix + name), 0))

        listHostPtrs = GroupCompound(listHostPtrs)
        fileAST.ext.append(listHostPtrs)

        listMemSize = []
        listDimSize = []
        listMemSizeCalcTemp = []
        listMemSizeCalc = dict()
        for n in arrays2.numSubscripts:
            prefix = 'hst_ptr'
            suffix = '_mem_size'
            listMemSize.append(TypeId(['size_t'], Id(prefix + n + suffix)))
            for i in xrange(arrays2.numSubscripts[n]):
                suffix = '_dim' + str(i+1)
                dimName = prefix + n + suffix
                listDimSize.append(\
                TypeId(['size_t'], Id(dimName)))
                listMemSizeCalcTemp.append(dimName)

            listMemSizeCalc[n] = listMemSizeCalcTemp
            listMemSizeCalcTemp = []
                
        fileAST.ext.append(GroupCompound(listMemSize))
        fileAST.ext.append(GroupCompound(listDimSize))
        fileAST.ext.append(TypeId(['size_t'], Id('isFirstTime')))

        allocateBuffer = EmptyFuncDecl('AllocateBuffers')
        fileAST.ext.append(allocateBuffer)

        listSetMemSize = []


        
        fileAST.show()



class ExchangeIndices(NodeVisitor):
    """ Exchanges the indices that we parallelize with the threadids """
    def __init__(self, idMap):
        self.idMap = idMap

    def visit_ArrayRef(self, node):
        for n in node.subscript:
            self.visit(n)
        
    def visit_Id(self, node):
        if node.name in self.idMap:
            ## print self.idMap[node.name]
            node.name = self.idMap[node.name]
        


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

    def visit_FuncDecl(self, node):
        funcstats = node.compound.statements
        if len(funcstats) == 1: # 
            if isinstance(funcstats[0], ForLoop):
                self.ast = funcstats[0]
                self.depth += 1
                loopstats = funcstats[0].compound.statements
                if len(loopstats) == 1:
                    if isinstance(loopstats[0], ForLoop):
                        self.depth += 1


        ## stats = node.compound.statements
        ## if len(stats) == 1:
        ##     if isinstance(stats[0], ForLoop):
                


class RewriteArrayRef(NodeVisitor):
    """ Rewrites the arrays references of form A[i][j] to
    A[i * JDIMSIZE + j]
    """
    def __init__(self, arrayDims):
        self.arrayDims = arrayDims
    
    def visit_ArrayRef(self, node):
        if len(self.arrayDims[node.name.name]) == 2:
            leftbinop = BinOp(node.subscript[0],'*', \
            # Id on first dimension
            self.arrayDims[node.name.name][0], node.coord)
            topbinop = BinOp(leftbinop,'+', \
            node.subscript[1], node.coord)
            ## print topbinop
            node.subscript = [topbinop]

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
                        node.arglist[count + 1 + n].name)
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
    """ Finds Id's in an for loop initialization.
    More generally: Finds all Ids and adds them to a list.    
    """
    def __init__(self):
        self.index = list()
    
    def visit_Id(self, node):
        self.index.append(node.name)

class Ids(NodeVisitor):
    """ Finds all unique Ids """
    def __init__(self):
        self.ids = set()
    def visit_Id(self, node):
        self.ids.add(node.name)

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

    
class Arrays(NodeVisitor):
    """ Finds array Ids """
    def __init__(self, loopindices):
        self.ids = set()
        self.numIndices = dict()
        self.indexIds = dict()
        self.loopindices = loopindices
        self.numSubscripts = dict()
    def visit_ArrayRef(self, node):
        name = node.name.name
        self.ids.add(name)
        numIndcs = NumIndices(99, self.loopindices)
        self.numSubscripts[name] = len(node.subscript)
        for s in node.subscript:
            numIndcs.visit(s)
        if name not in self.numIndices:
            self.numIndices[name] = numIndcs.num
            self.indexIds[name] = numIndcs.found

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