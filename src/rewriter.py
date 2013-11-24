import copy
import os
from lan_ast import *


class Rewriter(NodeVisitor):
    """ Class for rewriting of the original AST. Includes:
    1. the initial small rewritings,
    2. transformation into our representation,
    3. transforming from our representation to C-executable code,
    4. creating our representation of the device kernel code,
    5. creating a C-executable kernel code,
    6. Creating the host code (boilerplate code) 
    """

    
    def __init__(self):
        # List of loop indices
        self.index = list()
        # list of the upper limit of the loop indices
        self.UpperLimit = list()
        # The number of dimensions of each array 
        self.NumDims = dict()
        # The Ids of arrays, or pointers
        self.ArrayIds = set()
        # The indices that appear in the subscript of each array
        self.IndexInSubscript = dict()
        # All Ids that are not arrays, or pointers
        self.NonArrayIds = set()
        # Ids that we remove due to parallelization of loops
        self.RemovedIds = set()
        # The mapping from the indices that we parallelize
        # to their function returning their global id
        self.IndexToGlobalId = dict()
        # The indices that we parallelize
        self.GridIndices = list()
        # The OpenCl kernel before anything
        self.Kernel = None
        # The name of the kernel, i.e. the FuncName + Kernel
        self.KernelName = None
        # The mapping from the array ids to a list of 
        # the names of their dimensions
        self.ArrayIdToDimName = dict()
        # The argument list in our ER
        self.DevArgList = list()
        # The name and type of the kernel function.
        self.DevFuncTypeId = None
        # The name of the kernel function.
        self.DevFuncId = None
        # The device names of the pointers in the boilerplate code
        self.DevId = dict()
        # The host names of the pointers in the boilerplate code
        self.HstId = dict()
        # The types of the arguments for the kernel
        self.Type = dict()
        # The name of the variable denoting the memory size 
        self.Mem = dict()
        # Dimension of the parallelization
        self.ParDim = None
        # VarName of the global/local worksize array.
        self.Worksize = dict()
        # The dimension that the index indexes
        self.IdxToDim = dict()
        # Whether an array is read, write or both
        self.ReadWrite = dict()
        # List of arrays that are write only
        self.WriteOnly = list()

        
    def initOriginal(self, ast):
        loops = ForLoops()
        loops.visit(ast)
        forLoopAst = loops.ast
        loopIndices = LoopIndices()
        loopIndices.visit(forLoopAst)
        self.index = loopIndices.index
        self.UpperLimit = loopIndices.end

        norm = Norm(self.index)
        norm.visit(forLoopAst)
        arrays = Arrays(self.index)
        arrays.visit(ast)
        self.NumDims = arrays.numSubscripts
        self.ArrayIds = arrays.ids
        self.IndexInSubscript = arrays.indexIds

        typeIds = TypeIds()
        typeIds.visit(ast)

        ids = Ids()
        ids.visit(ast)
        otherIds = ids.ids - arrays.ids - typeIds.ids
        self.NonArrayIds = otherIds


    def initNewRepr(self, ast):
        perfectForLoop = PerfectForLoop()
        perfectForLoop.visit(ast)
        self.ParDim = perfectForLoop.depth
        initIds = InitIds()
        initIds.visit(perfectForLoop.ast.init)
        gridIds = list()
        idMap = dict()
        firstIdx = initIds.index[0]
        idMap[firstIdx] = 'get_global_id(0)'
        gridIds.extend(initIds.index)
        kernel = perfectForLoop.ast.compound
        if perfectForLoop.depth == 2:
            initIds = InitIds()
            initIds.visit(kernel.statements[0].init)
            kernel = kernel.statements[0].compound
            secondIdx = initIds.index[0]
            idMap[secondIdx] = 'get_global_id(1)'
            gridIds.extend(initIds.index)
            (idMap[gridIds[0]], idMap[gridIds[1]]) = (idMap[gridIds[1]], idMap[gridIds[0]])

        self.IndexToGlobalId = idMap
        self.GridIndices = gridIds
        self.Kernel = kernel
        for i, n in enumerate(reversed(self.GridIndices)):
            self.IdxToDim[i] = n
            
            
        findDim = FindDim(self.NumDims)
        findDim.visit(ast)
        self.ArrayIdToDimName = findDim.dimNames
        self.RemovedIds = set(self.UpperLimit[i] for i in self.GridIndices)

        otherIds = self.ArrayIds.union(self.NonArrayIds)
        findDeviceArgs = FindDeviceArgs(otherIds)
        findDeviceArgs.visit(ast)
        self.DevArgList = findDeviceArgs.arglist
        findFunction = FindFunction()
        findFunction.visit(ast)
        self.DevFuncTypeId = findFunction.typeid
        self.DevFuncId = self.DevFuncTypeId.name.name

        for n in self.ArrayIds:
            self.DevId[n] = 'dev_ptr' + n
            self.HstId[n] = 'hst_ptr' + n
            self.Mem[n] = 'hst_ptr' + n + '_mem_size'
            
        for n in self.DevArgList:
            name = n.name.name
            type = n.type[-2:]
            self.Type[name] = type

        kernelName = self.DevFuncTypeId.name.name
        
        self.KernelName = kernelName + 'Kernel'
        self.Worksize['local'] = kernelName + '_local_worksize'
        self.Worksize['global'] = kernelName + '_global_worksize'
        findReadWrite = FindReadWrite(self.ArrayIds)
        findReadWrite.visit(ast)
        self.ReadWrite = findReadWrite.ReadWrite

        for n in self.ReadWrite:
            pset = self.ReadWrite[n]
            if len(pset) == 1 and 'write' in pset:
                self.WriteOnly.append(n)

            
    def rewrite(self, ast, functionname = 'FunctionName', changeAST = True):
        """ Rewrites a few things in the AST to increase the
    	abstraction level.
        """

        typeid = TypeId(['void'], Id(functionname),ast.coord)
        arraysArg = list()
        for arrayid in self.ArrayIds:
            arraysArg.append(TypeId(['unknown','*'], Id(arrayid,ast.coord),ast.coord))
            for iarg in xrange(self.NumDims[arrayid]):
                arraysArg.append(TypeId(['size_t'], Id('hst_ptr'+arrayid+'_dim'+str(iarg+1),ast.coord),ast.coord))
                
        for arrayid in self.NonArrayIds:
             arraysArg.append(TypeId(['unknown'], Id(arrayid,ast.coord),ast.coord))
            
        arglist = ArgList([] + arraysArg,ast.coord)
        compound = Compound(ast.ext,ast.coord)
        if changeAST:
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

    def rewriteToDeviceCTemp(self, ast, changeAST = True):


        ## print otherIds
        findDeviceArgs = FindDeviceArgs(self.NonArrayIds)
        findDeviceArgs.visit(ast)
        ## print findDeviceArgs.arglist
        findFunction = FindFunction()
        findFunction.visit(ast)
        ## print findFunction.typeid

        # add OpenCL keywords to indicate the kernel function.
        findFunction.typeid.type.insert(0, '__kernel')
        
        exchangeIndices = ExchangeIndices(self.IndexToGlobalId)
        exchangeIndices.visit(self.Kernel)
        newast =  FuncDecl(findFunction.typeid, ArgList(findDeviceArgs.arglist,ast.coord), self.Kernel, ast.coord)
        if changeAST:
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


        dictNToNumScripts = self.NumDims
        dictNToDimNames = self.ArrayIdToDimName

        kernel = self.Kernel
        idMap = self.IndexToGlobalId
        gridIds = self.GridIndices
        NonArrayIds = copy.deepcopy(self.NonArrayIds)
        otherIds = self.ArrayIds.union(self.NonArrayIds) - self.RemovedIds

        kernelId = Id(self.KernelName)
        kernelTypeid = TypeId(['cl_kernel'], kernelId, 0)

        fileAST = FileAST([])
        fileAST.ext.append(kernelTypeid)
        ## fileAST.show()

        listDevBuffers = []

        for n in self.ArrayIds:
            name = 'dev_ptr' + n
            listDevBuffers.append(TypeId(['cl_mem'], Id(name), 0))

        dictNToDevPtr = self.DevId
        listDevBuffers = GroupCompound(listDevBuffers)

        fileAST.ext.append(listDevBuffers)

        listHostPtrs = []
        dictTypeHostPtrs = dict()
        dictNToHstPtr = dict()
        for n in self.DevArgList:
            name = n.name.name
            type = self.Type[name]
            try:
                name = self.HstId[name]
            except KeyError:
                pass
            listHostPtrs.append(TypeId(type, Id(name), 0))

        dictNToHstPtr = self.HstId
        dictTypeHostPtrs = copy.deepcopy(self.Type)
        listHostPtrs = GroupCompound(listHostPtrs)
        fileAST.ext.append(listHostPtrs)

        listMemSize = []
        listDimSize = []
        listMemSizeCalcTemp = []
        dictMemSizeCalc = dict()
        dictNToSize = self.Mem
        for n in self.NumDims:
            sizeName = self.Mem[n]
            listMemSize.append(TypeId(['size_t'], Id(sizeName)))
            for dimName in self.ArrayIdToDimName[n]:
                listDimSize.append(\
                TypeId(['size_t'], Id(dimName)))
                
        fileAST.ext.append(GroupCompound(listMemSize))
        fileAST.ext.append(GroupCompound(listDimSize))
        lval = TypeId(['size_t'], Id('isFirstTime'))
        op = '='
        rval = Constant(1)
        fileAST.ext.append(Assignment(lval,op,rval))

        allocateBuffer = EmptyFuncDecl('AllocateBuffers')
        fileAST.ext.append(allocateBuffer)

        listSetMemSize = []
        for entry in self.ArrayIdToDimName:
            n = self.ArrayIdToDimName[entry]
            lval = Id(self.Mem[entry])
            op = '='
            rval = BinOp(Id(n[0]),'*', Id('sizeof('+\
                dictTypeHostPtrs[entry][0]+')'))
            if len(n) == 2:
                rval = BinOp(Id(n[1]),'*', rval)
            listSetMemSize.append(Assignment(lval,op,rval))

        allocateBuffer.compound.statements.extend([GroupCompound(listSetMemSize)])
        #fileAST.ext.append(GroupCompound(listSetMemSize))

        ErrName = 'oclErrNum'
        lval = TypeId(['cl_int'], Id(ErrName))
        op = '='
        rval = Id('CL_SUCCESS')
        clSuc = Assignment(lval,op,rval)
        allocateBuffer.compound.statements.extend(\
            [GroupCompound([clSuc])])
        
        for n in dictNToDevPtr:
            lval = Id(dictNToDevPtr[n])
            op = '='
            arglist = ArgList([Id('context'),\
                               Id('CL_MEM_COPY_HOST_PTR'),\
                               Id(dictNToSize[n]),\
                               Id(dictNToHstPtr[n]),\
                               Id('&'+ErrName)])
            rval = FuncDecl(Id('clCreateBuffer'), arglist, Compound([]))
            allocateBuffer.compound.statements.append(\
                Assignment(lval,op,rval))
            arglist = ArgList([Id(ErrName), Constant("clCreateBuffer " + lval.name)])
            ErrCheck = FuncDecl(Id('oclCheckErr'),arglist, Compound([]))
            allocateBuffer.compound.statements.append(ErrCheck)

        setArgumentsKernel = EmptyFuncDecl('SetArguments'+self.DevFuncId)
        fileAST.ext.append(setArgumentsKernel)
        ArgBody = setArgumentsKernel.compound.statements
        ArgBody.append(clSuc)
        cntName = Id('counter')
        lval = TypeId(['int'], cntName)
        op = '='
        rval = Constant(0)
        ArgBody.append(Assignment(lval,op,rval))
        
        for n in dictNToDimNames:
            ## add dim arguments to set of ids
            NonArrayIds.add(dictNToDimNames[n][0])
            # Add types of dimensions for size arguments
            dictTypeHostPtrs[dictNToDimNames[n][0]] = ['size_t']
        
        for n in self.RemovedIds:
            dictTypeHostPtrs.pop(n,None)

        ## clSetKernelArg for Arrays
        for n in dictTypeHostPtrs:
            lval = Id(ErrName)
            op = '|='
            type = dictTypeHostPtrs[n]
            if len(type) == 2:
                arglist = ArgList([kernelId,\
                                   Increment(cntName,'++'),\
                                   Id('sizeof(cl_mem)'),\
                                   Id('(void *) &' + dictNToDevPtr[n])])
                rval = FuncDecl(Id('clSetKernelArg'),arglist, Compound([]))
                ArgBody.append(Assignment(lval,op,rval))
            else:
                type = type[0]
                if type == 'size_t' or type == 'unsigned':
                    cl_type = 'cl_uint'
                arglist = ArgList([kernelId,\
                                   Increment(cntName,'++'),\
                                   Id('sizeof('+cl_type+')'),\
                                   Id('(void *) &' + n)])
                rval = FuncDecl(Id('clSetKernelArg'),arglist, Compound([]))
                ArgBody.append(Assignment(lval,op,rval))
        
        arglist = ArgList([Id(ErrName), Constant('clSetKernelArg')])
        ErrId = Id('oclCheckErr')
        ErrCheck = FuncDecl(ErrId, arglist, Compound([]))
        ArgBody.append(ErrCheck)

        
        execKernel = EmptyFuncDecl('Exec' + self.DevFuncTypeId.name.name)
        fileAST.ext.append(execKernel)
        execBody = execKernel.compound.statements
        execBody.append(clSuc)
        eventName = Id('GPUExecution')
        event = TypeId(['cl_event'], eventName)
        execBody.append(event)

        for n in self.Worksize:
            lval = TypeId(['size_t'], Id(self.Worksize[n] + '[]'))
            op = '='
            if n == 'local':
                rval = ArrayInit([Id('LSIZE'), Id('LSIZE')])
            else:
                initlist = []
                for m in reversed(self.GridIndices):
                    initlist.append(Id(self.UpperLimit[m]))
                rval = ArrayInit(initlist)
            execBody.append(Assignment(lval,op,rval))

        lval = ErrId
        op = '='
        arglist = ArgList([Id('command_queue'),\
                           Id(self.KernelName),\
                           Constant(self.ParDim),\
                           Constant(0),\
                           Id(self.Worksize['global']),\
                           Id(self.Worksize['local']),\
                           Constant(0), Id('NULL'), \
                           Id('&' + eventName.name)])
        rval = FuncDecl(Id('clEnqueueNDRangeKernel'),arglist, Compound([]))
        execBody.append(Assignment(lval,op,rval))
        
        arglist = ArgList([Id(ErrName), Constant('clEnqueueNDRangeKernel')])
        ErrCheck = FuncDecl(ErrId, arglist, Compound([]))
        execBody.append(ErrCheck)


        for n in self.WriteOnly:
            lval = ErrId
            op = '='
            arglist = ArgList([Id('command_queue'),\
                               Id(self.DevId[n]),\
                               Id('CL_TRUE'),\
                               Constant(0),\
                               Id(self.Mem[n]),\
                               Id(self.HstId[n]),\
                               Constant(1),
                               Id('&' + eventName.name),Id('NULL')])
            rval = FuncDecl(Id('clEnqueueReadBuffer'),arglist, Compound([]))
            execBody.append(Assignment(lval,op,rval))
            
        arglist = ArgList([Id(ErrName), Constant('clEnqueueReadBuffer')])
        ErrCheck = FuncDecl(ErrId, arglist, Compound([]))
        execBody.append(ErrCheck)


        runOCL = EmptyFuncDecl('RunOCL' + self.KernelName)
        fileAST.ext.append(runOCL)
        runOCLBody = runOCL.compound.statements

        argIds = self.NonArrayIds.union(self.ArrayIds)
        typeIdList = []
        for n in argIds:
            type = self.Type[n]
            typeIdList.append(TypeId(type,Id(n)))
            try:
                for m in self.ArrayIdToDimName[n]:
                    type = ['size_t']
                    typeIdList.append(TypeId(type, Id(m)))
            except KeyError:
                pass
        
        arglist = ArgList(typeIdList)
        runOCL.arglist = arglist

        print "self.index " , self.index
        print "self.UpperLimit " , self.UpperLimit
        print "self.NumDims " , self.NumDims
        print "self.ArrayIds " , self.ArrayIds
        print "self.IndexInSubscript " , self.IndexInSubscript
        print "self.NonArrayIds " , self.NonArrayIds
        print "self.RemovedIds " , self.RemovedIds
        print "self.IndexToGlobalId " , self.IndexToGlobalId
        print "self.GridIndices " , self.GridIndices
        print "self.Kernel " , self.Kernel
        print "self.ArrayIdToDimName " , self.ArrayIdToDimName
        print "self.DevArgList " , self.DevArgList
        print "self.DevFuncTypeId " , self.DevFuncTypeId
        print "self.DevId " , self.DevId
        print "self.HstId " , self.HstId
        print "self.Type " , self.Type
        print "self.Mem " , self.Mem
        print "self.ParDim " , self.ParDim
        print "self.Worksize " , self.Worksize
        print "self.IdxToDim " , self.IdxToDim
        print "self.WriteOnly " , self.WriteOnly

        return fileAST

class FindReadWrite(NodeVisitor):
    """ Returns a mapping of arrays to either
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
    """ Finds Id's in an for loop initialization.
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
        self.end = dict()
    def visit_ForLoop(self, node):
        IdVis = InitIds()
        IdVis.visit(node.init)
        self.index.extend(IdVis.index)
        self.visit(node.compound)
        try:
            self.end[IdVis.index[0]] = (node.cond.rval.name)
        except AttributeError:
            self.end[IdVis.index[0]] = 'Unknown'
            

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
