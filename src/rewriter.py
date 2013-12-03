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
        # dict of the upper limit of the loop indices
        self.UpperLimit = dict()
        # dict of the lower limit of the loop indices
        self.LowerLimit = dict()
        # The local work group size
        self.Local = dict()
        self.Local['name'] = 'LSIZE'
        self.Local['size'] = '4'
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
        # to their function returning their thread id in the kernel
        self.IndexToThreadId = dict()
        self.IndexToLocalId = dict()
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
        # List of arguments for the kernel
        self.KernelArgs = list()
        ########################################################
        # Datastructures used when performing transformations  #
        ########################################################
        # Holds the sub-AST in AllocateBuffers
        # that we add transposition to.
        self.Transposition = None
        # Dict containing the name and type for each kernel argument
        # set in SetArguments
        self.KernelArgs = dict()
        # Holds information about which names have been swapped
        # in a transposition
        self.NameSwap = dict()
        # Holds information about which subscripts have been swapped
        # in a transposition
        self.SubSwap = dict()

        # Holds information about which indices have been swapped
        # in a transposition
        self.IdxSwap = dict()
        # Holds information about which dimensions have been swapped
        # in a transposition
        self.DimSwap = dict()
        # Holds additional variables that we add
        # when we to perform transformations       
        self.GlobalVars = dict()
        
        
    def initOriginal(self, ast):
        loops = ForLoops()
        loops.visit(ast)
        forLoopAst = loops.ast
        loopIndices = LoopIndices()
        loopIndices.visit(forLoopAst)
        self.index = loopIndices.index
        self.UpperLimit = loopIndices.end
        self.LowerLimit = loopIndices.start

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
        localMap = dict()
        firstIdx = initIds.index[0]
        idMap[firstIdx] = 'get_global_id(0)'
        localMap[firstIdx] = 'get_local_id(0)'
        gridIds.extend(initIds.index)
        kernel = perfectForLoop.ast.compound
        if perfectForLoop.depth == 2:
            initIds = InitIds()
            initIds.visit(kernel.statements[0].init)
            kernel = kernel.statements[0].compound
            secondIdx = initIds.index[0]
            idMap[secondIdx] = 'get_global_id(1)'
            localMap[secondIdx] = 'get_local_id(1)'
            gridIds.extend(initIds.index)
            (idMap[gridIds[0]], idMap[gridIds[1]]) = (idMap[gridIds[1]], idMap[gridIds[0]])
            (localMap[gridIds[0]], localMap[gridIds[1]]) = (localMap[gridIds[1]], localMap[gridIds[0]])

        self.IndexToLocalId = localMap
        self.IndexToThreadId = idMap
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

        for n in self.ArrayIdToDimName:
            for m in self.ArrayIdToDimName[n]:
                self.Type[m] = ['size_t']
            
            
        kernelName = self.DevFuncTypeId.name.name
        
        self.KernelName = kernelName + 'Kernel'
        self.Worksize['local'] = kernelName + '_local_worksize'
        self.Worksize['global'] = kernelName + '_global_worksize'
        self.Worksize['offset'] = kernelName + '_global_offset'
        findReadWrite = FindReadWrite(self.ArrayIds)
        findReadWrite.visit(ast)
        self.ReadWrite = findReadWrite.ReadWrite

        for n in self.ReadWrite:
            pset = self.ReadWrite[n]
            if len(pset) == 1 and 'write' in pset:
                self.WriteOnly.append(n)

        argIds = self.NonArrayIds.union(self.ArrayIds) - self.RemovedIds

        for n in argIds:
            tmplist = [n]
            try:
                if self.ParDim == 2:
                    tmplist.append(self.ArrayIdToDimName[n][0])
            except KeyError:
                pass
            for m in tmplist:
                self.KernelArgs[m] = self.Type[m]

        self.Transposition = GroupCompound([Comment('// Transposition')])

    def dataStructures(self):
        print "self.index " , self.index
        print "self.UpperLimit " , self.UpperLimit
        print "self.LowerLimit " , self.LowerLimit
        print "self.NumDims " , self.NumDims
        print "self.ArrayIds " , self.ArrayIds
        print "self.IndexInSubscript " , self.IndexInSubscript
        print "self.NonArrayIds " , self.NonArrayIds
        print "self.RemovedIds " , self.RemovedIds
        print "self.IndexToThreadId " , self.IndexToThreadId
        print "self.IndexToLocalId " , self.IndexToLocalId
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
        print "TRANSFORMATIONS"
        print "self.Transposition " , self.Transposition
        print "self.KernelArgs " , self.KernelArgs
        print "self.NameSwap " , self.NameSwap


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
        findDim = FindDim(arrays2.numIndices)
        findDim.visit(ast)
        rewriteArrayRef = RewriteArrayRef(self.NumDims,
                                          self.ArrayIdToDimName,
                                          self)
        rewriteArrayRef.visit(ast)

    def rewriteToDeviceCTemp(self, ast, changeAST = True):


        findDeviceArgs = FindDeviceArgs(self.NonArrayIds)
        findDeviceArgs.visit(ast)
        findFunction = FindFunction()
        findFunction.visit(ast)

        # add OpenCL keywords to indicate the kernel function.
        findFunction.typeid.type.insert(0, '__kernel')
        
        exchangeIndices = ExchangeIndices(self.IndexToThreadId)
        exchangeIndices.visit(self.Kernel)
        newast =  FuncDecl(findFunction.typeid, ArgList(findDeviceArgs.arglist,ast.coord), self.Kernel, ast.coord)
        if changeAST:
            ast.ext = list()
            ast.ext.append(newast)
        
    def rewriteToDeviceCRelease(self, ast):

        arglist = list()
        argIds = self.NonArrayIds.union(self.ArrayIds) - self.RemovedIds
        # The list of arguments for the kernel
        dictTypeHostPtrs = copy.deepcopy(self.Type)
        for n in self.ArrayIds:
            dictTypeHostPtrs[self.ArrayIdToDimName[n][0]] = ['size_t']

        for n in self.KernelArgs:
            type = dictTypeHostPtrs[n]
            if type[0] == 'size_t':
                type[0] = 'unsigned'
            if len(type) == 2:
                type.insert(0, '__global')
            arglist.append(TypeId(type, Id(n)))


        rewriteArrayRef = RewriteArrayRef(self.NumDims, self.ArrayIdToDimName,self)
        rewriteArrayRef.visit(self.Kernel)

        exchangeIndices = ExchangeIndices(self.IndexToThreadId)
        exchangeIndices.visit(self.Kernel)

        typeid = copy.deepcopy(self.DevFuncTypeId)
        typeid.type.insert(0, '__kernel')
        
        newast =  FuncDecl(typeid, ArgList(arglist), self.Kernel)
        ast.ext = list()
        ast.ext.append(Id('#define LSIZE ' + str(self.Local['size'])))
        ast.ext.append(newast)


    def localMemory(self, arrName, west = 0, north = 0, east = 0, south = 0):
        localName = arrName + '_local'
        arrayinit = '[' + self.Local['size'] + ']' + '[' + self.Local['size'] + ']'
        
        localId = Id(localName + arrayinit)
        localTypeId = TypeId(['__local'] + [self.Type[arrName][0]], localId)
        ## self.NumDims[localName] = self.NumDims[arrName]

        self.Kernel.statements.insert(0,localTypeId)
        
        arrayId = Id(arrName)
        arraySubscriptGlobal = []
        arraySubscriptLocal = []
        for n in self.GridIndices:
            arraySubscriptGlobal.append(Id(self.IndexToThreadId[n]))
            arraySubscriptLocal.append(Id(self.IndexToLocalId[n]))
        
        lval = ArrayRef(Id(localName), arraySubscriptLocal)
        rval = ArrayRef(arrayId, arraySubscriptGlobal)
        self.Kernel.statements.insert(1, Assignment(lval,rval))

    def transpose(self, arrName):
        if self.NumDims[arrName] != 2:
            print "Array ", arrName , "of dimension " , \
                  self.NumDims[arrName], "cannot be transposed"
            return
        hstName = self.HstId[arrName]
        hstTransName = hstName + '_trans'
        self.GlobalVars[hstTransName] = ''
        self.Type[hstTransName] = self.Type[arrName]
        # Swap the hst ptr
        self.NameSwap[hstName] = hstTransName
        # Swap the dimension argument
        dimName = self.ArrayIdToDimName[arrName]
        self.NameSwap[dimName[0]] = dimName[1]

        lval = Id(hstTransName)
        natType = self.Type[arrName][0]
        rval = Id('new ' + natType + '['\
                  + self.Mem[arrName] + ']')
        self.Transposition.statements.append(Assignment(lval,rval))
        arglist = ArgList([Id(hstName),\
                   Id(hstTransName),\
                   Id(dimName[0]),\
                   Id(dimName[1])])
        trans = FuncDecl(Id('transpose<'+natType+'>'), arglist, Compound([]))
        self.Transposition.statements.append(trans)
        self.SubSwap[arrName] = True

    def generateBoilerplateCode(self, ast):


        dictNToNumScripts = self.NumDims
        dictNToDimNames = self.ArrayIdToDimName

        kernel = self.Kernel
        idMap = self.IndexToThreadId
        gridIds = self.GridIndices
        NonArrayIds = copy.deepcopy(self.NonArrayIds)
        otherIds = self.ArrayIds.union(self.NonArrayIds) - self.RemovedIds


        fileAST = FileAST([])

        fileAST.ext.append(Id('#include \"StartUtil.cpp\"'))
        fileAST.ext.append(Id('using namespace std;'))
        fileAST.ext.append(Id('#define LSIZE ' + str(self.Local['size'])))


        kernelId = Id(self.KernelName)
        kernelTypeid = TypeId(['cl_kernel'], kernelId, 0)
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

        for n in self.GlobalVars:
            type = self.Type[n]
            listHostPtrs.append(TypeId(type, Id(n), 0))

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
        rval = Constant(1)
        fileAST.ext.append(Assignment(lval,rval))

        allocateBuffer = EmptyFuncDecl('AllocateBuffers')
        fileAST.ext.append(allocateBuffer)

        listSetMemSize = []
        for entry in self.ArrayIdToDimName:
            n = self.ArrayIdToDimName[entry]
            lval = Id(self.Mem[entry])
            rval = BinOp(Id(n[0]),'*', Id('sizeof('+\
                self.Type[entry][0]+')'))
            if len(n) == 2:
                rval = BinOp(Id(n[1]),'*', rval)
            listSetMemSize.append(Assignment(lval,rval))

        allocateBuffer.compound.statements.append(\
            GroupCompound(listSetMemSize))

        allocateBuffer.compound.statements.append(\
            self.Transposition)
        
        ErrName = 'oclErrNum'
        lval = TypeId(['cl_int'], Id(ErrName))
        rval = Id('CL_SUCCESS')
        clSuc = Assignment(lval,rval)
        allocateBuffer.compound.statements.extend(\
            [GroupCompound([clSuc])])

        for n in dictNToDevPtr:
            lval = Id(dictNToDevPtr[n])
            op = '='
            arrayn = dictNToHstPtr[n]
            try:
                arrayn = self.NameSwap[arrayn]
            except KeyError:
                pass
            arglist = ArgList([Id('context'),\
                               Id('CL_MEM_COPY_HOST_PTR'),\
                               Id(dictNToSize[n]),\
                               Id(arrayn),\
                               Id('&'+ErrName)])
            rval = FuncDecl(Id('clCreateBuffer'), arglist, Compound([]))
            allocateBuffer.compound.statements.append(\
                Assignment(lval,rval))
            arglist = ArgList([Id(ErrName), Constant("clCreateBuffer " + lval.name)])
            ErrCheck = FuncDecl(Id('oclCheckErr'),arglist, Compound([]))
            allocateBuffer.compound.statements.append(ErrCheck)

        setArgumentsKernel = EmptyFuncDecl('SetArguments'+self.DevFuncId)
        fileAST.ext.append(setArgumentsKernel)
        ArgBody = setArgumentsKernel.compound.statements
        ArgBody.append(clSuc)
        cntName = Id('counter')
        lval = TypeId(['int'], cntName)
        rval = Constant(0)
        ArgBody.append(Assignment(lval,rval))
        
        for n in dictNToDimNames:
            ## add dim arguments to set of ids
            NonArrayIds.add(dictNToDimNames[n][0])
            # Add types of dimensions for size arguments
            dictTypeHostPtrs[dictNToDimNames[n][0]] = ['size_t']
        
        for n in self.RemovedIds:
            dictTypeHostPtrs.pop(n,None)
        
        ## clSetKernelArg for Arrays
        for n in self.KernelArgs:
            lval = Id(ErrName)
            op = '|='
            type = self.Type[n]
            if len(type) == 2:
                arglist = ArgList([kernelId,\
                                   Increment(cntName,'++'),\
                                   Id('sizeof(cl_mem)'),\
                                   Id('(void *) &' + dictNToDevPtr[n])])
                rval = FuncDecl(Id('clSetKernelArg'),arglist, Compound([]))
            else:
                try:
                    n = self.NameSwap[n]
                except KeyError:
                    pass
                cl_type = type[0]
                arglist = ArgList([kernelId,\
                                   Increment(cntName,'++'),\
                                   Id('sizeof('+cl_type+')'),\
                                   Id('(void *) &' + n)])
                rval = FuncDecl(Id('clSetKernelArg'),arglist, Compound([]))
            ArgBody.append(Assignment(lval,rval,op))
        
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
            if n == 'local':
                rval = ArrayInit([Id('LSIZE'), Id('LSIZE')])
            elif n == 'global':
                initlist = []
                for m in reversed(self.GridIndices):
                    initlist.append(Id(self.UpperLimit[m]\
                                       +' - '+ self.LowerLimit[m]))
                rval = ArrayInit(initlist)
            else:
                initlist = []
                for m in reversed(self.GridIndices):
                    initlist.append(Id(self.LowerLimit[m]))
                rval = ArrayInit(initlist)
                
            execBody.append(Assignment(lval,rval))

        lval = Id(ErrName)
        arglist = ArgList([Id('command_queue'),\
                           Id(self.KernelName),\
                           Constant(self.ParDim),\
                           Id(self.Worksize['offset']),\
                           Id(self.Worksize['global']),\
                           Id(self.Worksize['local']),\
                           Constant(0), Id('NULL'), \
                           Id('&' + eventName.name)])
        rval = FuncDecl(Id('clEnqueueNDRangeKernel'),arglist, Compound([]))
        execBody.append(Assignment(lval,rval))
        
        arglist = ArgList([Id(ErrName), Constant('clEnqueueNDRangeKernel')])
        ErrCheck = FuncDecl(ErrId, arglist, Compound([]))
        execBody.append(ErrCheck)


        for n in self.WriteOnly:
            lval = Id(ErrName)
            arglist = ArgList([Id('command_queue'),\
                               Id(self.DevId[n]),\
                               Id('CL_TRUE'),\
                               Constant(0),\
                               Id(self.Mem[n]),\
                               Id(self.HstId[n]),\
                               Constant(1),
                               Id('&' + eventName.name),Id('NULL')])
            rval = FuncDecl(Id('clEnqueueReadBuffer'),arglist, Compound([]))
            execBody.append(Assignment(lval,rval))
            
        arglist = ArgList([Id(ErrName), Constant('clEnqueueReadBuffer')])
        ErrCheck = FuncDecl(ErrId, arglist, Compound([]))
        execBody.append(ErrCheck)


        runOCL = EmptyFuncDecl('RunOCL' + self.KernelName)
        fileAST.ext.append(runOCL)
        runOCLBody = runOCL.compound.statements

        argIds = self.NonArrayIds.union(self.ArrayIds)
        typeIdList = []
        ifThenList = []
        for n in argIds:
            type = self.Type[n]
            argn = Id('arg_'+n)
            typeIdList.append(TypeId(type,argn))
            try:
                newn = self.HstId[n]
            except KeyError:
                newn = n
            lval = Id(newn)
            rval = argn
            ifThenList.append(Assignment(lval,rval))
            try:
                for m in self.ArrayIdToDimName[n]:
                    type = ['size_t']
                    argm = Id('arg_'+m)
                    lval = Id(m)
                    rval = argm
                    ifThenList.append(Assignment(lval,rval))
                    typeIdList.append(TypeId(type, argm))
            except KeyError:
                pass
        
        arglist = ArgList(typeIdList)
        runOCL.arglist = arglist

        arglist = ArgList([])
        ifThenList.append(FuncDecl(Id('StartUpGPU'), arglist, Compound([])))
        ifThenList.append(FuncDecl(Id('AllocateBuffers'), arglist, Compound([])))
        arglist = ArgList([Constant(self.DevFuncId),
                           Constant(self.DevFuncId+'.cl'),
                           Id('&' + self.KernelName),
                           Constant('')])
        ifThenList.append(FuncDecl(Id('compileKernelFromFile'), arglist, Compound([])))

        ifThenList.append(FuncDecl(Id('SetArguments'+self.DevFuncId), ArgList([]), Compound([])))

        runOCLBody.append(IfThen(Id('isFirstTime'), Compound(ifThenList)))
        arglist = ArgList([])
        runOCLBody.append(FuncDecl(Id('Exec' + self.DevFuncId), arglist, Compound([])))


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
    def __init__(self, NumDims, ArrayIdToDimName, data):
        self.data = data
        self.NumDims = NumDims
        self.ArrayIdToDimName = ArrayIdToDimName
    
    def visit_ArrayRef(self, node):
        n = node.name.name
        try:
            if self.data.NumDims[n] == 2:
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
                ## print topbinop
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
        self.start = dict()
    def visit_ForLoop(self, node):
        IdVis = InitIds()
        IdVis.visit(node.init)
        self.index.extend(IdVis.index)
        self.visit(node.compound)
        try:
            self.end[IdVis.index[0]] = (node.cond.rval.name)
            self.start[IdVis.index[0]] = (node.init.rval.value)
        except AttributeError:
            self.end[IdVis.index[0]] = 'Unknown'
            self.start[IdVis.index[0]] = 'Unknown'
            

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
        for s in node.subscript:
            numIndcs.visit(s)
        if name not in self.numIndices:
            self.numIndices[name] = numIndcs.num
            self.numSubscripts[name] = numIndcs.num
            self.indexIds[name] = numIndcs.found
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
