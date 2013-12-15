import copy
import os
from visitor import *

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
        self.Local['size'] = '8'
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
        self.IndexToLocalVar = dict()
        # The indices that we parallelize
        self.GridIndices = list()
        # The OpenCl kernel before anything
        self.Kernel = None
        # The "inside" of the OpenCl kernel after parallelization
        self.InsideKernel = None
        # The name of the kernel, i.e. the FuncName + Kernel
        self.KernelName = None
        # The mapping from the array ids to a list of 
        # the names of their dimensions
        self.ArrayIdToDimName = dict()
        # ArrayRef inside a loop in the kernel
        # Mapping from Id to AST ArrayRef node
        self.LoopArray = dict()
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
        # dict of indices to loops in the kernel
        self.Loops = dict()
        
        # List of arguments for the kernel
        ## self.KernelArgs = list()
        ########################################################
        # Datastructures used when performing transformations  #
        ########################################################
        # Holds the sub-AST in AllocateBuffers
        # that we add transposition to.
        self.Transposition = None
        # Holds the sub-AST in AllocateBuffers
        # that we add constant memory pointer initializations to.
        self.ConstantMemory = None
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
        # Holds additional global variables such as pointers that we add
        # when we to perform transformations       
        self.GlobalVars = dict()
        # Holds additional cl_mem variables that we add
        # when we to perform Constant Memory transformation       
        self.ConstantMem = dict()
        # Name swap in relation to [local memory]
        self.LocalSwap = dict()
        # Extra things that we add to ids [local memory]
        self.Add = dict()
        
        
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

        innerbody = perfectForLoop.inner
        firstLoop = ForLoops()
        firstLoop.visit(innerbody.compound)
        loopIndices = LoopIndices()
        loopIndices.visit(firstLoop.ast)
        self.Loops = loopIndices.Loops


        self.InsideKernel = firstLoop.ast


        arrays = Arrays([])
        if firstLoop.ast is None: # No inner loop
            arrays.visit(innerbody.compound)
        else:
            arrays.visit(firstLoop.ast)

        self.LoopArrays = arrays.LoopArrays
        
        initIds = InitIds()
        initIds.visit(perfectForLoop.ast.init)
        gridIds = list()
        idMap = dict()
        localMap = dict()
        localVarMap = dict()
        firstIdx = initIds.index[0]
        idMap[firstIdx] = 'get_global_id(0)'
        localMap[firstIdx] = 'get_local_id(0)'
        localVarMap[firstIdx] = 'l' + firstIdx
        self.ReverseIdx = dict()
        self.ReverseIdx[0] = 1
        gridIds.extend(initIds.index)
        kernel = perfectForLoop.ast.compound
        self.ReverseIdx[1] = 0
        if perfectForLoop.depth == 2:
            initIds = InitIds()
            initIds.visit(kernel.statements[0].init)
            kernel = kernel.statements[0].compound
            secondIdx = initIds.index[0]
            idMap[secondIdx] = 'get_global_id(1)'
            localMap[secondIdx] = 'get_local_id(1)'
            localVarMap[secondIdx] = 'l' + secondIdx
            gridIds.extend(initIds.index)
            (idMap[gridIds[0]], idMap[gridIds[1]]) = (idMap[gridIds[1]], idMap[gridIds[0]])
            (localMap[gridIds[0]], localMap[gridIds[1]]) = (localMap[gridIds[1]], localMap[gridIds[0]])
            ## (localVarMap[gridIds[0]], localVarMap[gridIds[1]]) = (localVarMap[gridIds[1]], localVarMap[gridIds[0]])

        self.IndexToLocalId = localMap
        self.IndexToLocalVar = localVarMap
        self.IndexToThreadId = idMap
        self.GridIndices = gridIds
        self.Kernel = kernel
        for i, n in enumerate(reversed(self.GridIndices)):
            self.IdxToDim[i] = n
            
        
        findDim = FindDim(self.NumDims)
        findDim.visit(ast)
        self.ArrayIdToDimName = findDim.dimNames
        self.RemovedIds = set(self.UpperLimit[i] for i in self.GridIndices)

        idsStillInKernel = Ids()
        idsStillInKernel.visit(self.Kernel)
        self.RemovedIds = self.RemovedIds - idsStillInKernel.ids

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
                if self.NumDims[n] == 2:
                    tmplist.append(self.ArrayIdToDimName[n][0])
            except KeyError:
                pass
            for m in tmplist:
                self.KernelArgs[m] = self.Type[m]

        self.Transposition = GroupCompound([Comment('// Transposition')])
        self.ConstantMemory = GroupCompound([Comment('// Constant Memory')])
        arrays = Arrays(self.index)
        arrays.visit(ast)
        self.Subscript = arrays.Subscript

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
        print "self.IndexToLocalVar " , self.IndexToLocalVar
        print "self.ReverseIdx ", self.ReverseIdx
        print "self.GridIndices " , self.GridIndices
        ## print "self.Kernel " , self.Kernel
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
        print "self.Subscript " , self.Subscript
        print "TRANSFORMATIONS"
        print "self.Transposition " , self.Transposition
        print "self.ConstantMemory " , self.ConstantMemory
        print "self.KernelArgs " , self.KernelArgs
        print "self.NameSwap " , self.NameSwap
        print "self.LocalSwap " , self.LocalSwap
        print "self.LoopArrays " , self.LoopArrays
        print "self.Add ", self.Add
        print "self.GlobalVars ", self.GlobalVars
        print "self.ConstantMem " , self.ConstantMem
        print "self.Loops " , self.Loops

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
            type = copy.deepcopy(self.KernelArgs[n])
            if type[0] == 'size_t':
                type[0] = 'unsigned'
            if len(type) == 2:
                type.insert(0, '__global')
            arglist.append(TypeId(type, Id(n)))


        exchangeArrayId = ExchangeArrayId(self.LocalSwap)

        for n in self.LoopArrays.values():
            for m in n:
                exchangeArrayId.visit(m)

        for n in self.Add:
            addToIds = AddToId(n, self.Add[n])
            addToIds.visit(self.InsideKernel.compound)

        rewriteArrayRef = RewriteArrayRef(self.NumDims, self.ArrayIdToDimName, self)
        rewriteArrayRef.visit(self.Kernel)

        arrays = self.ArrayIds
        
        exchangeIndices = ExchangeId(self.IndexToThreadId)
        exchangeIndices.visit(self.Kernel)


        exchangeTypes = ExchangeTypes()
        exchangeTypes.visit(self.Kernel)
        
        
        typeid = copy.deepcopy(self.DevFuncTypeId)
        typeid.type.insert(0, '__kernel')


        
        newast =  FuncDecl(typeid, ArgList(arglist), self.Kernel)
        ast.ext = list()
        ast.ext.append(Id('#define LSIZE ' + str(self.Local['size'])))
        ast.ext.append(newast)


    def constantMemory(self, arrNames):

        
        split = dict()
        for name in arrNames:
            globalarr = set()
            constantarr = set()
            for n in self.Subscript[name]:
                ids = Ids()
                for s in n:
                    ids.visit(s)
                
                for i in self.GridIndices:
                    if i in ids.ids:
                        globalarr.add(name)
                    else:
                        constantarr.add(name)
                        
                if name in globalarr and name in constantarr:
                    split[name] = True
                else:
                    split[name] = False

        print "split ", split

        # Add new constant pointer
        ptrname = 'constant' + ''.join(arrNames)
        hst_ptrname = 'hst_ptr_' + ptrname
        dev_ptrname = 'dev_ptr_' + ptrname
        print ptrname
        typeset = set()
        for name in arrNames:
            typeset.add(self.Type[name][0])

        if len(typeset) > 1:
            print "Conflicting types in constant memory transformation... Aborting"
            return

        ptrtype = [typeset.pop(), '*']
        # Add the ptr to central data structures
        self.Type[ptrname] = ptrtype
        self.DevId[ptrname] = dev_ptrname
        self.HstId[ptrname] = hst_ptrname
        self.Mem[ptrname] = self.HstId[ptrname]+'_mem_size'

        # Add the ptr to be a kernel argument
        self.KernelArgs[ptrname] = ['__constant'] +  ptrtype
        self.GlobalVars[ptrname] = ''
        self.ConstantMem[ptrname] = arrNames

        for n in split:
            if not split[n]:
                self.KernelArgs.pop(n)
                self.DevId.pop(n)
        # Add pointer allocation to AllocateBuffers
        lval = Id(self.HstId[ptrname])
        rval = Id('new ' + self.Type[ptrname][0] + '['\
                  + self.Mem[ptrname] + ']')
        
        self.ConstantMemory.statements.append(Assignment(lval, rval))
        forloop = copy.deepcopy(self.InsideKernel)
        newcomp = []
        forcomp = []
        groupComp = GroupCompound(newcomp)
        forloop.compound = Compound(forcomp)
        loopcount = forloop.init.lval.name.name
        # Add the for loop from the kernel
        newcomp.append(forloop)
        # Add the array reads to the loop body
        constantdim = sum([ (len(n)) for n in [self.LoopArrays[m] for m in arrNames]])
        constantcount = constantdim
        constantcount = 0
        for n in arrNames:
            aref = self.LoopArrays[n]
            
            for a in aref:
                acopy = copy.deepcopy(a)
                
                rewriteArrayRef = RewriteArrayRef(self.NumDims, self.ArrayIdToDimName, self)
                rewriteArrayRef.visit(acopy)

                sub = acopy.subscript
                name = ptrname
                a.name.name = name
                lsub = [Id(str(constantdim) + '*' + loopcount + '+' + str(constantcount) )]
                lval = ArrayRef(Id(self.HstId[ptrname]), lsub)
                rval = ArrayRef(Id(self.HstId[n]), sub)
                constantcount += 1
                a.subscript = lsub
                ass = Assignment(lval, rval)
                forcomp.append(ass)
                ## For exchanges array ids, does not seem useful atm.
                ## if self.HstId[n] in self.LoopArrays:
                ##     self.LoopArrays[self.HstId[n]].append(rval)
                ## else:
                ##     self.LoopArrays[self.HstId[n]] = [rval]
                                        

        # Add dimensions to ptrname
        print "constantdim ", constantdim
        
        
        #self.ArrayIdToDimName
        self.ConstantMemory.statements.append(groupComp)
        

    def localMemory2(self, arrNames):
        # Find the subscripts that would be suitable to have in local memory
        local = dict()
        for name in arrNames:
            for sub in self.Subscript[name]:
                for s in sub:
                    ids = LoopIds(self.Loops.keys())
                    ids.visit(s)
                    iset = ids.ids
                    if len(iset) > 1:
                        print "More than one loop index in a subscript?"
                        return
                    if len(iset) == 1:
                        if name in local:
                            local[name].append(sub)
                        else:
                            local[name] = [sub]


        for n in local:
            print local[n]

        ## local[arrNames[0]][0][1].name = 'lll'

    def localMemory(self, arrNames, west = 0, north = 0, east = 0, south = 0, middle = 1):

        isInsideLoop = False
        try: 
            # Find out if arrName is inside a loop
            forLoops = ForLoops()
            forLoops.visit(self.Kernel)
            forLoopAst = forLoops.ast
            arrays = Arrays([])
            arrays.visit(forLoopAst)
            for arrName in arrNames:
                if arrName in arrays.ids:
                    isInsideLoop = True
                    print "INSIDE LOOP"
        except AttributeError:
            print "NOT INSIDE LOOP"

        if isInsideLoop:
            # find loop index
            loopIndices = LoopIndices()
            loopIndices.visit(forLoopAst)
            outeridx = loopIndices.index[0]
            print outeridx
            forLoopAst.inc = Increment(Id(outeridx), '+='+self.Local['size'])
            
            inneridx = outeridx*2
            self.Add[outeridx] = inneridx
            cond = BinOp(Id(inneridx), '<' , Constant(self.Local['size']))
            innerinc = Increment(Id(inneridx), '++')
            innercomp = copy.copy(forLoopAst.compound)
            innerloop = ForLoop(ConstantAssignment(inneridx), cond, \
                                innerinc, innercomp)
            forLoopAst.compound = Compound([innerloop])

        direction = [west, north, east, south, middle]
        dirname = [(0,-1), (1,0), (0,1), (-1,0), (0,0)]
        loadings = [elem for i, elem in enumerate(dirname)
                    if direction[i] == 1]
        if not loadings:
            loadings = [(0,0)]


        ## finding the correct local memory size
        arrName = arrNames[0]
        localDims = [int(self.Local['size'])\
                     for i in xrange(self.NumDims[arrName])]
        if self.ParDim == 1 and len(localDims) == 2:
            localDims[0] = 1;
        arrIdx = self.IndexInSubscript[arrName]
        localOffset = [int(self.LowerLimit[i])\
                     for i in arrIdx]

        for (x,y) in loadings:
            localDims[0] += abs(x)
            if self.NumDims[arrName] == 2:
                localDims[1] += abs(y)

        print "localDims " , localDims
        print "localOffset " , localOffset

        stats = []
        for arrName in arrNames:
            localName = arrName + '_local'
            arrayinit = '['
            for i, d in enumerate(localDims):
                arrayinit += str(d)
                if i == 0 and len(localDims) == 2:
                    arrayinit += '*'
            arrayinit += ']'

            localId = Id(localName + arrayinit)
            localTypeId = TypeId(['__local'] + [self.Type[arrName][0]], localId)
            self.NumDims[localName] = self.NumDims[arrName]
            self.LocalSwap[arrName] = localName
            self.ArrayIdToDimName[localName] = [self.Local['size'], self.Local['size']]
            stats.append(localTypeId)

        InitComp = GroupCompound(stats)
        stats2 = []
        LoadComp = GroupCompound(stats2)

        ## Insert local id with offset
        for i, offset in enumerate(localOffset):
            idd = self.ReverseIdx[i] if len(localOffset) == 2 else i
            if offset != 0:
                
                rval = BinOp(Id('get_local_id('+str(idd)+')'), '+', \
                             Constant(offset))
            else:
                rval = Id('get_local_id('+str(idd)+')')
            lval = TypeId(['unsigned'], Id('l'+self.GridIndices[i]))
            stats.append(Assignment(lval,rval))

        print "loadings " , loadings
        exchangeIndices = ExchangeIndices(self.IndexToLocalVar,  self.LocalSwap.values())
        
        ## Creating the loading of values into the local array.
        for arrName in arrNames:
            for k, l in enumerate(loadings):
                arrayId = Id(arrName)
                # get first ArrayRef
                aref = self.LoopArrays[arrName][k]
                subscript = aref.subscript
                lsub = copy.deepcopy(subscript)
                lval = ArrayRef(Id(self.LocalSwap[arrName]), lsub)
                rsub = copy.deepcopy(subscript)
                print "lsub ", lsub
                rval = ArrayRef(arrayId, rsub, extra = {'localMemory' : True})
                load = Assignment(lval,rval)
                exchangeId = ExchangeId(self.IndexToLocalVar)
                orisub = subscript
                for m in orisub:
                    exchangeId.visit(m)
                if isInsideLoop:
                    for i, n in enumerate(orisub):
                        addToId = Ids()
                        addToId.visit(n)
                        # REMEMBER: orisub[i] might not simply be an Id
                        # might need to do something more complicated here
                        if outeridx in addToId.ids:
                            orisub[i] = Id(inneridx)
                    print "outeridx ", outeridx

                    for i, n in enumerate(rsub): # GlobalLoad
                        idd = self.ReverseIdx[i] if self.NumDims[arrName] == 2 else i
                        locIdx = 'get_local_id('+str(idd)+')'
                        addToId = Ids()
                        addToId.visit(n)
                        if outeridx in addToId.ids:
                            rsub[i] = BinOp(rsub[i],'+',\
                            Id(locIdx))
                    for i, n in enumerate(lsub): # Local Write
                        idd = self.ReverseIdx[i] if self.NumDims[arrName] == 2 else i
                        locIdx = 'get_local_id('+str(idd)+')'
                        exchangeId = ExchangeId({''+outeridx : locIdx})
                        exchangeId.visit(n)                    

                stats2.append(load)

        # Must also create the barrier
        arglist = ArgList([Id('CLK_LOCAL_MEM_FENCE')])
        func = EmptyFuncDecl('barrier', type = [])
        func.arglist = arglist
        stats2.append(func)

        exchangeIndices.visit(InitComp)
        exchangeIndices.visit(LoadComp)
        if isInsideLoop:
            forLoopAst.compound.statements.insert(0,LoadComp)
        else:
            self.Kernel.statements.insert(0, LoadComp)
        self.Kernel.statements.insert(0, InitComp)
        
        

    def transpose(self, arrName):
        if self.NumDims[arrName] != 2:
            print "Array ", arrName , "of dimension " , \
                  self.NumDims[arrName], "cannot be transposed"
            return
        hstName = self.HstId[arrName]
        hstTransName = hstName + '_trans'
        self.GlobalVars[hstTransName] = ''
        self.HstId[hstTransName] = hstTransName
        self.Type[hstTransName] = self.Type[arrName]
        # Swap the hst ptr
        self.NameSwap[hstName] = hstTransName
        # Swap the dimension argument
        dimName = self.ArrayIdToDimName[arrName]
        self.NameSwap[dimName[0]] = dimName[1]

        if arrName not in self.WriteOnly:
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

        # Forget this and just swap subscripts immediately
        ## self.SubSwap[arrName] = True

        for sub in self.Subscript[arrName]:
            (sub[0], sub[1]) = \
                     (sub[1], sub[0])


    def generateBoilerplateCode(self, ast):


        dictNToNumScripts = self.NumDims
        dictNToDimNames = self.ArrayIdToDimName

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
            try:
                name = self.DevId[n]
                listDevBuffers.append(TypeId(['cl_mem'], Id(name)))
            except KeyError:
                pass

        for n in self.ConstantMem:
            name = self.DevId[n]
            listDevBuffers.append(TypeId(['cl_mem'], Id(name)))

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
            name = self.HstId[n]
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
        for n in self.Mem:
            sizeName = self.Mem[n]
            listMemSize.append(TypeId(['size_t'], Id(sizeName)))

        for n in self.ArrayIds:
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
        for entry in self.ArrayIds:
            n = self.ArrayIdToDimName[entry]
            lval = Id(self.Mem[entry])
            rval = BinOp(Id(n[0]),'*', Id('sizeof('+\
                self.Type[entry][0]+')'))
            if len(n) == 2:
                rval = BinOp(Id(n[1]),'*', rval)
            listSetMemSize.append(Assignment(lval,rval))

        for n in self.ConstantMem:
            terms = self.ConstantMem[n]
            rval = Id(self.Mem[terms[0]])
            for s in terms[1:]:
                rval = BinOp(rval, '+', Id(self.Mem[s]))

            lval = Id(self.Mem[n])
            listSetMemSize.append(Assignment(lval,rval))
            
        allocateBuffer.compound.statements.append(\
            GroupCompound(listSetMemSize))

        allocateBuffer.compound.statements.append(\
            self.Transposition)

        allocateBuffer.compound.statements.append(\
            self.ConstantMemory)
        
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
            if n in self.WriteOnly:
                flag = Id('CL_MEM_WRITE_ONLY')
                arraynId = Id('NULL')
            else:
                flag = Id('CL_MEM_USE_HOST_PTR')
                arraynId = Id(arrayn)
            arglist = ArgList([Id('context'),\
                               flag,\
                               Id(dictNToSize[n]),\
                               arraynId,\
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
                if cl_type == 'size_t':
                    cl_type = 'unsigned'
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
                local_worksize = [Id('LSIZE') for i in xrange(self.ParDim)]
                rval = ArrayInit(local_worksize)
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

        # add clFinish statement
        arglist = ArgList([Id('command_queue')])
        finish = FuncDecl(Id('clFinish'), arglist, Compound([]))
        execBody.append(Assignment(Id(ErrName), finish))
        
        arglist = ArgList([Id(ErrName), Constant('clFinish')])
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

