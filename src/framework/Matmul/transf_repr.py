
from Matmul.transf_visitor import *

class Transf_Repr(NodeVisitor):
    """ Class for rewriting of the original AST. Includes:
    1. the initial small rewritings,
    2. transformation into our representation,
    3. transforming from our representation to C-executable code,
    4. creating our representation of the device kernel code,
    5. creating a C-executable kernel code,
    6. Creating the host code (boilerplate code) 
    """

    
    def __init__(self, astrepr):
        # Original repr
        self.astrepr = astrepr
        # The types of the arguments for the kernel
        self.Type = astrepr.Type
        # The local work group size
        self.Local = dict()
        self.Local['name'] = 'LSIZE'
        self.Local['size'] = ['64']
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
        # The argument list in our IR
        self.DevArgList = list()
        # The name and type of the kernel function.
        self.DevFuncTypeId = None
        # The name of the kernel function.
        self.DevFuncId = None
        # The device names of the pointers in the boilerplate code
        self.DevId = dict()
        # The host names of the pointers in the boilerplate code
        self.HstId = dict()
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
        # List of arrays that are read only
        self.ReadOnly = list()
        # dict of indices to loops in the kernel
        self.Loops = dict()
        # Contains the loop indices for each subscript
        self.SubIdx = dict()
        # Contains a list for each arrayref of what loop indices
        # appear in the subscript.
        self.Subscript = dict()
        # The same as above but names are saved as strings instead
        # of Id(strings)
        self.SubscriptNoId = dict()
        # Decides whether we read back data from GPU
        self.NoReadBack = False
        # A list of calls to the transpose function which we perform
        # after data was read back from the GPU.
        self.WriteTranspose = list()
        # A mapping from array references to the loops they appear in.
        self.RefToLoop = dict()
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
        # Holds the sub-AST in AllocateBuffers
        # where we set the defines for the kernel.
        self.Define = None
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
        # Holds includes for the kernel
        self.Includes = list()
        # Holds the ast for a function that returns the kernelstring
        self.KernelStringStream = list()
        # Holds a list of which loops we will unroll
        self.UnrollLoops = list()
        # True is SetDefine were called.
        self.DefinesAreMade = False
        # List of what kernel arguments changes
        self.Change = list()

        self.IfThenElse = None


    def initNewRepr(self, ast, dev = 'GPU'):

        perfectForLoop = PerfectForLoop()
        perfectForLoop.visit(ast)
        
        if self.ParDim is None:
            self.ParDim = perfectForLoop.depth

        if self.ParDim == 1:
            self.Local['size'] = ['256']
            if dev == 'CPU':
                self.Local['size'] = ['16']
        else:
            self.Local['size'] = ['16','16']
            if dev == 'CPU':
                self.Local['size'] = ['4','4']
        
            
        innerbody = perfectForLoop.inner
        if perfectForLoop.depth == 2 and self.ParDim == 1:
            innerbody = perfectForLoop.outer
        firstLoop = ForLoops()
        
        firstLoop.visit(innerbody.compound)
        loopIndices = LoopIndices()
        if firstLoop.ast is not None:
            loopIndices.visit(innerbody.compound)
            self.Loops = loopIndices.Loops        
            self.InsideKernel = firstLoop.ast


        arrays = Arrays(self.astrepr.index)
        
        arrays.visit(innerbody.compound)

        self.NumDims = arrays.numSubscripts
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
        if self.ParDim == 2:
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
        
        self.RemovedIds = set(self.astrepr.UpperLimit[i] for i in self.GridIndices)

        idsStillInKernel = Ids()
        idsStillInKernel.visit(self.Kernel)
        self.RemovedIds = self.RemovedIds - idsStillInKernel.ids

        otherIds = self.astrepr.ArrayIds.union(self.astrepr.NonArrayIds)
        findDeviceArgs = FindDeviceArgs(otherIds)
        findDeviceArgs.visit(ast)
        self.DevArgList = findDeviceArgs.arglist
        findFunction = FindFunction()
        findFunction.visit(ast)
        self.DevFuncTypeId = findFunction.typeid
        self.DevFuncId = self.DevFuncTypeId.name.name

        for n in self.astrepr.ArrayIds:
            self.DevId[n] = 'dev_ptr' + n
            self.HstId[n] = 'hst_ptr' + n
            self.Mem[n] = 'hst_ptr' + n + '_mem_size'
            
        ## for n in self.DevArgList:
        ##     name = n.name.name
        ##     type = n.type[-2:]
        ##     self.Type[name] = type

            
        for n in self.ArrayIdToDimName:
            for m in self.ArrayIdToDimName[n]:
                self.Type[m] = ['size_t']
            
            
        kernelName = self.DevFuncTypeId.name.name
        
        self.KernelName = kernelName + 'Kernel'
        self.Worksize['local'] = kernelName + '_local_worksize'
        self.Worksize['global'] = kernelName + '_global_worksize'
        self.Worksize['offset'] = kernelName + '_global_offset'
        findReadWrite = FindReadWrite(self.astrepr.ArrayIds)
        findReadWrite.visit(ast)
        self.ReadWrite = findReadWrite.ReadWrite

        for n in self.ReadWrite:
            pset = self.ReadWrite[n]
            if len(pset) == 1:
                if 'write' in pset:
                    self.WriteOnly.append(n)
                else:
                    self.ReadOnly.append(n)

        argIds = self.astrepr.NonArrayIds.union(self.astrepr.ArrayIds) - self.RemovedIds

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
        self.Define = GroupCompound([Comment('// Defines for the kernel')])

        arrays = Arrays(self.astrepr.index)
        arrays.visit(ast)
        self.Subscript = arrays.Subscript
        self.SubIdx = arrays.SubIdx
        self.SubscriptNoId = copy.deepcopy(self.Subscript)
        for n in self.SubscriptNoId.values():
            for m in n:
                for i,k in enumerate(m):
                    try:
                        m[i] = k.name
                    except AttributeError:
                        try:
                            m[i] = k.value
                        except AttributeError:
                            m[i] = 'unknown'


        refToLoop = RefToLoop(self.GridIndices)
        refToLoop.visit(ast)
        self.RefToLoop = refToLoop.RefToLoop
        
    def DataStructures(self):
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
        print "self.ReadOnly " , self.ReadOnly
        print "self.Subscript " , self.Subscript
        print "self.SubscriptNoId " , self.SubscriptNoId
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
        print "self.RefToLoop ", self.RefToLoop


