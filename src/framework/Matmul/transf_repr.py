from Matmul.transf_visitor import *


class TransfRepr(NodeVisitor):
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
        self.ReverseIdx = dict()
        self.IfThenElse = None

    def init_rew_repr(self, ast, dev='GPU'):

        perfect_for_loop = PerfectForLoop()
        perfect_for_loop.visit(ast)

        if self.ParDim is None:
            self.ParDim = perfect_for_loop.depth

        if self.ParDim == 1:
            self.Local['size'] = ['256']
            if dev == 'CPU':
                self.Local['size'] = ['16']
        else:
            self.Local['size'] = ['16', '16']
            if dev == 'CPU':
                self.Local['size'] = ['4', '4']

        innerbody = perfect_for_loop.inner
        if perfect_for_loop.depth == 2 and self.ParDim == 1:
            innerbody = perfect_for_loop.outer
        firstLoop = ForLoops()

        firstLoop.visit(innerbody.compound)
        loop_indices = LoopIndices()
        if firstLoop.ast is not None:
            loop_indices.visit(innerbody.compound)
            self.Loops = loop_indices.Loops
            self.InsideKernel = firstLoop.ast

        arrays = Arrays(self.astrepr.loop_index)
        arrays.visit(innerbody.compound)

        self.astrepr.num_array_dims = arrays.numSubscripts
        self.astrepr.LoopArrays = arrays.LoopArrays

        init_ids = InitIds()
        init_ids.visit(perfect_for_loop.ast.init)

        grid_ids = list()
        id_map = dict()
        local_map = dict()
        local_var_map = dict()
        first_idx = init_ids.index[0]
        id_map[first_idx] = 'get_global_id(0)'
        local_map[first_idx] = 'get_local_id(0)'
        local_var_map[first_idx] = 'l' + first_idx
        self.ReverseIdx = dict()
        self.ReverseIdx[0] = 1
        self.ReverseIdx[1] = 0
        grid_ids.extend(init_ids.index)
        kernel = perfect_for_loop.ast.compound
        if self.ParDim == 2:
            init_ids = InitIds()
            init_ids.visit(kernel.statements[0].init)
            kernel = kernel.statements[0].compound
            second_idx = init_ids.index[0]
            id_map[second_idx] = 'get_global_id(1)'
            local_map[second_idx] = 'get_local_id(1)'
            local_var_map[second_idx] = 'l' + second_idx
            grid_ids.extend(init_ids.index)
            (id_map[grid_ids[0]], id_map[grid_ids[1]]) = (id_map[grid_ids[1]], id_map[grid_ids[0]])
            (local_map[grid_ids[0]], local_map[grid_ids[1]]) = (local_map[grid_ids[1]], local_map[grid_ids[0]])

        self.IndexToLocalId = local_map
        self.IndexToLocalVar = local_var_map
        self.IndexToThreadId = id_map
        self.GridIndices = grid_ids
        self.Kernel = kernel
        for i, n in enumerate(reversed(self.GridIndices)):
            self.IdxToDim[i] = n

        find_dim = FindDim(self.astrepr.num_array_dims)
        find_dim.visit(ast)
        self.ArrayIdToDimName = find_dim.dimNames

        self.RemovedIds = set(self.astrepr.UpperLimit[i] for i in self.GridIndices)

        ids_still_in_kernel = Ids()
        ids_still_in_kernel.visit(self.Kernel)
        self.RemovedIds = self.RemovedIds - ids_still_in_kernel.ids

        other_ids = self.astrepr.ArrayIds.union(self.astrepr.NonArrayIds)
        find_device_args = FindDeviceArgs(other_ids)
        find_device_args.visit(ast)
        self.DevArgList = find_device_args.arglist
        find_function = FindFunction()
        find_function.visit(ast)
        self.DevFuncTypeId = find_function.typeid
        self.DevFuncId = self.DevFuncTypeId.name.name

        for n in self.astrepr.ArrayIds:
            self.DevId[n] = 'dev_ptr' + n
            self.HstId[n] = 'hst_ptr' + n
            self.Mem[n] = 'hst_ptr' + n + '_mem_size'

        for n in self.ArrayIdToDimName:
            for m in self.ArrayIdToDimName[n]:
                self.Type[m] = ['size_t']

        kernel_name = self.DevFuncTypeId.name.name

        self.KernelName = kernel_name + 'Kernel'
        self.Worksize['local'] = kernel_name + '_local_worksize'
        self.Worksize['global'] = kernel_name + '_global_worksize'
        self.Worksize['offset'] = kernel_name + '_global_offset'
        find_read_write = FindReadWrite(self.astrepr.ArrayIds)
        find_read_write.visit(ast)
        self.ReadWrite = find_read_write.ReadWrite

        for n in self.ReadWrite:
            pset = self.ReadWrite[n]
            if len(pset) == 1:
                if 'write' in pset:
                    self.WriteOnly.append(n)
                else:
                    self.ReadOnly.append(n)

        arg_ids = self.astrepr.NonArrayIds.union(self.astrepr.ArrayIds) - self.RemovedIds

        for n in arg_ids:
            tmplist = [n]
            try:
                if self.astrepr.num_array_dims[n] == 2:
                    tmplist.append(self.ArrayIdToDimName[n][0])
            except KeyError:
                pass
            for m in tmplist:
                self.KernelArgs[m] = self.Type[m]

        self.Transposition = GroupCompound([Comment('// Transposition')])
        self.ConstantMemory = GroupCompound([Comment('// Constant Memory')])
        self.Define = GroupCompound([Comment('// Defines for the kernel')])

        arrays = Arrays(self.astrepr.loop_index)
        arrays.visit(ast)
        self.Subscript = arrays.Subscript
        self.SubIdx = arrays.SubIdx
        self.SubscriptNoId = copy.deepcopy(self.Subscript)
        for n in self.SubscriptNoId.values():
            for m in n:
                for i, k in enumerate(m):
                    try:
                        m[i] = k.name
                    except AttributeError:
                        try:
                            m[i] = k.value
                        except AttributeError:
                            m[i] = 'unknown'

        ref_to_loop = RefToLoop(self.GridIndices)
        ref_to_loop.visit(ast)
        self.RefToLoop = ref_to_loop.RefToLoop

    def data_structures(self):
        print "self.RemovedIds ", self.RemovedIds
        print "self.IndexToThreadId ", self.IndexToThreadId
        print "self.IndexToLocalId ", self.IndexToLocalId
        print "self.IndexToLocalVar ", self.IndexToLocalVar
        print "self.GridIndices ", self.GridIndices
        print "self.ArrayIdToDimName ", self.ArrayIdToDimName
        print "self.DevArgList ", self.DevArgList
        print "self.DevFuncTypeId ", self.DevFuncTypeId
        print "self.DevId ", self.DevId
        print "self.HstId ", self.HstId
        print "self.Type ", self.Type
        print "self.Mem ", self.Mem
        print "self.ParDim ", self.ParDim
        print "self.Worksize ", self.Worksize
        print "self.IdxToDim ", self.IdxToDim
        print "self.WriteOnly ", self.WriteOnly
        print "self.ReadOnly ", self.ReadOnly
        print "self.Subscript ", self.Subscript
        print "self.SubscriptNoId ", self.SubscriptNoId
        print "TRANSFORMATIONS"
        print "self.Transposition ", self.Transposition
        print "self.ConstantMemory ", self.ConstantMemory
        print "self.KernelArgs ", self.KernelArgs
        print "self.NameSwap ", self.NameSwap
        print "self.LocalSwap ", self.LocalSwap
        print "self.Add ", self.Add
        print "self.GlobalVars ", self.GlobalVars
        print "self.ConstantMem ", self.ConstantMem
        print "self.Loops ", self.Loops
        print "self.RefToLoop ", self.RefToLoop
