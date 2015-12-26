import copy
import ast_buildingblock as ast_bb
import lan
import collect
import collect_transformation_info as cti
import exchange

class Stencil(object):
    def __init__(self):
        self.Kernel = None
        self.Local = dict()
        self.Local['name'] = 'LSIZE'
        self.Local['size'] = ['64']
        self.num_array_dims = dict()
        self.IndexInSubscript = dict()
        self.LowerLimit = dict()
        self.type = dict()
        self.ArrayIdToDimName = dict()
        self.ReverseIdx = dict()
        self.LoopArrays = dict()
        self.IndexToLocalVar = dict()
        self.GridIndices = list()
        self.ParDim = None
        self.Add = dict()

        self.num_array_dims = dict()
        self.LocalSwap = dict()
        self.ArrayIdToDimName = dict()
        self.Kernel = None
        self.LoopArrays = dict()

    def set_datastructures(self, ast, dev='CPU'):
        perfect_for_loop = cti.FindPerfectForLoop()
        perfect_for_loop.collect(ast)

        if self.ParDim is None:
            self.ParDim = perfect_for_loop.par_dim

        find_kernel = collect.FindKernel(self.ParDim)
        find_kernel.visit(ast)
        self.Kernel = find_kernel.kernel

        gen_local_array_idx = collect.GenLocalArrayIdx()
        gen_local_array_idx.collect(ast, self.ParDim)
        self.IndexToLocalVar = gen_local_array_idx.IndexToLocalVar

        col_li = collect.LoopIndices(self.ParDim)
        col_li.visit(ast)
        grid_indices = col_li.grid_indices
        self.GridIndices = grid_indices

        loop_limit = collect.LoopLimit()
        loop_limit.visit(ast)
        self.LowerLimit = loop_limit.lower_limit

        num_array_dim = collect.NumArrayDim(ast)
        num_array_dim.visit(ast)
        self.num_array_dims = num_array_dim.numSubscripts

        indices_in_array_ref = collect.IndicesInArrayRef()
        indices_in_array_ref.collect(ast, self.ParDim)
        self.IndexInSubscript = indices_in_array_ref.indexIds

        array_name_to_ref = collect.ArrayNameToRef()
        array_name_to_ref.visit(ast)
        self.LoopArrays = array_name_to_ref.LoopArrays

        find_local = cti.FindLocal()
        find_local.collect(ast, dev)
        self.Local = find_local.Local

        mytype_ids = collect.GlobalTypeIds()
        mytype_ids.visit(ast)
        self.type = mytype_ids.types

        array_dim_names = collect.GenArrayDimNames()
        array_dim_names.collect(ast)

        self.ArrayIdToDimName = array_dim_names.ArrayIdToDimName
        self.ReverseIdx = dict()
        self.ReverseIdx[0] = 1
        self.ReverseIdx[1] = 0

    def stencil(self, arrNames, west=0, north=0, east=0, south=0, middle=1):

        direction = [west, north, east, south, middle]
        dirname = [(0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)]
        loadings = [elem for i, elem in enumerate(dirname)
                    if direction[i] == 1]
        if not loadings:
            loadings = [(0, 0)]

        ## finding the correct local memory size
        arrName = arrNames[0]
        localDims = [int(self.Local['size'][0]) \
                     for i in xrange(self.num_array_dims[arrName])]
        if self.ParDim == 1 and len(localDims) == 2:
            localDims[0] = 1;
        arrIdx = self.IndexInSubscript[arrName]
        localOffset = [int(self.LowerLimit[i]) \
                       for i in arrIdx]

        for (x, y) in loadings:
            localDims[0] += abs(x)
            if self.num_array_dims[arrName] == 2:
                localDims[1] += abs(y)

        stats = []
        for arrName in arrNames:
            localName = arrName + '_local'
            arrayinit = '['
            for i, d in enumerate(localDims):
                arrayinit += str(d)
                if i == 0 and len(localDims) == 2:
                    arrayinit += '*'
            arrayinit += ']'

            localId = lan.Id(localName + arrayinit)
            localTypeId = lan.TypeId(['__local'] + [self.type[arrName][0]], localId)
            self.num_array_dims[localName] = self.num_array_dims[arrName]
            self.LocalSwap[arrName] = localName
            self.ArrayIdToDimName[localName] = [self.Local['size'][0], self.Local['size'][0]]
            stats.append(localTypeId)

        InitComp = lan.GroupCompound(stats)
        stats2 = []
        LoadComp = lan.GroupCompound(stats2)

        ## Insert local id with offset
        for i, offset in enumerate(localOffset):
            idd = self.ReverseIdx[i] if len(localOffset) == 2 else i
            if offset != 0:

                rval = lan.BinOp(lan.Id('get_local_id(' + str(idd) + ')'), '+', \
                                 lan.Constant(offset))
            else:
                rval = lan.Id('get_local_id(' + str(idd) + ')')
            lval = lan.TypeId(['unsigned'], lan.Id('l' + self.GridIndices[i]))
            stats.append(lan.Assignment(lval, rval))

        exchangeIndices = exchange.ExchangeIndices(self.IndexToLocalVar, self.LocalSwap.values())

        ## Creating the loading of values into the local array.
        for arrName in arrNames:
            for k, l in enumerate(loadings):
                arrayId = lan.Id(arrName)
                # get first ArrayRef
                aref = self.LoopArrays[arrName][k]
                subscript = aref.subscript
                lsub = copy.deepcopy(subscript)
                lval = lan.ArrayRef(lan.Id(self.LocalSwap[arrName]), lsub)
                rsub = copy.deepcopy(subscript)
                rval = lan.ArrayRef(arrayId, rsub, extra={'localMemory': True})
                load = lan.Assignment(lval, rval)
                exchangeId = exchange.ExchangeId(self.IndexToLocalVar)
                orisub = subscript
                for m in orisub:
                    exchangeId.visit(m)

                stats2.append(load)

        # Must also create the barrier
        arglist = lan.ArgList([lan.Id('CLK_LOCAL_MEM_FENCE')])
        func = ast_bb.EmptyFuncDecl('barrier', type=[])
        func.arglist = arglist
        stats2.append(func)

        exchangeIndices.visit(InitComp)
        exchangeIndices.visit(LoadComp)

        self.Kernel.statements.insert(0, LoadComp)
        self.Kernel.statements.insert(0, InitComp)
