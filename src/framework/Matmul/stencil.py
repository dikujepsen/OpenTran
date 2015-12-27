import copy
import ast_buildingblock as ast_bb
import lan
import collect_device as cd
import collect_transformation_info as cti
import exchange
import collect_gen as cg
import collect_array as ca
import collect_id as ci
import collect_loop as cl


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

        find_kernel = cd.FindKernel(self.ParDim)
        find_kernel.visit(ast)
        self.Kernel = find_kernel.kernel

        gen_local_array_idx = cg.GenLocalArrayIdx()
        gen_local_array_idx.collect(ast, self.ParDim)
        self.IndexToLocalVar = gen_local_array_idx.IndexToLocalVar

        col_li = cl.LoopIndices(self.ParDim)
        col_li.visit(ast)
        grid_indices = col_li.grid_indices
        self.GridIndices = grid_indices

        loop_limit = cl.LoopLimit()
        loop_limit.visit(ast)
        self.LowerLimit = loop_limit.lower_limit

        num_array_dim = ca.NumArrayDim(ast)
        num_array_dim.visit(ast)
        self.num_array_dims = num_array_dim.numSubscripts

        indices_in_array_ref = ca.IndicesInArrayRef()
        indices_in_array_ref.collect(ast, self.ParDim)
        self.IndexInSubscript = indices_in_array_ref.indexIds

        array_name_to_ref = ca.ArrayNameToRef()
        array_name_to_ref.visit(ast)
        self.LoopArrays = array_name_to_ref.LoopArrays

        find_local = cti.FindLocal()
        find_local.collect(ast, dev)
        self.Local = find_local.Local

        mytype_ids = ci.GlobalTypeIds()
        mytype_ids.visit(ast)
        self.type = mytype_ids.types

        array_dim_names = cg.GenArrayDimNames()
        array_dim_names.collect(ast)

        self.ArrayIdToDimName = array_dim_names.ArrayIdToDimName

        reverse_idx = cg.GenReverseIdx()
        self.ReverseIdx = reverse_idx.ReverseIdx

    def stencil(self, arr_names, west=0, north=0, east=0, south=0, middle=1):

        direction = [west, north, east, south, middle]
        dirname = [(0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)]
        loadings = [elem for i, elem in enumerate(dirname)
                    if direction[i] == 1]
        if not loadings:
            loadings = [(0, 0)]

        # finding the correct local memory size
        arr_name = arr_names[0]
        local_dims = [int(self.Local['size'][0]) for _ in xrange(self.num_array_dims[arr_name])]
        if self.ParDim == 1 and len(local_dims) == 2:
            local_dims[0] = 1
        arr_idx = self.IndexInSubscript[arr_name]
        local_offset = [int(self.LowerLimit[i]) for i in arr_idx]

        for (x, y) in loadings:
            local_dims[0] += abs(x)
            if self.num_array_dims[arr_name] == 2:
                local_dims[1] += abs(y)

        stats = []
        for arr_name in arr_names:
            local_name = arr_name + '_local'
            arrayinit = '['
            for i, d in enumerate(local_dims):
                arrayinit += str(d)
                if i == 0 and len(local_dims) == 2:
                    arrayinit += '*'
            arrayinit += ']'
            array_init = lan.Constant(local_dims[0])
            if len(local_dims) == 2:
                array_init = [lan.BinOp(lan.Constant(local_dims[0]), '*', lan.Constant(local_dims[1]))]

            local_array_type_id = lan.ArrayTypeId(['__local'] + [self.type[arr_name][0]], lan.Id(local_name),
                                                  array_init)
            self.num_array_dims[local_name] = self.num_array_dims[arr_name]
            self.LocalSwap[arr_name] = local_name
            self.ArrayIdToDimName[local_name] = [self.Local['size'][0], self.Local['size'][0]]
            stats.append(local_array_type_id)

        init_comp = lan.GroupCompound(stats)
        stats2 = []
        load_comp = lan.GroupCompound(stats2)

        # Insert local id with offset
        for i, offset in enumerate(local_offset):
            idd = self.ReverseIdx[i] if len(local_offset) == 2 else i

            get_local_func_decl = ast_bb.FuncCall('get_local_id', [lan.Constant(idd)])

            if offset != 0:

                rval = lan.BinOp(get_local_func_decl, '+', lan.Constant(offset))
            else:
                rval = lan.Id(get_local_func_decl)

            lval = lan.TypeId(['unsigned'], lan.Id('l' + self.GridIndices[i]))
            stats.append(lan.Assignment(lval, rval))

        exchange_indices = exchange.ExchangeIndices(self.IndexToLocalVar, self.LocalSwap.values())

        # Creating the loading of values into the local array.
        for arr_name in arr_names:
            for k, l in enumerate(loadings):
                array_id = lan.Id(arr_name)
                # get first ArrayRef
                aref = self.LoopArrays[arr_name][k]
                subscript = aref.subscript
                lsub = copy.deepcopy(subscript)
                lval = lan.ArrayRef(lan.Id(self.LocalSwap[arr_name]), lsub)
                rsub = copy.deepcopy(subscript)
                rval = lan.ArrayRef(array_id, rsub, extra={'localMemory': True})
                load = lan.Assignment(lval, rval)
                exchange_id = exchange.ExchangeId(self.IndexToLocalVar)
                orisub = subscript
                for m in orisub:
                    exchange_id.visit(m)

                stats2.append(load)

        # Must also create the barrier
        arglist = lan.ArgList([lan.Id('CLK_LOCAL_MEM_FENCE')])

        func = ast_bb.EmptyFuncDecl('barrier', type=[])
        func.arglist = arglist
        stats2.append(func)

        exchange_indices.visit(init_comp)
        exchange_indices.visit(load_comp)

        self.Kernel.statements.insert(0, load_comp)
        self.Kernel.statements.insert(0, init_comp)
