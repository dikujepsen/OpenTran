import copy

from processing import collect_array as ca
from processing import collect_device as cd
from processing import collect_gen as cg
from processing import collect_id as ci
from Matmul import exchange

from Matmul import ast_buildingblock as ast_bb
import lan
from processing import collect_loop as cl


class Stencil(object):
    def __init__(self, ast):
        self.ast = ast

    def stencil(self, arr_names, west=0, north=0, east=0, south=0, middle=1):

        direction = [west, north, east, south, middle]
        dirname = [(0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)]
        loadings = [elem for i, elem in enumerate(dirname)
                    if direction[i] == 1]
        if not loadings:
            loadings = [(0, 0)]

        # finding the correct local memory size
        arr_name = arr_names[0]

        local = cl.get_local(self.ast)
        num_array_dims = ca.get_num_array_dims(self.ast)
        par_dim = cl.get_par_dim(self.ast)




        local_dims = [int(local['size'][0]) for _ in xrange(num_array_dims[arr_name])]
        if par_dim == 1 and len(local_dims) == 2:
            local_dims[0] = 1
        index_in_subscript = ca.get_indices_in_array_ref(self.ast)
        arr_idx = index_in_subscript[arr_name]

        (lower_limit, _) = cl.get_loop_limits(self.ast)
        local_offset = [int(lower_limit[i]) for i in arr_idx]

        for (x, y) in loadings:
            local_dims[0] += abs(x)
            if num_array_dims[arr_name] == 2:
                local_dims[1] += abs(y)

        stats = []
        types = ci.get_types(self.ast)
        for arr_name in arr_names:
            local_name = arr_name + '_local'
            array_init = lan.Constant(local_dims[0])
            if len(local_dims) == 2:
                array_init = [lan.BinOp(lan.Constant(local_dims[0]), '*', lan.Constant(local_dims[1]))]

            local_array_type_id = lan.ArrayTypeId(['__local'] + [types[arr_name][0]], lan.Id(local_name),
                                                  array_init)

            self.ast.ext.append(lan.Stencil(lan.Id(arr_name), lan.Id(local_name),
                                            [local['size'][0], local['size'][0]]))
            stats.append(local_array_type_id)

        init_comp = lan.GroupCompound(stats)
        stats2 = []
        load_comp = lan.GroupCompound(stats2)
        reverse_idx = cg.get_reverse_idx(self.ast)
        grid_indices = cl.get_grid_indices(self.ast)
        # Insert local id with offset
        for i, offset in enumerate(local_offset):
            idd = reverse_idx[i] if len(local_offset) == 2 else i

            get_local_func_decl = ast_bb.FuncCall('get_local_id', [lan.Constant(idd)])

            if offset != 0:

                rval = lan.BinOp(get_local_func_decl, '+', lan.Constant(offset))
            else:
                rval = lan.Id(get_local_func_decl)

            lval = lan.TypeId(['unsigned'], lan.Id('l' + grid_indices[i]))
            stats.append(lan.Assignment(lval, rval))

        local_swap = ci.get_local_swap(self.ast)

        index_to_local_var = cg.get_local_array_idx(self.ast)
        exchange_indices = exchange.ExchangeIndices(index_to_local_var, local_swap.values())

        loop_arrays = ca.get_loop_arrays(self.ast)
        # Creating the loading of values into the local array.
        for arr_name in arr_names:
            for k, l in enumerate(loadings):
                array_id = lan.Id(arr_name)
                # get first ArrayRef
                aref = loop_arrays[arr_name][k]
                subscript = aref.subscript
                lsub = copy.deepcopy(subscript)
                lval = lan.ArrayRef(lan.Id(local_swap[arr_name]), lsub)
                rsub = copy.deepcopy(subscript)
                rval = lan.ArrayRef(array_id, rsub, extra={'localMemory': True})
                load = lan.Assignment(lval, rval)
                exchange_id = exchange.ExchangeId(index_to_local_var)
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

        kernel = cd.get_kernel(self.ast)
        kernel.statements.insert(0, load_comp)
        kernel.statements.insert(0, init_comp)
