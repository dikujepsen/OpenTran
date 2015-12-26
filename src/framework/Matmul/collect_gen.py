import collect_transformation_info as cti
import collect


class GenReverseIdx(object):
    def __init__(self):
        self.ReverseIdx = dict()
        self.ReverseIdx[0] = 1
        self.ReverseIdx[1] = 0


class GenHostArrayData(object):
    def __init__(self):
        super(GenHostArrayData, self).__init__()
        self.HstId = dict()
        self.Mem = dict()

    def collect(self, ast):
        arrays_ids = collect.GlobalArrayIds()
        arrays_ids.visit(ast)

        for n in arrays_ids.ids:
            self.HstId[n] = 'hst_ptr' + n
            self.Mem[n] = 'hst_ptr' + n + '_mem_size'


class GenArrayDimNames(object):
    def __init__(self):
        self.num_array_dims = dict()
        self.ArrayIdToDimName = dict()

    def collect(self, ast):
        num_array_dim = collect.NumArrayDim(ast)
        num_array_dim.visit(ast)
        self.num_array_dims = num_array_dim.numSubscripts

        for array_name, num_dims in num_array_dim.numSubscripts.items():
            tmp = list()
            for i in xrange(num_dims):
                tmp.append('hst_ptr' + array_name + '_dim' + str(i + 1))

            self.ArrayIdToDimName[array_name] = tmp


class GenIdxToDim(object):
    def __init__(self):
        self.IdxToDim = dict()

    def collect(self, ast, par_dim=2):
        col_li = collect.LoopIndices(par_dim)
        col_li.visit(ast)

        grid_indices = col_li.grid_indices
        for i, n in enumerate(reversed(grid_indices)):
            self.IdxToDim[i] = n


class GenKernelArgs(object):
    def __init__(self):
        self.kernel_args = dict()

    def collect(self, ast, par_dim=2):
        arrays_ids = collect.GlobalArrayIds()
        arrays_ids.visit(ast)
        array_ids = arrays_ids.ids
        # print self.ArrayIds

        nonarray_ids = collect.GlobalNonArrayIds()
        nonarray_ids.visit(ast)
        non_array_ids = nonarray_ids.ids

        mytype_ids = collect.GlobalTypeIds()
        mytype_ids.visit(ast)
        types = mytype_ids.types

        gen_removed_ids = GenRemovedIds()
        gen_removed_ids.collect(ast, par_dim)
        removed_ids = gen_removed_ids.removed_ids
        arg_ids = non_array_ids.union(array_ids) - removed_ids

        gen_array_dimnames = GenArrayDimNames()
        gen_array_dimnames.collect(ast)
        num_array_dims = gen_array_dimnames.num_array_dims
        arrayid_to_dimname = gen_array_dimnames.ArrayIdToDimName

        for n in arg_ids:
            tmplist = [n]
            try:
                if num_array_dims[n] == 2:
                    tmplist.append(arrayid_to_dimname[n][0])
            except KeyError:
                pass
            for m in tmplist:
                self.kernel_args[m] = types[m]


class GenRemovedIds(object):
    def __init__(self):
        self.removed_ids = set()

    def collect(self, ast, par_dim=2):
        col_li = collect.LoopIndices(par_dim)
        col_li.visit(ast)

        grid_indices = col_li.grid_indices

        col_loop_limit = collect.LoopLimit()
        col_loop_limit.visit(ast)
        upper_limit = col_loop_limit.upper_limit

        upper_limits = set(upper_limit[i] for i in grid_indices)

        find_kernel = collect.FindKernel(par_dim)
        find_kernel.visit(ast)
        ids_still_in_kernel = collect.Ids()
        ids_still_in_kernel.visit(find_kernel.kernel)
        self.removed_ids = upper_limits - ids_still_in_kernel.ids


class GenLocalArrayIdx(object):
    def __init__(self):
        self.IndexToLocalVar = dict()

    def collect(self, ast, par_dim=2):
        col_li = collect.LoopIndices(par_dim)
        col_li.visit(ast)
        grid_indices = col_li.grid_indices

        for var in grid_indices:
            self.IndexToLocalVar[var] = 'l' + var


class GenIdxToThreadId(object):
    def __init__(self):
        self.IndexToThreadId = dict()

    def collect(self, ast, par_dim=2):
        col_li = collect.LoopIndices(par_dim)
        col_li.visit(ast)

        grid_indices = col_li.grid_indices
        for i, n in enumerate(reversed(grid_indices)):
            self.IndexToThreadId[n] = 'get_global_id(' + str(i) + ')'
