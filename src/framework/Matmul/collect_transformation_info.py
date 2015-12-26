import collect_device as cd
import collect_gen as cg
import collect_array as ca
import collect_id as ci
import collect_loop as cl


def print_dict_sorted(mydict):
    keys = sorted(mydict)

    entries = ""
    for key in keys:
        value = mydict[key]
        entries += "'" + key + "': " + value.__repr__() + ","

    return "{" + entries[:-1] + "}"


class FindPerfectForLoop(object):
    def __init__(self):
        self.perfect_for_loop = cl.FindPerfectForLoop()
        self.ParDim = None

    def collect(self, ast):
        self.perfect_for_loop.visit(ast)

    @property
    def par_dim(self):
        if self.ParDim is None:
            return self.perfect_for_loop.depth
        else:
            return self.ParDim


class FindLocal(FindPerfectForLoop):
    def __init__(self):
        super(FindLocal, self).__init__()
        self.Local = dict()
        self.Local['name'] = 'LSIZE'
        self.Local['size'] = ['64']

    def collect(self, ast, dev='CPU'):
        super(FindLocal, self).collect(ast)
        if self.par_dim == 1:
            self.Local['size'] = ['256']
            if dev == 'CPU':
                self.Local['size'] = ['16']
        else:
            self.Local['size'] = ['16', '16']
            if dev == 'CPU':
                self.Local['size'] = ['4', '4']


class FindGridIndices(FindPerfectForLoop):
    def __init__(self):
        super(FindGridIndices, self).__init__()
        self.GridIndices = list()
        self.Kernel = None
        self.IdxToDim = dict()
        self.RefToLoop = dict()

    def collect(self, ast):
        super(FindGridIndices, self).collect(ast)

        fker = cd.FindKernel(self.par_dim)
        fker.visit(ast)
        self.Kernel = fker.kernel

        col_li = cl.LoopIndices(self.par_dim)
        col_li.visit(ast)

        self.GridIndices = col_li.grid_indices

        gi_to_dim = cg.GenIdxToDim()
        gi_to_dim.collect(ast, self.par_dim)
        self.IdxToDim = gi_to_dim.IdxToDim

        find_ref_to_loop_index = ca.FindRefToLoopIndex(self.par_dim)
        find_ref_to_loop_index.collect(ast)
        self.RefToLoop = find_ref_to_loop_index.RefToLoop


class FindLoops(FindPerfectForLoop):
    def __init__(self):
        super(FindLoops, self).__init__()
        self.ArrayIdToDimName = dict()
        self.Loops = dict()
        self.col_loop_limit = cl.LoopLimit()
        self.num_array_dims = dict()

    def collect(self, ast):
        super(FindLoops, self).collect(ast)

        find_inner_loops = cl.FindInnerLoops(self.par_dim)
        find_inner_loops.collect(ast)
        self.Loops = find_inner_loops.Loops

        self.col_loop_limit = cl.LoopLimit()
        self.col_loop_limit.visit(ast)

        num_array_dim = ca.NumArrayDim(ast)
        num_array_dim.visit(ast)

        self.num_array_dims = num_array_dim.numSubscripts

        gen_array_dim_names = cg.GenArrayDimNames()
        gen_array_dim_names.collect(ast)
        self.ArrayIdToDimName = gen_array_dim_names.ArrayIdToDimName

    @property
    def upper_limit(self):
        return self.col_loop_limit.upper_limit

    @property
    def lower_limit(self):
        return self.col_loop_limit.lower_limit


class FindSubscripts(FindLoops):
    def __init__(self):
        super(FindSubscripts, self).__init__()
        self.Subscript = dict()
        self.SubscriptNoId = dict()

    def collect(self, ast):
        super(FindSubscripts, self).collect(ast)

        arr_subs = ca.ArraySubscripts()
        arr_subs.collect(ast)
        self.Subscript = arr_subs.Subscript
        self.SubscriptNoId = arr_subs.subscript_no_id


class RemovedLoopLimit(FindLoops):
    def __init__(self):
        super(RemovedLoopLimit, self).__init__()
        self.RemovedIds = set()
        self.ParDim = None

    def collect(self, ast):
        super(RemovedLoopLimit, self).collect(ast)
        fgi = FindGridIndices()
        fgi.ParDim = self.ParDim
        fgi.collect(ast)

        find_removed_ids = cg.GenRemovedIds()
        find_removed_ids.collect(ast, self.par_dim)
        self.RemovedIds = find_removed_ids.removed_ids


class FindArrayIds(RemovedLoopLimit):
    def __init__(self):
        super(FindArrayIds, self).__init__()
        self.ArrayIds = set()
        self.NonArrayIds = set()
        self.type = set()
        self.kernel_args = dict()

    def collect(self, ast):
        super(FindArrayIds, self).collect(ast)

        arrays_ids = ca.GlobalArrayIds()
        arrays_ids.visit(ast)
        self.ArrayIds = arrays_ids.ids
        # print self.ArrayIds

        nonarray_ids = ci.GlobalNonArrayIds()
        nonarray_ids.visit(ast)
        self.NonArrayIds = nonarray_ids.ids

        mytype_ids = ci.GlobalTypeIds()
        mytype_ids.visit(ast)
        # print print_dict_sorted(mytype_ids.dictIds)
        self.type = mytype_ids.types

        gen_kernel_args = cg.GenKernelArgs()
        gen_kernel_args.collect(ast, par_dim=self.par_dim)
        self.kernel_args = gen_kernel_args.kernel_args


class FindReadWrite(FindArrayIds):
    def __init__(self):
        super(FindReadWrite, self).__init__()
        self.ReadWrite = dict()
        self.WriteOnly = list()
        self.ReadOnly = list()

    def collect(self, ast):
        super(FindReadWrite, self).collect(ast)
        find_read_write = ca.FindReadWrite(self.ArrayIds)
        find_read_write.visit(ast)
        self.ReadWrite = find_read_write.ReadWrite

        for n in self.ReadWrite:
            pset = self.ReadWrite[n]
            if len(pset) == 1:
                if 'write' in pset:
                    self.WriteOnly.append(n)
                else:
                    self.ReadOnly.append(n)
