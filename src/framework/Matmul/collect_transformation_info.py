import transf_visitor as tvisitor
import visitor
import copy
import collect
import lan


def print_dict_sorted(mydict):
    keys = sorted(mydict)

    entries = ""
    for key in keys:
        value = mydict[key]
        entries += "'" + key + "': " + value.__repr__() + ","

    return "{" + entries[:-1] + "}"


class FindPerfectForLoop(object):
    def __init__(self):
        self.perfect_for_loop = collect.FindPerfectForLoop()
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

        fker = collect.FindKernel(self.par_dim)
        fker.visit(ast)
        self.Kernel = fker.kernel

        col_li = collect.LoopIndices(self.par_dim)
        col_li.visit(ast)

        self.GridIndices = col_li.grid_indices

        gi_to_dim = collect.GenIdxToDim()
        gi_to_dim.collect(ast, self.par_dim)
        self.IdxToDim = gi_to_dim.IdxToDim

        find_ref_to_loop_index = collect.FindRefToLoopIndex(self.par_dim)
        find_ref_to_loop_index.collect(ast)
        self.RefToLoop = find_ref_to_loop_index.RefToLoop


class FindLoops(FindPerfectForLoop):
    def __init__(self):
        super(FindLoops, self).__init__()
        self.loop_indices = visitor.LoopIndices()
        self.forLoops = visitor.ForLoops()
        self.ArrayIdToDimName = dict()
        self.Loops = dict()
        self.col_loop_limit = collect.LoopLimit()
        self.num_array_dims = dict()

    def collect(self, ast):
        super(FindLoops, self).collect(ast)
        innerbody = self.perfect_for_loop.inner
        if self.perfect_for_loop.depth == 2 and self.ParDim == 1:
            innerbody = self.perfect_for_loop.outer
        first_loop = tvisitor.ForLoops()

        first_loop.visit(innerbody.compound)
        loop_indices = tvisitor.LoopIndices()
        if first_loop.ast is not None:
            loop_indices.visit(innerbody.compound)
            self.Loops = loop_indices.Loops

        self.forLoops.visit(ast)

        self.col_loop_limit = collect.LoopLimit()
        self.col_loop_limit.visit(ast)

        num_array_dim = collect.NumArrayDim(ast)
        num_array_dim.visit(ast)

        self.num_array_dims = num_array_dim.numSubscripts

        gen_array_dim_names = collect.GenArrayDimNames()
        gen_array_dim_names.collect(ast)
        self.ArrayIdToDimName = gen_array_dim_names.ArrayIdToDimName
        # print print_dict_sorted(gen_array_dim_names.ArrayIdToDimName)

        # print self.ArrayIdToDimName

    @property
    def upper_limit(self):
        return self.col_loop_limit.upper_limit

    @property
    def lower_limit(self):
        return self.col_loop_limit.lower_limit

    @property
    def for_loop_ast(self):
        return self.forLoops.ast


class FindSubscripts(FindLoops):
    def __init__(self):
        super(FindSubscripts, self).__init__()
        self.Subscript = dict()
        self.SubscriptNoId = dict()

    def collect(self, ast):
        super(FindSubscripts, self).collect(ast)

        arr_subs = collect.ArraySubscripts()
        arr_subs.visit(ast)
        self.Subscript = arr_subs.Subscript

        self.SubscriptNoId = copy.deepcopy(self.Subscript)
        for n in self.SubscriptNoId.values():
            for m in n:
                for i, k in enumerate(m):
                    if isinstance(k, lan.Id):
                        m[i] = k.name
                    else:
                        m[i] = 'unknown'


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

        upper_limits = set(self.upper_limit[i] for i in fgi.GridIndices)
        ids_still_in_kernel = tvisitor.Ids()
        ids_still_in_kernel.visit(fgi.Kernel)
        self.RemovedIds = upper_limits - ids_still_in_kernel.ids


class FindArrayIds(RemovedLoopLimit):
    def __init__(self):
        super(FindArrayIds, self).__init__()
        self.ArrayIds = set()
        self.NonArrayIds = set()
        self.type = set()
        self.kernel_args = dict()

    def collect(self, ast):
        super(FindArrayIds, self).collect(ast)

        arrays_ids = collect.GlobalArrayIds()
        arrays_ids.visit(ast)
        self.ArrayIds = arrays_ids.ids
        # print self.ArrayIds

        nonarray_ids = collect.GlobalNonArrayIds()
        nonarray_ids.visit(ast)
        self.NonArrayIds = nonarray_ids.ids

        mytype_ids = collect.GlobalTypeIds()
        mytype_ids.visit(ast)
        # print print_dict_sorted(mytype_ids.dictIds)
        self.type = mytype_ids.dictIds

        arg_ids = self.NonArrayIds.union(self.ArrayIds) - self.RemovedIds
        # print self.ArrayIdToDimName
        # print arg_ids, "qwe123"
        # print self.ArrayIdToDimName, "qwe123"
        # print arg_ids
        for n in arg_ids:
            tmplist = [n]
            try:
                if self.num_array_dims[n] == 2:
                    tmplist.append(self.ArrayIdToDimName[n][0])
            except KeyError:
                pass
            for m in tmplist:
                self.kernel_args[m] = self.type[m]


class GenHostArrayData(FindArrayIds):
    def __init__(self):
        super(GenHostArrayData, self).__init__()
        self.HstId = dict()
        self.Mem = dict()

    def generate(self):
        for n in self.ArrayIds:
            self.HstId[n] = 'hst_ptr' + n
            self.Mem[n] = 'hst_ptr' + n + '_mem_size'


class FindReadWrite(GenHostArrayData):
    def __init__(self):
        super(FindReadWrite, self).__init__()
        self.ReadWrite = dict()
        self.WriteOnly = list()
        self.ReadOnly = list()

    def collect(self, ast):
        super(FindReadWrite, self).collect(ast)
        self.generate()
        find_read_write = collect.FindReadWrite(self.ArrayIds)
        find_read_write.visit(ast)
        self.ReadWrite = find_read_write.ReadWrite

        for n in self.ReadWrite:
            pset = self.ReadWrite[n]
            if len(pset) == 1:
                if 'write' in pset:
                    self.WriteOnly.append(n)
                else:
                    self.ReadOnly.append(n)
