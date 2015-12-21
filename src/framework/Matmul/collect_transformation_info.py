import transf_visitor as tvisitor
import visitor
import copy
import collect

class FindPerfectForLoop(object):
    def __init__(self):
        self.perfect_for_loop = tvisitor.PerfectForLoop()
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
        grid_ids = list()
        init_ids = tvisitor.InitIds()
        init_ids.visit(self.perfect_for_loop.ast.init)
        grid_ids.extend(init_ids.index)
        kernel = self.perfect_for_loop.ast.compound
        if self.par_dim == 2:
            init_ids = tvisitor.InitIds()
            init_ids.visit(kernel.statements[0].init)
            kernel = kernel.statements[0].compound
            grid_ids.extend(init_ids.index)

        self.GridIndices = grid_ids
        self.Kernel = kernel
        for i, n in enumerate(reversed(self.GridIndices)):
            self.IdxToDim[i] = n

        ref_to_loop = tvisitor.RefToLoop(self.GridIndices)
        ref_to_loop.visit(ast)
        self.RefToLoop = ref_to_loop.RefToLoop


class FindLoops(FindPerfectForLoop):
    def __init__(self):
        super(FindLoops, self).__init__()
        self.loop_indices = visitor.LoopIndices()
        self.arrays = None
        self.forLoops = visitor.ForLoops()
        self.ArrayIdToDimName = dict()
        self.Loops = dict()
        self.col_loop_limit = collect.LoopLimit()

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

        col_li = collect.LoopIndices()
        col_li.visit(ast)

        self.loop_indices.visit(self.forLoops.ast)
        self.arrays = visitor.Arrays(self.loop_indices.index)

        # self.arrays = visitor.Arrays(col_li.index)
        self.arrays.visit(ast)
        for n in self.arrays.numIndices:
            if self.arrays.numIndices[n] == 2:
                self.arrays.numSubscripts[n] = 2
            elif self.arrays.numIndices[n] > 2:
                self.arrays.numSubscripts[n] = 1


        find_dim = tvisitor.FindDim(self.arrays.numSubscripts)
        find_dim.visit(ast)
        self.ArrayIdToDimName = find_dim.dimNames
        # print self.ArrayIdToDimName

    @property
    def upper_limit(self):
        return self.col_loop_limit.upper_limit

    @property
    def lower_limit(self):
        return self.col_loop_limit.lower_limit

    @property
    def num_array_dims(self):
        return self.arrays.numSubscripts

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

        self.Subscript = self.arrays.Subscript

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

        self.RemovedIds = set(self.upper_limit[i] for i in fgi.GridIndices)
        ids_still_in_kernel = tvisitor.Ids()
        ids_still_in_kernel.visit(fgi.Kernel)
        self.RemovedIds = self.RemovedIds - ids_still_in_kernel.ids


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
        find_read_write = tvisitor.FindReadWrite(self.ArrayIds)
        find_read_write.visit(ast)
        self.ReadWrite = find_read_write.ReadWrite

        for n in self.ReadWrite:
            pset = self.ReadWrite[n]
            if len(pset) == 1:
                if 'write' in pset:
                    self.WriteOnly.append(n)
                else:
                    self.ReadOnly.append(n)
