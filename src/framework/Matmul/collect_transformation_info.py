import transf_visitor as tvisitor
import visitor


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


class FindGridIndices(FindPerfectForLoop):
    def __init__(self):
        super(FindGridIndices, self).__init__()
        self.GridIndices = list()
        self.Kernel = None

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

class FindLoops(object):
    def __init__(self):
        self.loop_indices = visitor.LoopIndices()
        self.arrays = None
        self.loops = visitor.ForLoops()
        self.ArrayIdToDimName = dict()

    def collect(self, ast):
        self.loops.visit(ast)
        self.loop_indices.visit(self.loops.ast)
        self.arrays = visitor.Arrays(self.loop_indices.index)
        self.arrays.visit(ast)
        for n in self.arrays.numIndices:
            if self.arrays.numIndices[n] == 2:
                self.arrays.numSubscripts[n] = 2
            elif self.arrays.numIndices[n] > 2:
                self.arrays.numSubscripts[n] = 1

        find_dim = tvisitor.FindDim(self.arrays.numSubscripts)
        find_dim.visit(ast)
        self.ArrayIdToDimName = find_dim.dimNames

    @property
    def upper_limit(self):
        return self.loop_indices.end

    @property
    def num_array_dims(self):
        return self.arrays.numSubscripts

    @property
    def for_loop_ast(self):
        return self.loops.ast


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
        type_ids = visitor.TypeIds()
        type_ids.visit(self.for_loop_ast)

        ids = visitor.Ids2()
        ids.visit(ast)

        # print ast
        #
        # print ids.ids, "123qwe"
        # print arrays.ids
        # print type_ids.ids
        other_ids = ids.ids - self.arrays.ids - type_ids.ids
        self.ArrayIds = self.arrays.ids - type_ids.ids
        self.NonArrayIds = other_ids

        type_ids2 = visitor.TypeIds()
        type_ids2.visit(ast)
        for n in type_ids.ids:
            type_ids2.dictIds.pop(n)

        self.type = type_ids2.dictIds

        arg_ids = self.NonArrayIds.union(self.ArrayIds) - self.RemovedIds

        # print arg_ids, "qwe123"
        # print self.ArrayIdToDimName, "qwe123"

        for n in arg_ids:
            tmplist = [n]
            try:
                if self.num_array_dims[n] == 2:
                    tmplist.append(self.ArrayIdToDimName[n][0])
            except KeyError:
                pass
            for m in tmplist:
                self.kernel_args[m] = self.type[m]
