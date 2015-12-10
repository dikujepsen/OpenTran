import transf_visitor as tvisitor
import visitor


class FindPerfectForLoop(object):
    def __init__(self):
        self.perfect_for_loop = tvisitor.PerfectForLoop()

    def collect(self, ast):
        self.perfect_for_loop.visit(ast)

    def par_dim(self):
        return self.perfect_for_loop.depth


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


    def loop_index(self):
        return self.loop_indices.index

    def upper_limit(self):
        return self.loop_indices.end

    def num_array_dims(self):
        return self.arrays.numSubscripts

    def for_loop_ast(self):
        return self.loops.ast


class RemovedLoopLimit(FindLoops):
    def __init__(self):
        super(RemovedLoopLimit, self).__init__()
        self.RemovedIds = set()


    def collect(self, ast):
        super(RemovedLoopLimit, self).collect(ast)
        fgi = FindGridIndices()
        fgi.collect(ast)

        self.RemovedIds = set(self.upper_limit() for i in fgi.GridIndices)
        ids_still_in_kernel = tvisitor.Ids()
        ids_still_in_kernel.visit(fgi.Kernel)
        self.RemovedIds = self.RemovedIds - ids_still_in_kernel.ids



class FindArrayIds(FindLoops):
    def __init__(self):
        super(FindLoops, self).__init__()
        self.ArrayIds = set()
        self.NonArrayIds = set()
        self.type = set()
    
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
