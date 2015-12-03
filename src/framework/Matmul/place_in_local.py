import lan
import copy
import visitor
import transf_visitor as tvisitor


class PlaceInLocal(object):
    def __init__(self):
        self.SubscriptNoId = dict()
        self.GridIndices = list()
        self.ParDim = None  # int
        self.Loops = dict()
        self.PlaceInLocalArgs = list()
        self.PlaceInLocalCond = None
        self.UpperLimit = dict()
        self.LowerLimit = dict()
        self.Local = dict()
        self.Local['name'] = 'LSIZE'
        self.Local['size'] = ['64']

    def set_datastructures(self, ast, dev='CPU'):
        perfect_for_loop = tvisitor.PerfectForLoop()
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

        grid_ids = list()
        init_ids = tvisitor.InitIds()
        init_ids.visit(perfect_for_loop.ast.init)
        grid_ids.extend(init_ids.index)
        kernel = perfect_for_loop.ast.compound
        if self.ParDim == 2:
            init_ids = tvisitor.InitIds()
            init_ids.visit(kernel.statements[0].init)
            kernel = kernel.statements[0].compound
            grid_ids.extend(init_ids.index)

        self.GridIndices = grid_ids

        innerbody = perfect_for_loop.inner
        if perfect_for_loop.depth == 2 and self.ParDim == 1:
            innerbody = perfect_for_loop.outer
        first_loop = tvisitor.ForLoops()

        first_loop.visit(innerbody.compound)
        loop_indices = tvisitor.LoopIndices()
        if first_loop.ast is not None:
            loop_indices.visit(innerbody.compound)
            self.Loops = loop_indices.Loops

        loops = visitor.ForLoops()
        loops.visit(ast)
        for_loop_ast = loops.ast
        loop_indices = visitor.LoopIndices()
        loop_indices.visit(for_loop_ast)
        self.loop_index = loop_indices.index
        self.UpperLimit = loop_indices.end
        self.LowerLimit = loop_indices.start

        arrays = visitor.Arrays(self.loop_index)
        arrays.visit(ast)

        self.Subscript = arrays.Subscript

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

    def place_in_local(self):
        """ Find all array references that can be optimized
        	through the use of shared memory.
            Then rewrite the code in this fashion.
        """

        args = dict()
        loopindex = set()
        for k, v in self.SubscriptNoId.items():
            for i, n in enumerate(v):
                if set(n) & set(self.GridIndices) and \
                                set(n) & set(self.Loops.keys()):
                    if self.ParDim == 2:
                        args[k] = [i]
                        loopindex = loopindex.union(set(n) & set(self.Loops.keys()))

        loopindex = list(loopindex)
        if args:
            self.PlaceInLocalArgs.append(args)

        for m in loopindex:
            cond = lan.BinOp(lan.BinOp(lan.BinOp(lan.Id(self.UpperLimit[m]), '-', \
                                                 lan.Id(self.LowerLimit[m])), '%', \
                                       lan.Constant(self.Local['size'][0])), '==', lan.Constant(0))
            self.PlaceInLocalCond = cond
