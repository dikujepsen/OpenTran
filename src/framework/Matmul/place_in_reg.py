import visitor
import transf_visitor as tvisitor
import copy
import lan
from itertools import chain


class PlaceInReg(object):
    def __init__(self):
        self.RefToLoop = dict()
        self.GridIndices = list()
        self.ParDim = None  # int
        self.WriteOnly = list()
        self.ReadWrite = dict()
        self.ArrayIds = set()
        self.SubscriptNoId = dict()
        self.PlaceInRegArgs = list()
        self.PlaceInRegCond = None
        self.UpperLimit = dict()
        self.LowerLimit = dict()
        self.Loops = dict()

    def set_datastructures(self, ast):
        perfect_for_loop = tvisitor.PerfectForLoop()
        perfect_for_loop.visit(ast)
        if self.ParDim is None:
            self.ParDim = perfect_for_loop.depth

        innerbody = perfect_for_loop.inner
        if perfect_for_loop.depth == 2 and self.ParDim == 1:
            innerbody = perfect_for_loop.outer
        first_loop = tvisitor.ForLoops()

        first_loop.visit(innerbody.compound)
        loop_indices = tvisitor.LoopIndices()
        if first_loop.ast is not None:
            loop_indices.visit(innerbody.compound)
            self.Loops = loop_indices.Loops

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
        self.Kernel = kernel

        ref_to_loop = tvisitor.RefToLoop(self.GridIndices)
        ref_to_loop.visit(ast)
        self.RefToLoop = ref_to_loop.RefToLoop

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

        type_ids = visitor.TypeIds()
        type_ids.visit(for_loop_ast)

        ids = visitor.Ids2()
        ids.visit(ast)

        other_ids = ids.ids - arrays.ids - type_ids.ids
        self.ArrayIds = arrays.ids - type_ids.ids

        find_read_write = tvisitor.FindReadWrite(self.ArrayIds)
        find_read_write.visit(ast)
        self.ReadWrite = find_read_write.ReadWrite

        for n in self.ReadWrite:
            pset = self.ReadWrite[n]
            if len(pset) == 1:
                if 'write' in pset:
                    self.WriteOnly.append(n)

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

    def place_in_reg(self):
        """ Find all array references that can be cached in registers.
        	Then rewrite the code in this fashion.
        """

        optim = dict()
        for n in self.RefToLoop:
            optim[n] = []
        insideloop = set()
        for n in self.RefToLoop:
            if n in self.WriteOnly:
                continue

            ref1 = self.RefToLoop[n]
            sub1 = self.SubscriptNoId[n]

            for (ref, sub, i) in zip(ref1, sub1, range(len(ref1))):
                ## print rw.GridIndices, sub
                if set(self.GridIndices) & set(sub):
                    outerloops = set(ref) - set(sub)
                    if outerloops:
                        insideloop |= set(sub) - set(self.GridIndices)
                        optim[n].append(i)

                        ## print i, n, outerloops, rw.GridIndices, sub, \
                        ##   set(sub) - set(rw.GridIndices)

        insideloop = {k for k in insideloop if k in self.Loops}
        if len(insideloop) > 1:
            print """ PlaceInReg: array references was inside
    				  two loops. No optimization """
            return
        ## print insideloop
        args = {k: v for k, v in optim.items() if v}
        insideloop = list(insideloop)
        if args:
            ## print 'Register ' , args
            self.PlaceInRegArgs.append((args, insideloop))

            numref = len(list(chain.from_iterable(args.values())))
            if insideloop:
                m = insideloop[0]
                lhs = lan.BinOp(lan.Id(self.UpperLimit[m]), '-', lan.Id(self.LowerLimit[m]))
            else:
                lhs = lan.Constant(1)
            self.PlaceInRegCond = lan.BinOp(lan.BinOp(lhs, '*', lan.Constant(numref)), '<', lan.Constant(40))
