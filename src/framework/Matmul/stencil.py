import copy
import ast_buildingblock as ast_bb
import lan
import transf_visitor as tvisitor
from Matmul import visitor
import collect


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
        perfect_for_loop = tvisitor.PerfectForLoop()
        perfect_for_loop.visit(ast)

        if self.ParDim is None:
            self.ParDim = perfect_for_loop.depth

        find_kernel = collect.FindKernel(self.ParDim)
        find_kernel.visit(ast)
        self.Kernel = find_kernel.kernel

        gen_local_array_idx = collect.GenLocalArrayIdx()
        gen_local_array_idx.collect(ast, self.ParDim)
        self.IndexToLocalVar = gen_local_array_idx.IndexToLocalVar

        col_li = collect.LoopIndices(self.ParDim)
        col_li.visit(ast)
        grid_indices = col_li.grid_indices
        self.GridIndices = grid_indices

        loops = visitor.ForLoops()
        loops.visit(ast)
        for_loop_ast = loops.ast
        loop_indices = visitor.LoopIndices()
        loop_indices.visit(for_loop_ast)
        self.LowerLimit = loop_indices.start

        arrays = visitor.Arrays(loop_indices.index)
        arrays.visit(ast)

        for n in arrays.numIndices:
            if arrays.numIndices[n] == 2:
                arrays.numSubscripts[n] = 2
            elif arrays.numIndices[n] > 2:
                arrays.numSubscripts[n] = 1

        self.num_array_dims = arrays.numSubscripts
        self.IndexInSubscript = arrays.indexIds
        self.LoopArrays = arrays.LoopArrays
        if self.ParDim == 1:
            self.Local['size'] = ['256']
            if dev == 'CPU':
                self.Local['size'] = ['16']
        else:
            self.Local['size'] = ['16', '16']
            if dev == 'CPU':
                self.Local['size'] = ['4', '4']

        type_ids = visitor.TypeIds()
        type_ids.visit(for_loop_ast)
        type_ids2 = visitor.TypeIds()
        type_ids2.visit(ast)
        for n in type_ids.ids:
            type_ids2.dictIds.pop(n)

        self.type = type_ids2.dictIds

        find_dim = tvisitor.FindDim(self.num_array_dims)
        find_dim.visit(ast)
        self.ArrayIdToDimName = find_dim.dimNames
        self.ReverseIdx = dict()
        self.ReverseIdx[0] = 1
        self.ReverseIdx[1] = 0

    def stencil(self, arrNames, west=0, north=0, east=0, south=0, middle=1):

        isInsideLoop = False
        try:
            # Find out if arrName is inside a loop
            forLoops = visitor.ForLoops()
            forLoops.visit(self.Kernel)
            forLoopAst = forLoops.ast
            arrays = visitor.Arrays([])
            arrays.visit(forLoopAst)
            for arrName in arrNames:
                if arrName in arrays.ids:
                    isInsideLoop = True
        except AttributeError:
            pass
            ## print "NOT INSIDE LOOP"

        if isInsideLoop:
            # find loop index
            loopIndices = visitor.LoopIndices()
            loopIndices.visit(forLoopAst)
            outeridx = loopIndices.index[0]
            forLoopAst.inc = lan.Increment(lan.Id(outeridx), '+=' + self.Local['size'][0])

            inneridx = outeridx * 2
            self.Add[outeridx] = inneridx
            cond = lan.BinOp(lan.Id(inneridx), '<', lan.Constant(self.Local['size'][0]))
            innerinc = lan.Increment(lan.Id(inneridx), '++')
            innercomp = copy.copy(forLoopAst.compound)
            innerloop = lan.ForLoop(ast_bb.ConstantAssignment(inneridx), cond, \
                                    innerinc, innercomp)
            forLoopAst.compound = lan.Compound([innerloop])

        direction = [west, north, east, south, middle]
        dirname = [(0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)]
        loadings = [elem for i, elem in enumerate(dirname)
                    if direction[i] == 1]
        if not loadings:
            loadings = [(0, 0)]

        ## finding the correct local memory size
        arrName = arrNames[0]
        localDims = [int(self.Local['size'][0]) \
                     for i in xrange(self.num_array_dims[arrName])]
        if self.ParDim == 1 and len(localDims) == 2:
            localDims[0] = 1;
        arrIdx = self.IndexInSubscript[arrName]
        localOffset = [int(self.LowerLimit[i]) \
                       for i in arrIdx]

        for (x, y) in loadings:
            localDims[0] += abs(x)
            if self.num_array_dims[arrName] == 2:
                localDims[1] += abs(y)

        stats = []
        for arrName in arrNames:
            localName = arrName + '_local'
            arrayinit = '['
            for i, d in enumerate(localDims):
                arrayinit += str(d)
                if i == 0 and len(localDims) == 2:
                    arrayinit += '*'
            arrayinit += ']'

            localId = lan.Id(localName + arrayinit)
            localTypeId = lan.TypeId(['__local'] + [self.type[arrName][0]], localId)
            self.num_array_dims[localName] = self.num_array_dims[arrName]
            self.LocalSwap[arrName] = localName
            self.ArrayIdToDimName[localName] = [self.Local['size'][0], self.Local['size'][0]]
            stats.append(localTypeId)

        InitComp = lan.GroupCompound(stats)
        stats2 = []
        LoadComp = lan.GroupCompound(stats2)

        ## Insert local id with offset
        for i, offset in enumerate(localOffset):
            idd = self.ReverseIdx[i] if len(localOffset) == 2 else i
            if offset != 0:

                rval = lan.BinOp(lan.Id('get_local_id(' + str(idd) + ')'), '+', \
                                 lan.Constant(offset))
            else:
                rval = lan.Id('get_local_id(' + str(idd) + ')')
            lval = lan.TypeId(['unsigned'], lan.Id('l' + self.GridIndices[i]))
            stats.append(lan.Assignment(lval, rval))

        exchangeIndices = tvisitor.ExchangeIndices(self.IndexToLocalVar, self.LocalSwap.values())

        ## Creating the loading of values into the local array.
        for arrName in arrNames:
            for k, l in enumerate(loadings):
                arrayId = lan.Id(arrName)
                # get first ArrayRef
                aref = self.LoopArrays[arrName][k]
                subscript = aref.subscript
                lsub = copy.deepcopy(subscript)
                lval = lan.ArrayRef(lan.Id(self.LocalSwap[arrName]), lsub)
                rsub = copy.deepcopy(subscript)
                rval = lan.ArrayRef(arrayId, rsub, extra={'localMemory': True})
                load = lan.Assignment(lval, rval)
                exchangeId = tvisitor.ExchangeId(self.IndexToLocalVar)
                orisub = subscript
                for m in orisub:
                    exchangeId.visit(m)
                if isInsideLoop:
                    print "INSIDE22"
                    for i, n in enumerate(orisub):
                        addToId = visitor.Ids()
                        addToId.visit(n)
                        # REMEMBER: orisub[i] might not simply be an Id
                        # might need to do something more complicated here
                        if outeridx in addToId.ids:
                            orisub[i] = lan.Id(inneridx)

                    for i, n in enumerate(rsub):  # GlobalLoad
                        idd = self.ReverseIdx[i] if self.num_array_dims[arrName] == 2 else i
                        locIdx = 'get_local_id(' + str(idd) + ')'
                        addToId = lan.Ids()
                        addToId.visit(n)
                        if outeridx in addToId.ids:
                            rsub[i] = lan.BinOp(rsub[i], '+', \
                                                lan.Id(locIdx))
                    for i, n in enumerate(lsub):  # Local Write
                        idd = self.ReverseIdx[i] if self.num_array_dims[arrName] == 2 else i
                        locIdx = 'get_local_id(' + str(idd) + ')'
                        exchangeId = lan.ExchangeId({'' + outeridx: locIdx})
                        exchangeId.visit(n)

                stats2.append(load)

        # Must also create the barrier
        arglist = lan.ArgList([lan.Id('CLK_LOCAL_MEM_FENCE')])
        func = ast_bb.EmptyFuncDecl('barrier', type=[])
        func.arglist = arglist
        stats2.append(func)

        exchangeIndices.visit(InitComp)
        exchangeIndices.visit(LoadComp)

        if isInsideLoop:
            forLoopAst.compound.statements.insert(0, LoadComp)
            forLoopAst.compound.statements.append(func)
        else:
            self.Kernel.statements.insert(0, LoadComp)
        self.Kernel.statements.insert(0, InitComp)
