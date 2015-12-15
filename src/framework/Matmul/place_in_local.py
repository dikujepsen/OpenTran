import lan
import copy
import visitor
import transf_visitor as tvisitor
import ast_buildingblock as ast_bb
import collect_transformation_info as cti

class PlaceInLocal(object):
    def __init__(self):
        self.SubscriptNoId = dict()
        self.GridIndices = list()
        self.ParDim = None  # int
        self.Loops = dict()
        self.UpperLimit = dict()
        self.LowerLimit = dict()

        self.PlaceInLocalArgs = list()
        self.PlaceInLocalCond = None
        self.Local = dict()
        self.Local['name'] = 'LSIZE'
        self.Local['size'] = ['64']

    def set_datastructures(self, ast, dev='CPU'):

        fpl = cti.FindGridIndices()
        fpl.ParDim = self.ParDim
        fpl.collect(ast)

        fs = cti.FindSubscripts()
        fs.collect(ast)

        self.ParDim = fpl.par_dim
        self.GridIndices = fpl.GridIndices
        self.Loops = fs.Loops
        self.UpperLimit = fs.upper_limit
        self.LowerLimit = fs.lower_limit
        self.SubscriptNoId = fs.SubscriptNoId

        fl = cti.FindLocal()
        fl.ParDim = self.ParDim
        fl.collect(ast)
        self.Local = fl.Local


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

    def localMemory3(self, ks, arrDict, loopDict=None, blockDict=None):
        initstats = []
        initComp = lan.GroupCompound(initstats)
        ks.Kernel.statements.insert(0, initComp)

        if loopDict is None:
            loopDict = dict()
            # So we create it
            for n in arrDict:
                for i in arrDict[n]:
                    loopDict[(n, i)] = []

            for n in arrDict:
                for i in arrDict[n]:
                    subscript = ks.SubscriptNoId[n][i]
                    acc = []
                    for m in subscript:
                        try:
                            _ = int(m)
                        except:
                            if m not in ks.GridIndices:
                                acc.append(m)
                    loopDict[(n, i)] = acc

        # Check that all ArrayRefs are blocked using only one loop
        # otherwise we do not know what to do
        for n in arrDict:
            for i in arrDict[n]:
                if len(loopDict[(n, i)]) > 1:
                    print "Array %r is being blocked by %r. Returning..." \
                          % (n, loopDict[(n, i)])
                    return

        # Find which loops must be extended
        loopext = set()
        for n in arrDict:
            for i in arrDict[n]:
                loopext.add(loopDict[(n, i)][0])

        # do the extending
        for n in loopext:
            outerloop = ks.Loops[n]
            outeridx = n
            compound = outerloop.compound
            outerloop.compound = lan.Compound([])
            innerloop = copy.deepcopy(outerloop)
            innerloop.compound = compound
            outerstats = outerloop.compound.statements
            outerstats.insert(0, innerloop)
            loadstats = []
            loadComp = lan.GroupCompound(loadstats)
            outerstats.insert(0, loadComp)
            # change increment of outer loop
            outerloop.inc = lan.Increment(lan.Id(outeridx), '+=' + ks.Local['size'][0])
            inneridx = outeridx * 2
            # For adding to this index in other subscripts
            ks.Add[outeridx] = inneridx

            # new inner loop
            innerloop.cond = lan.BinOp(lan.Id(inneridx), '<', lan.Constant(ks.Local['size'][0]))
            innerloop.inc = lan.Increment(lan.Id(inneridx), '++')
            innerloop.init = ast_bb.ConstantAssignment(inneridx)
            ks.Loops[inneridx] = innerloop

        for n in arrDict:
            # Add array allocations

            localName = n + '_local'
            arrayinit = '['
            arrayinit += ks.Local['size'][0]
            if ks.num_array_dims[n] == 2 and ks.ParDim == 2:
                arrayinit += '*' + ks.Local['size'][1]
            arrayinit += ']'

            localId = lan.Id(localName + arrayinit)
            localTypeId = lan.TypeId(['__local'] + [ks.Type[n][0]], localId)
            initstats.append(localTypeId)

        loadings = []
        for n in arrDict:
            loc_name = n + '_local'
            for i in arrDict[n]:
                glob_subs = copy.deepcopy(ks.LoopArrays[n][i])
                # Change loop idx to local idx
                loopname = loopDict[(n, i)][0]
                loc_subs = copy.deepcopy(glob_subs).subscript
                for k, m in enumerate(loc_subs):
                    if isinstance(m, lan.Id) and \
                                    m.name not in ks.GridIndices:
                        tid = str(ks.ReverseIdx[k])
                        tidstr = 'get_local_id(' + tid + ')'
                        exchangeId = tvisitor.ExchangeId({loopname: tidstr})
                        exchangeId.visit(m)
                        exchangeId2 = tvisitor.ExchangeId({loopname: '(' + loopname + ' + ' + tidstr + ')'})
                        exchangeId2.visit(glob_subs.subscript[k])
                loc_ref = lan.ArrayRef(lan.Id(loc_name), loc_subs)

                loadings.append(lan.Assignment(loc_ref, glob_subs))
                if ks.ParDim == 2:
                    exchangeId = tvisitor.ExchangeId(
                        {ks.GridIndices[1]: 'get_local_id(0)', ks.GridIndices[0]: 'get_local_id(1)'})
                else:
                    exchangeId = tvisitor.ExchangeId({ks.GridIndices[0]: 'get_local_id(0)'})
                exchangeId.visit(loc_ref)

                inner_loc = ks.LoopArrays[n][i]
                inner_loc.name.name = loc_name
                exchangeId2 = tvisitor.ExchangeId({loopname: loopname * 2})
                exchangeId2.visit(inner_loc)
                exchangeId.visit(inner_loc)

            ks.ArrayIdToDimName[loc_name] = ks.Local['size']
            ks.num_array_dims[loc_name] = ks.num_array_dims[n]
        # Must also create the barrier
        arglist = lan.ArgList([lan.Id('CLK_LOCAL_MEM_FENCE')])
        func = ast_bb.EmptyFuncDecl('barrier', type=[])
        func.arglist = arglist
        loadings.append(func)

        outerstats.insert(0, lan.GroupCompound(loadings))
        outerstats.append(func)
