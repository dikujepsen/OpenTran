import lan
import copy
import ast_buildingblock as ast_bb
import collect_transformation_info as cti
import exchange
import collect_gen as cg
import collect_boilerplate_info as cbi
import collect_id as ci
import collect_loop as cl
import collect_array as ca
import collect_device as cd


class PlaceInLocal(object):
    def __init__(self):
        self.SubscriptNoId = dict()
        self.GridIndices = list()
        self.ParDim = None  # int
        self.UpperLimit = dict()
        self.LowerLimit = dict()

        self.PlaceInLocalArgs = list()
        self.PlaceInLocalCond = None
        self.Local = dict()
        self.Local['name'] = 'LSIZE'
        self.Local['size'] = ['64']
        self.ReverseIdx = dict()

        self.LoopArrays = dict()
        self.Type = dict()
        self.ast = None

    def set_datastructures(self, ast, dev='CPU'):
        self.ast = ast
        fpl = cti.FindGridIndices()
        fpl.ParDim = self.ParDim
        fpl.collect(ast)

        fs = cti.FindSubscripts()
        fs.collect(ast)

        self.ParDim = fpl.par_dim
        self.GridIndices = fpl.GridIndices
        self.UpperLimit = fs.upper_limit
        self.LowerLimit = fs.lower_limit
        self.SubscriptNoId = fs.SubscriptNoId

        fl = cti.FindLocal()
        fl.ParDim = self.ParDim
        fl.collect(ast, dev)
        self.Local = fl.Local

        gen_reverse_idx = cg.GenReverseIdx()
        self.ReverseIdx = gen_reverse_idx.ReverseIdx

        fai = cbi.FindLoopArrays()
        fai.collect(ast)
        self.LoopArrays = fai.loop_arrays
        self.Type = ci.get_types(ast)

    def place_in_local(self):
        """ Find all array references that can be optimized
            through the use of shared memory.
            Then rewrite the code in this fashion.
        """

        args = dict()
        loopindex = set()
        inner_loop_indices = cl.get_inner_loops_indices(self.ast, self.ParDim)

        # print "SubscriptNoId", self.SubscriptNoId
        # print "self.GridIndices", self.GridIndices
        # print "inner_loop_indices", inner_loop_indices

        for k, sub_list in self.SubscriptNoId.items():
            for i, sub in enumerate(sub_list):
                if self.__can_be_put_in_local(sub, inner_loop_indices):
                    args[k] = i
                    loopindex = loopindex.union(set(sub).intersection(set(inner_loop_indices)))

        loopindex = list(loopindex)
        # print loopindex
        # print args
        if args:
            self.PlaceInLocalArgs.append(args)

        # print self.PlaceInLocalArgs

        for m in loopindex:
            cond = lan.BinOp(lan.BinOp(lan.BinOp(lan.Id(self.UpperLimit[m]), '-',
                                                 lan.Id(self.LowerLimit[m])), '%',
                                       lan.Constant(self.Local['size'][0])), '==', lan.Constant(0))
            self.PlaceInLocalCond = cond

    def __can_be_put_in_local(self, sub, inner_loop_indices):
        """
        The subscript must be two dimensional. One index must be a grid index, the other an inner loop index.
        :param sub:
        :param inner_loop_indices:
        :return:
        """
        return set(sub).intersection(set(self.GridIndices)) and \
               set(sub).intersection(set(inner_loop_indices)) \
               and self.ParDim == 2

    def local_memory3(self, arr_dict, loop_dict=None):
        initstats = []
        init_comp = lan.GroupCompound(initstats)
        kernel = cd.get_kernel(self.ast, self.ParDim)
        kernel.statements.insert(0, init_comp)

        if loop_dict is None:
            loop_dict = dict()
            # So we create it
            for n in arr_dict:
                i = arr_dict[n]
                loop_dict[(n, i)] = []

            for n in arr_dict:
                i = arr_dict[n]
                subscript = self.SubscriptNoId[n][i]
                acc = []
                for m in subscript:
                    try:
                        _ = int(m)
                    except ValueError:
                        if m not in self.GridIndices:
                            acc.append(m)
                loop_dict[(n, i)] = acc

        # Check that all ArrayRefs are blocked using only one loop
        # otherwise we do not know what to do
        for n in arr_dict:
            i = arr_dict[n]

            if len(loop_dict[(n, i)]) > 1:
                print "Array %r is being blocked by %r. Returning..." \
                      % (n, loop_dict[(n, i)])
                return

        # Find which loops must be extended
        loopext = set()
        for n in arr_dict:
            i = arr_dict[n]
            loopext.add(loop_dict[(n, i)][0])

        loops = cl.get_inner_loops(self.ast, self.ParDim)
        # print loops

        outerstats = []
        # do the extending
        for n in loopext:
            outerloop = loops[n]
            outeridx = n
            compound = outerloop.compound
            outerloop.compound = lan.Compound([])
            innerloop = copy.deepcopy(outerloop)
            innerloop.compound = compound
            outerstats = outerloop.compound.statements
            outerstats.insert(0, innerloop)
            loadstats = []
            load_comp = lan.GroupCompound(loadstats)
            outerstats.insert(0, load_comp)
            # change increment of outer loop
            outerloop.inc = lan.Increment(lan.Id(outeridx), '+=' + self.Local['size'][0])
            inneridx = outeridx * 2

            # new inner loop
            innerloop.cond = lan.BinOp(lan.Id(inneridx), '<', lan.Constant(self.Local['size'][0]))
            innerloop.inc = lan.Increment(lan.Id(inneridx), '++')
            innerloop.init = ast_bb.ConstantAssignment(inneridx)

        num_array_dims = ca.get_num_array_dims(self.ast)
        for n in arr_dict:
            # Add array allocations

            local_array_name = n + '_local'
            arrayinit = lan.Constant(self.Local['size'][0])
            if num_array_dims[n] == 2 and self.ParDim == 2:
                arrayinit = lan.BinOp(arrayinit, '*', lan.Constant(self.Local['size'][1]))

            local_array_id = lan.Id(local_array_name)

            local_type_id = lan.ArrayTypeId(['__local', self.Type[n][0]], local_array_id, [arrayinit])
            initstats.append(local_type_id)

        loadings = []
        loop_arrays = ca.get_loop_arrays(self.ast)
        for n in arr_dict:
            loc_name = n + '_local'
            i = arr_dict[n]
            glob_subs = copy.deepcopy(loop_arrays[n][i])
            # Change loop idx to local idx
            loopname = loop_dict[(n, i)][0]
            loc_subs = copy.deepcopy(glob_subs).subscript
            loc_subs_2 = copy.deepcopy(glob_subs).subscript
            my_new_glob_sub = copy.deepcopy(glob_subs).subscript
            my_new_glob_sub_2 = copy.deepcopy(loop_arrays[n][i])
            for k, m in enumerate(loc_subs):
                if isinstance(m, lan.Id) and \
                                m.name not in self.GridIndices:
                    tid = str(self.ReverseIdx[k])
                    tidstr = ast_bb.FuncCall('get_local_id', [lan.Constant(tid)])

                    loc_subs_2[k] = tidstr

                    my_new_glob_sub[k] = lan.BinOp(lan.Id(loopname), '+', tidstr)
                    my_new_glob_sub_2 = lan.ArrayRef(lan.Id(n), my_new_glob_sub)

            for k, m in enumerate(loc_subs_2):
                if isinstance(m, lan.Id) and \
                                m.name in self.GridIndices:
                    tid = str(self.ReverseIdx[k])
                    loc_subs_2[k] = ast_bb.FuncCall('get_local_id', [lan.Constant(tid)])

            loc_ref = lan.ArrayRef(lan.Id(loc_name), loc_subs_2)

            loadings.append(lan.Assignment(loc_ref, my_new_glob_sub_2))

            inner_loc = loop_arrays[n][i]

            inner_loc.name.name = loc_name
            exchange_id2 = exchange.ExchangeId({loopname: loopname * 2})
            exchange_id2.visit(inner_loc)

            for k, m in enumerate(inner_loc.subscript):
                if isinstance(m, lan.Id) and \
                                m.name in self.GridIndices:
                    tid = str(self.ReverseIdx[k])
                    inner_loc.subscript[k] = ast_bb.FuncCall('get_local_id', [lan.Constant(tid)])

                    # exchange_id.visit(inner_loc)

            self.ast.ext.append(lan.Block(lan.Id(loc_name), self.Local['size']))

        # Must also create the barrier
        arglist = lan.ArgList([lan.Id('CLK_LOCAL_MEM_FENCE')])
        func = ast_bb.EmptyFuncDecl('barrier', type=[])
        func.arglist = arglist
        loadings.append(func)

        outerstats.insert(0, lan.GroupCompound(loadings))
        outerstats.append(func)
