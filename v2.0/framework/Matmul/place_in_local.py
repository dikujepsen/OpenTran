import lan
import copy
import ast_buildingblock as ast_bb
import exchange
import collect_gen as cg
import collect_id as ci
import collect_loop as cl
import collect_array as ca
import collect_device as cd


class PlaceInLocal(object):
    def __init__(self, ast):
        self.ast = ast
        self.PlaceInLocalArgs = list()
        self.PlaceInLocalCond = None

    def place_in_local(self):
        """ Find all array references that can be optimized
            through the use of shared memory.
            Then rewrite the code in this fashion.
        """

        args = dict()
        loopindex = set()
        inner_loop_indices = cl.get_inner_loops_indices(self.ast)

        subscript_no_id = ca.get_subscript_no_id(self.ast)

        for k, sub_list in subscript_no_id.items():
            for i, sub in enumerate(sub_list):
                if self.__can_be_put_in_local(sub, inner_loop_indices):
                    args[k] = i
                    loopindex = loopindex.union(set(sub).intersection(set(inner_loop_indices)))

        loopindex = list(loopindex)

        if len(loopindex) > 1:
            raise Exception("""place_in_reg: loopindex length above 1""")

        if args:
            self.PlaceInLocalArgs.append(args)

        self.__set_condition(loopindex)

    def __set_condition(self, loopindex):
        (lower_limit, upper_limit) = cl.get_loop_limits(self.ast)
        local = cl.get_local(self.ast)
        for m in loopindex:
            cond = lan.BinOp(lan.BinOp(lan.BinOp(lan.Id(upper_limit[m]), '-',
                                                 lan.Id(lower_limit[m])), '%',
                                       lan.Constant(local['size'][0])), '==', lan.Constant(0))
            self.PlaceInLocalCond = cond

    def __can_be_put_in_local(self, sub, inner_loop_indices):
        """
        The subscript must be two dimensional. One index must be a grid index, the other an inner loop index.
        :param sub:
        :param inner_loop_indices:
        :return:
        """
        grid_indices = cl.get_grid_indices(self.ast)
        par_dim = cl.get_par_dim(self.ast)
        return set(sub).intersection(set(grid_indices)) and \
               set(sub).intersection(set(inner_loop_indices)) \
               and par_dim == 2

    def local_memory3(self, arr_dict):
        initstats = []
        init_comp = lan.GroupCompound(initstats)
        kernel = cd.get_kernel(self.ast)
        kernel.statements.insert(0, init_comp)

        loop_dict = self.__find_array_ref_to_inner_loop_idx_mapping(arr_dict)

        self.__loop_dict_is_not_safe(arr_dict, loop_dict)

        # Find which loops must be extended
        loops_to_be_extended = set()
        for n in arr_dict:
            i = arr_dict[n]
            loops_to_be_extended.add(loop_dict[(n, i)][0])

        outerstats = self.__extend_loops(loops_to_be_extended)

        self.__allocate_local_arrays(initstats, arr_dict)

        loadings = []
        loop_arrays = ca.get_loop_arrays(self.ast)
        local = cl.get_local(self.ast)
        for n in arr_dict:
            loc_name = n + '_local'
            i = arr_dict[n]
            glob_subs = copy.deepcopy(loop_arrays[n][i])
            # Change loop idx to local idx
            loopname = loop_dict[(n, i)][0]
            loc_subs_2 = copy.deepcopy(glob_subs).subscript
            my_new_glob_sub_2 = self.__create_glob_load_subscript(glob_subs, loc_subs_2, loopname, n)

            self.__set_local_sub(loc_subs_2)

            loc_ref = lan.ArrayRef(lan.Id(loc_name), loc_subs_2)

            loadings.append(lan.Assignment(loc_ref, my_new_glob_sub_2))

            inner_loc = loop_arrays[n][i]
            self.__exchange_load_local_loop_idx(loopname, loc_name, inner_loc)

            self.__exchange_load_local_idx(inner_loc)

            self.ast.ext.append(lan.Block(lan.Id(loc_name), local['size']))

        # Must also create the barrier
        mem_fence_func = self.__create_local_mem_fence()
        loadings.append(mem_fence_func)

        outerstats.insert(0, lan.GroupCompound(loadings))
        outerstats.append(mem_fence_func)

    def __create_local_mem_fence(self):
        arglist = lan.ArgList([lan.Id('CLK_LOCAL_MEM_FENCE')])
        func = ast_bb.EmptyFuncDecl('barrier', type=[])
        func.arglist = arglist
        return func

    def __find_array_ref_to_inner_loop_idx_mapping(self, arr_dict):
        subscript_no_id = ca.get_subscript_no_id(self.ast)
        grid_indices = cl.get_grid_indices(self.ast)

        loop_dict = dict()
        # So we create it
        for n in arr_dict:
            i = arr_dict[n]
            loop_dict[(n, i)] = []

        for n in arr_dict:
            i = arr_dict[n]
            subscript = subscript_no_id[n][i]
            inner_loop_idx = []
            for m in subscript:
                try:
                    _ = int(m)
                except ValueError:
                    if m not in grid_indices:
                        inner_loop_idx.append(m)
            loop_dict[(n, i)] = inner_loop_idx

        return loop_dict

    def __loop_dict_is_not_safe(self, arr_dict, loop_dict):
        # Check that all ArrayRefs are blocked using only one loop
        # otherwise we do not know what to do
        retval = False
        for n in arr_dict:
            i = arr_dict[n]

            if len(loop_dict[(n, i)]) > 1:
                print "Array %r is being blocked by %r. Returning..." \
                      % (n, loop_dict[(n, i)])
                retval = True

        return retval

    def __extend_loops(self, loops_to_be_extended):
        outerstats = []
        loops = cl.get_inner_loops(self.ast)
        local = cl.get_local(self.ast)

        for n in loops_to_be_extended:
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
            outerloop.inc = lan.Increment(lan.Id(outeridx), '+=' + local['size'][0])
            inneridx = outeridx * 2

            # new inner loop
            innerloop.cond = lan.BinOp(lan.Id(inneridx), '<', lan.Constant(local['size'][0]))
            innerloop.inc = lan.Increment(lan.Id(inneridx), '++')
            innerloop.init = ast_bb.ConstantAssignment(inneridx)
        return outerstats

    def __allocate_local_arrays(self, initstats, arr_dict):
        types = ci.get_types(self.ast)
        local = cl.get_local(self.ast)
        num_array_dims = ca.get_num_array_dims(self.ast)
        for n in arr_dict:
            # Add array allocations

            local_array_name = n + '_local'
            arrayinit = lan.Constant(local['size'][0])
            if num_array_dims[n] == 2:
                arrayinit = lan.BinOp(arrayinit, '*', lan.Constant(local['size'][1]))

            local_array_id = lan.Id(local_array_name)

            local_type_id = lan.ArrayTypeId(['__local', types[n][0]], local_array_id, [arrayinit])
            initstats.append(local_type_id)

    def __exchange_load_local_loop_idx(self, loopname, loc_name, inner_loc):
        inner_loc.name.name = loc_name
        exchange_id2 = exchange.ExchangeId({loopname: loopname * 2})
        exchange_id2.visit(inner_loc)

    def __exchange_load_local_idx(self, inner_loc):
        reverse_idx = cg.get_reverse_idx(self.ast)
        grid_indices = cl.get_grid_indices(self.ast)
        for k, m in enumerate(inner_loc.subscript):
            if isinstance(m, lan.Id) and \
                            m.name in grid_indices:
                tid = str(reverse_idx[k])
                inner_loc.subscript[k] = ast_bb.FuncCall('get_local_id', [lan.Constant(tid)])

    def __create_glob_load_subscript(self, glob_subs, loc_subs_2, loopname, n):
        loc_subs = copy.deepcopy(glob_subs).subscript
        my_new_glob_sub = copy.deepcopy(glob_subs).subscript
        my_new_glob_sub_2 = copy.deepcopy(glob_subs)
        reverse_idx = cg.get_reverse_idx(self.ast)
        grid_indices = cl.get_grid_indices(self.ast)
        for k, m in enumerate(loc_subs):
            if isinstance(m, lan.Id) and \
                            m.name not in grid_indices:
                tid = str(reverse_idx[k])
                tidstr = ast_bb.FuncCall('get_local_id', [lan.Constant(tid)])

                loc_subs_2[k] = tidstr

                my_new_glob_sub[k] = lan.BinOp(lan.Id(loopname), '+', tidstr)
                my_new_glob_sub_2 = lan.ArrayRef(lan.Id(n), my_new_glob_sub)

        return my_new_glob_sub_2

    def __set_local_sub(self, loc_subs_2):
        reverse_idx = cg.get_reverse_idx(self.ast)
        grid_indices = cl.get_grid_indices(self.ast)
        for k, m in enumerate(loc_subs_2):
            if isinstance(m, lan.Id) and \
                            m.name in grid_indices:
                tid = str(reverse_idx[k])
                loc_subs_2[k] = ast_bb.FuncCall('get_local_id', [lan.Constant(tid)])
