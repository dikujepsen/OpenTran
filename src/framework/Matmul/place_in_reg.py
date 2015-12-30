import copy
import lan
from itertools import chain
import collect_transformation_info as cti
import exchange
import collect_array as ca


class PlaceInReg(object):
    def __init__(self):
        self.GridIndices = list()
        self.ParDim = None  # int
        self.ReadWrite = dict()
        self.UpperLimit = dict()
        self.LowerLimit = dict()
        self.Loops = dict()

        self.PlaceInRegFinding = tuple()
        self.PlaceInRegCond = None

        self.ks = None
        self.perform_transformation = False

    def set_datastructures(self, ast):

        fpl = cti.FindGridIndices()
        fpl.ParDim = self.ParDim
        fpl.collect(ast)

        fs = cti.FindSubscripts()
        fs.collect(ast)

        fai = cti.FindReadWrite()
        fai.ParDim = self.ParDim
        fai.collect(ast)

        self.UpperLimit = fai.upper_limit
        self.LowerLimit = fai.lower_limit
        self.Loops = fs.Loops
        self.GridIndices = fpl.GridIndices
        self.ReadWrite = fai.ReadWrite

    def place_in_reg(self, ast, par_dim):
        """ Find all array references that can be cached in registers.
            Then rewrite the code in this fashion.
            :param ast:
            :param par_dim:
        """

        optimizable_arrays = dict()
        hoist_loop_set = set()

        ref_to_loop = ca.get_ref_to_loop(ast, par_dim=par_dim)
        write_only = ca.get_write_only(ast)
        subscript_no_id = ca.get_subscript_no_id(ast)

        for n in ref_to_loop:
            if n in write_only:
                continue

            ref1 = ref_to_loop[n]
            sub1 = subscript_no_id[n]

            for (ref, sub, i) in zip(ref1, sub1, range(len(ref1))):
                if self._can_perform_optimization(ref, sub):
                    hoist_loop_set |= set(sub) - set(self.GridIndices)
                    try:
                        optimizable_arrays[n].append(i)
                    except KeyError:
                        optimizable_arrays[n] = [i]

        hoist_loop_set = self._remove_unknown_loops(hoist_loop_set)

        if len(hoist_loop_set) > 1:
            print """ PlaceInReg: array references was inside two loops. No optimization. """
            return

        hoist_loop_list = list(hoist_loop_set)

        if optimizable_arrays:
            self._set_optimization_arg(optimizable_arrays, hoist_loop_list)

            self._set_optimization_condition(optimizable_arrays, hoist_loop_list)

    def _set_optimization_condition(self, optimizable_arrays, hoistloop):
        num_ref_hoisted = len(list(chain.from_iterable(optimizable_arrays.values())))
        if hoistloop:
            m = hoistloop[0]
            lhs = lan.BinOp(lan.Id(self.UpperLimit[m]), '-', lan.Id(self.LowerLimit[m]))
        else:
            lhs = lan.Constant(1)
        self.PlaceInRegCond = lan.BinOp(lan.BinOp(lhs, '*', lan.Constant(num_ref_hoisted)), '<', lan.Constant(40))

    def _set_optimization_arg(self, optimizable_arrays, hoistloop):
        self.PlaceInRegFinding = (optimizable_arrays, hoistloop)

    def _remove_unknown_loops(self, insideloop):
        return {k for k in insideloop if k in self.Loops}

    def _can_perform_optimization(self, loop_idx, sub_idx):
        """
        # for each array, for each array ref, collect which loop, loop_idx, it is in
        # and what indices, sub_idx, are in its subscript.
        # if there is a grid_idx in sub_idx and there exists a loop_idx not in sub_idx

        :param loop_idx:
        :param sub_idx:
        :return:
        """
        return set(sub_idx).intersection(set(self.GridIndices)) and \
            set(loop_idx).difference(set(sub_idx))

    def place_in_reg2(self, ks, arr_dict):
        self.ks = ks
        kernel_stats = ks.Kernel.statements
        self._insert_cache_in_reg(kernel_stats, arr_dict)
        self._replace_global_ref_with_reg_id(arr_dict)

    def _insert_cache_in_reg(self, kernel_stats, arr_dict):
        initstats = []
        # Create the loadings
        for i, n in enumerate(arr_dict):
            for m in arr_dict[n]:
                regid = self._create_reg_var_id(m, n)
                types = self.ks.Type[n][0]
                reg = lan.TypeId([types], regid)
                assign = self._create_reg_assignment(m, n, reg)
                initstats.append(assign)
        kernel_stats.insert(0, lan.GroupCompound(initstats))

    def _replace_global_ref_with_reg_id(self, arr_dict):
        # Replace the global Arefs with the register vars
        for i, n in enumerate(arr_dict):
            for m in arr_dict[n]:
                idx = m
                reg_id = self._create_reg_var_id(m, n)
                parent = self.ks.LoopArraysParent[n][idx]
                aref_old = self.ks.LoopArrays[n][idx]
                exchange_array_id_with_id = exchange.ExchangeArrayIdWithId(aref_old, reg_id)
                exchange_array_id_with_id.visit(parent)

    @staticmethod
    def _create_reg_var_id(m, n):
        return lan.Id(n + str(m) + '_reg')

    def _create_reg_assignment(self, m, n, reg):

        idx = m
        glob_array_ref = copy.deepcopy(self.ks.LoopArrays[n][idx])
        assign = lan.Assignment(reg, glob_array_ref)
        return assign

    def place_in_reg3(self, ast, par_dim, ks):
        """ Check if the arrayref is inside a loop and use a static
            array for the allocation of the registers
            :param ks:
                kernelstruct
            :param ast:
                tree
            :param par_dim:
                number of parallel dimensions
        """
        self.ks = ks
        kernel_stats = ks.Kernel.statements
        self.place_in_reg(ast, par_dim)

        # print "NEXT" , (optimizable_arrays, hoist_loop_list)

        if self.PlaceInRegFinding is ():
            return

        (optimizable_arrays, hoist_loop_list) = self.PlaceInRegFinding
        self.perform_transformation = True
        # print self.ks.PlaceInRegArgs
        # print []

        if not optimizable_arrays:
            return

        if not hoist_loop_list:
            self.place_in_reg2(ks, optimizable_arrays)
            return

        hoist_loop = hoist_loop_list[0]

        if hoist_loop == '':
            print "placeInReg3 only works when the ArrayRef is inside a loop"
            print optimizable_arrays
            return

        initstats = self._create_reg_array_alloc(optimizable_arrays, hoist_loop)

        # add the load loop to the initiation stage
        loopstats = self._create_load_loop(hoist_loop, initstats)

        # Create the loadings
        for i, n in enumerate(optimizable_arrays):
            for m in optimizable_arrays[n]:
                regid = self._create_reg_array_var(n, hoist_loop)
                assign = self._create_reg_assignment(m, n, regid)
                loopstats.append(assign)

        kernel_stats.insert(0, lan.GroupCompound(initstats))
        # Replace the global Arefs with the register Arefs
        for i, n in enumerate(optimizable_arrays):
            for m in optimizable_arrays[n]:
                idx = m
                regid = self._create_reg_array_var(n, hoist_loop)
                aref_new = copy.deepcopy(regid)
                aref_old = ks.LoopArrays[n][idx]
                # Copying the internal data of the two arefs
                aref_old.name.name = aref_new.name.name
                aref_old.subscript = aref_new.subscript

    def _create_load_loop(self, hoist_loop, initstats):
        loop = copy.deepcopy(self.Loops[hoist_loop])
        loopstats = []
        loop.compound.statements = loopstats
        initstats.append(loop)
        return loopstats

    @staticmethod
    def _create_reg_array_var(n, hoist_loop):
        regid = lan.ArrayRef(lan.Id(n + '_reg'), [lan.Id(hoist_loop)])
        return regid

    def _create_reg_array_alloc(self, optimizable_arrays, hoist_loop):
        initstats = []
        # Add allocation of registers to the initiation stage
        for n in optimizable_arrays:
            lval = lan.TypeId([self.ks.Type[n][0]],
                              lan.Id(n + '_reg[' + str(self.UpperLimit[hoist_loop]) + ']'))
            initstats.append(lval)
        return initstats
