import copy
import lan
from itertools import chain
import collect_transformation_info as cti
import exchange
import collect_array as ca
import collect_loop as cl
import collect_id as ci
import collect_device as cd


class PlaceInReg(object):
    def __init__(self, ast):
        self.ast = ast

        self.PlaceInRegFinding = tuple()
        self.PlaceInRegCond = None

        self.perform_transformation = False

    def place_in_reg(self):
        """ Find all array references that can be cached in registers.
            Then rewrite the code in this fashion.
        """
        optimizable_arrays = dict()
        hoist_loop_set = set()

        ref_to_loop = ca.get_ref_to_loop(self.ast)
        write_only = ca.get_write_only(self.ast)
        subscript_no_id = ca.get_subscript_no_id(self.ast)
        grid_indices = cl.get_grid_indices(self.ast)

        for n in ref_to_loop:
            if n in write_only:
                continue

            ref1 = ref_to_loop[n]
            sub1 = subscript_no_id[n]

            for (ref, sub, i) in zip(ref1, sub1, range(len(ref1))):
                if self._can_perform_optimization(ref, sub):
                    hoist_loop_set |= set(sub) - set(grid_indices)
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
        (lower_limit, upper_limit) = cl.get_loop_limits(self.ast)
        if hoistloop:
            m = hoistloop[0]
            lhs = lan.BinOp(lan.Id(upper_limit[m]), '-', lan.Id(lower_limit[m]))
        else:
            lhs = lan.Constant(1)
        self.PlaceInRegCond = lan.BinOp(lan.BinOp(lhs, '*', lan.Constant(num_ref_hoisted)), '<', lan.Constant(40))

    def _set_optimization_arg(self, optimizable_arrays, hoistloop):
        self.PlaceInRegFinding = (optimizable_arrays, hoistloop)

    def _remove_unknown_loops(self, insideloop):
        loops = cl.get_inner_loops(self.ast)
        return {k for k in insideloop if k in loops}

    def _can_perform_optimization(self, loop_idx, sub_idx):
        """
        # for each array, for each array ref, collect which loop, loop_idx, it is in
        # and what indices, sub_idx, are in its subscript.
        # if there is a grid_idx in sub_idx and there exists a loop_idx not in sub_idx

        :param loop_idx:
        :param sub_idx:
        :return:
        """
        grid_indices = cl.get_grid_indices(self.ast)
        return set(sub_idx).intersection(set(grid_indices)) and \
               set(loop_idx).difference(set(sub_idx))

    def place_in_reg2(self, arr_dict):
        self._insert_cache_in_reg(arr_dict)
        self._replace_global_ref_with_reg_id(arr_dict)

    def _insert_cache_in_reg(self, arr_dict):
        initstats = []
        # Create the loadings
        types = ci.get_types(self.ast)

        kernel = cd.get_kernel(self.ast)
        kernel_stats = kernel.statements

        for i, n in enumerate(arr_dict):
            for m in arr_dict[n]:
                regid = self._create_reg_var_id(m, n)

                reg_type = types[n][0]
                reg = lan.TypeId([reg_type], regid)
                assign = self._create_reg_assignment(m, n, reg)

                initstats.append(assign)
        kernel_stats.insert(0, lan.GroupCompound(initstats))

    def _replace_global_ref_with_reg_id(self, arr_dict):
        # Replace the global Arefs with the register vars
        loop_arrays = ca.get_loop_arrays(self.ast)
        loop_arrays_parent = ca.get_loop_arrays_parent(self.ast)

        for i, n in enumerate(arr_dict):
            for m in arr_dict[n]:
                idx = m
                reg_id = self._create_reg_var_id(m, n)
                parent = loop_arrays_parent[n][idx]
                aref_old = loop_arrays[n][idx]
                exchange_array_id_with_id = exchange.ExchangeArrayIdWithId(aref_old, reg_id)
                exchange_array_id_with_id.visit(parent)

    @staticmethod
    def _create_reg_var_id(m, n):
        return lan.Id(n + str(m) + '_reg')

    def _create_reg_assignment(self, m, n, reg):

        idx = m
        loop_arrays = ca.get_loop_arrays(self.ast)
        glob_array_ref = copy.deepcopy(loop_arrays[n][idx])
        reg_dict = {'isReg': []}
        glob_array_ref.extra = reg_dict
        assign = lan.Assignment(reg, glob_array_ref)
        return assign

    def place_in_reg3(self):
        """ Check if the arrayref is inside a loop and use a static
            array for the allocation of the registers
        """

        kernel = cd.get_kernel(self.ast)

        kernel_stats = kernel.statements
        self.place_in_reg()

        if self.PlaceInRegFinding is ():
            return

        (optimizable_arrays, hoist_loop_list) = self.PlaceInRegFinding
        self.perform_transformation = True

        if not optimizable_arrays:
            return

        if not hoist_loop_list:
            self.place_in_reg2(optimizable_arrays)
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
        loop_arrays = ca.get_loop_arrays(self.ast)
        for i, n in enumerate(optimizable_arrays):
            for m in optimizable_arrays[n]:
                idx = m
                regid = self._create_reg_array_var(n, hoist_loop)
                aref_new = copy.deepcopy(regid)
                aref_old = loop_arrays[n][idx]
                # Copying the internal data of the two arefs
                aref_old.name.name = aref_new.name.name
                aref_old.subscript = aref_new.subscript

    def _create_load_loop(self, hoist_loop, initstats):
        loops = cl.get_inner_loops(self.ast)
        loop = copy.deepcopy(loops[hoist_loop])
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
        types = ci.get_types(self.ast)
        (_, upper_limit) = cl.get_loop_limits(self.ast)
        # Add allocation of registers to the initiation stage
        for n in optimizable_arrays:
            array_init = lan.ArrayTypeId([types[n][0]], lan.Id(n + '_reg'), [lan.Id(upper_limit[hoist_loop])])
            initstats.append(array_init)
        return initstats
