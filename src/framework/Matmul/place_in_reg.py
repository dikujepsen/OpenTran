import copy
import lan
from itertools import chain
import collect_transformation_info as cti


class PlaceInReg(object):
    def __init__(self):
        self.RefToLoop = dict()
        self.GridIndices = list()
        self.ParDim = None  # int
        self.WriteOnly = list()
        self.ReadWrite = dict()
        self.SubscriptNoId = dict()
        self.UpperLimit = dict()
        self.LowerLimit = dict()
        self.Loops = dict()

        self.PlaceInRegArgs = list()
        self.PlaceInRegCond = None

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
        self.RefToLoop = fpl.RefToLoop
        self.ReadWrite = fai.ReadWrite
        self.WriteOnly = fai.WriteOnly
        self.SubscriptNoId = fs.SubscriptNoId

    def place_in_reg(self):
        """ Find all array references that can be cached in registers.
            Then rewrite the code in this fashion.
        """

        optim = dict()
        insideloop = set()
        for n in self.RefToLoop:
            if n in self.WriteOnly:
                continue

            ref1 = self.RefToLoop[n]
            sub1 = self.SubscriptNoId[n]

            for (ref, sub, i) in zip(ref1, sub1, range(len(ref1))):
                if self._can_perform_optimization(ref, sub):
                    insideloop |= set(sub) - set(self.GridIndices)
                    try:
                        optim[n].append(i)
                    except KeyError:
                        optim[n] = [i]

        insideloop = self._remove_unknown_loops(insideloop)

        if len(insideloop) > 1:
            print """ PlaceInReg: array references was inside two loops. No optimization """
            return

        args = optim

        insideloop = list(insideloop)
        if args:
            # print 'Register ' , args
            self.PlaceInRegArgs.append((args, insideloop))

            numref = len(list(chain.from_iterable(args.values())))
            if insideloop:
                m = insideloop[0]
                lhs = lan.BinOp(lan.Id(self.UpperLimit[m]), '-', lan.Id(self.LowerLimit[m]))
            else:
                lhs = lan.Constant(1)
            self.PlaceInRegCond = lan.BinOp(lan.BinOp(lhs, '*', lan.Constant(numref)), '<', lan.Constant(40))

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


    @staticmethod
    def place_in_reg2(ks, arr_dict):
        stats = ks.Kernel.statements
        initstats = []
        writes = []
        # Create the loadings
        for i, n in enumerate(arr_dict):
            for m in arr_dict[n]:
                idx = m
                sub = copy.deepcopy(ks.LoopArrays[n][idx])
                types = ks.Type[n][0]
                regid = lan.Id(n + str(m) + '_reg')
                reg = lan.TypeId([types], regid)
                writes.append(regid)
                assign = lan.Assignment(reg, sub)
                initstats.append(assign)

        stats.insert(0, lan.GroupCompound(initstats))

        # Replace the global Arefs with the register Arefs
        count = 0
        for i, n in enumerate(arr_dict):
            for m in arr_dict[n]:
                idx = m
                aref_new = writes[count]
                aref_old = ks.LoopArrays[n][idx]
                # Copying the internal data of the two arefs
                aref_old.name.name = aref_new.name
                aref_old.subscript = []
                count += 1

    def place_in_reg3(self, ks, arr_dict, inside_list):
        """ Check if the arrayref is inside a loop and use a static
            array for the allocation of the registers
            :param ks:
                kernelstruct
            :param arr_dict:
                dictionary of arrays
            :param inside_list:
                what loop we are inside
        """
        stats = ks.Kernel.statements
        initstats = []
        writes = []

        if not arr_dict:
            return

        if not inside_list:
            self.place_in_reg2(ks, arr_dict)
            return

        insideloop = inside_list[0]

        if insideloop == '':
            print "placeInReg3 only works when the ArrayRef is inside a loop"
            print arr_dict
            return

        # Add allocation of registers to the initiation stage
        for n in arr_dict:
            lval = lan.TypeId([ks.Type[n][0]],
                              lan.Id(n + '_reg[' + str(self.UpperLimit[insideloop]) + ']'))
            initstats.append(lval)

        # add the loop to the initiation stage
        loop = copy.deepcopy(self.Loops[insideloop])
        loopstats = []
        # Exchange loop index
        loop.compound.statements = loopstats

        initstats.append(loop)

        # Create the loadings
        for i, n in enumerate(arr_dict):
            for m in arr_dict[n]:
                idx = m
                sub = copy.deepcopy(ks.LoopArrays[n][idx])
                regid = lan.ArrayRef(lan.Id(n + '_reg'), [lan.Id(insideloop)])
                writes.append(regid)
                assign = lan.Assignment(regid, sub)
                loopstats.append(assign)

        stats.insert(0, lan.GroupCompound(initstats))
        # Replace the global Arefs with the register Arefs
        count = 0
        for i, n in enumerate(arr_dict):
            for m in arr_dict[n]:
                idx = m
                aref_new = copy.deepcopy(writes[count])
                aref_old = ks.LoopArrays[n][idx]
                # Copying the internal data of the two arefs
                aref_old.name.name = aref_new.name.name
                aref_old.subscript = aref_new.subscript
                count += 1
