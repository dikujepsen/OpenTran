import visitor
import transf_visitor as tvisitor
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
        self.ArrayIds = set()
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
        self.ArrayIds = fai.ArrayIds
        self.ReadWrite = fai.ReadWrite
        self.WriteOnly = fai.WriteOnly
        self.SubscriptNoId = fs.SubscriptNoId



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


    def placeInReg2(self, ks, arrDict):
        stats = ks.Kernel.statements
        initstats = []
        loadings = []
        writes = []
        # Create the loadings
        for i, n in enumerate(arrDict):
            for m in arrDict[n]:
                idx = m
                sub = copy.deepcopy(ks.LoopArrays[n][idx])
                type = ks.Type[n][0]
                regid = lan.Id(n + str(m) + '_reg')
                reg = lan.TypeId([type], regid)
                writes.append(regid)
                assign = lan.Assignment(reg, sub)
                initstats.append(assign)

        stats.insert(0, lan.GroupCompound(initstats))

        # Replace the global Arefs with the register Arefs
        count = 0
        for i, n in enumerate(arrDict):
            for m in arrDict[n]:
                idx = m
                aref_new = writes[count]
                aref_old = ks.LoopArrays[n][idx]
                # Copying the internal data of the two arefs
                aref_old.name.name = aref_new.name
                aref_old.subscript = []
                count += 1

    def placeInReg3(self, ks, arrDict, insideList):
        """ Check if the arrayref is inside a loop and use a static
            array for the allocation of the registers
        """
        stats = ks.Kernel.statements
        initstats = []
        writes = []

        if not arrDict:
            return

        if not insideList:
            self.placeInReg2(ks, arrDict)
            return

        insideloop = insideList[0]

        if insideloop == '':
            print "placeInReg3 only works when the ArrayRef is inside a loop"
            print arrDict
            return

        # Add allocation of registers to the initiation stage
        for n in arrDict:
            lval = lan.TypeId([ks.Type[n][0]], \
                              lan.Id(n + '_reg[' + str(ks.UpperLimit[insideloop]) \
                                     + ']'))
            initstats.append(lval)

        # add the loop to the initiation stage
        loop = copy.deepcopy(ks.Loops[insideloop])
        loopstats = []
        # Exchange loop index
        loop.compound.statements = loopstats

        initstats.append(loop)

        # Create the loadings
        for i, n in enumerate(arrDict):
            for m in arrDict[n]:
                idx = m
                sub = copy.deepcopy(ks.LoopArrays[n][idx])
                regid = lan.ArrayRef(lan.Id(n + '_reg'), [lan.Id(insideloop)])
                writes.append(regid)
                assign = lan.Assignment(regid, sub)
                loopstats.append(assign)

        stats.insert(0, lan.GroupCompound(initstats))
        # Replace the global Arefs with the register Arefs
        count = 0
        for i, n in enumerate(arrDict):
            for m in arrDict[n]:
                idx = m
                aref_new = copy.deepcopy(writes[count])
                aref_old = ks.LoopArrays[n][idx]
                # Copying the internal data of the two arefs
                aref_old.name.name = aref_new.name.name
                aref_old.subscript = aref_new.subscript
                count += 1
