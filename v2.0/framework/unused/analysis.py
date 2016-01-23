import copy
from itertools import chain

import codegen.snippetgen
import lan
import lan.ast_buildingblock as ast_bb


class Analysis:
    """ Apply transformations to the original AST. Includes:
    1. Local Memory
    2. Stencil Local Memory
    3. Placing data in registers
    4. Transposing arrays
    5. Unrolling loops
    6. Adding defines
    7. Setting the number of dimensions to parallelize
    8. Setting the local work-group size
    9. Setting if we should read data back from the GPU
    10. Setting which kernel arguments changes
    """

    def __init__(self, rw, tf):
        # The rewriter
        self.rw = rw
        # The TRANSFORMER
        self.tf = tf
        self.PlaceInRegArgs = list()
        self.PlaceInRegCond = None
        self.PlaceInLocalArgs = list()
        self.PlaceInLocalCond = None

    def DefineArguments(self):
        """ Find all kernel arguments that can be defined
            at compilation time. Then defines them.
        """
        rw = self.rw
        # print "2131"
        # print rw.KernelArgs

        defines = list()
        for n in rw.KernelArgs:
            if len(rw.Type[n]) < 2:
                defines.append(n)

        # print defines
        # print "3123"
        self.tf.SetDefine(defines)

    def Transpose(self):
        """ Find all arrays that *should* be transposed
            and transpose them
        """
        rw = self.rw
        # self.SubscriptNoId

        # self.IdxToDim idx to dim number
        notranspose = set()
        transpose = set()
        maybetranspose = set()
        print rw.SubscriptNoId, "subnoid"
        print rw.IdxToDim, "idxtodim"
        for (n, sub) in rw.SubscriptNoId.items():
            # Every ArrayRef, Every list of subscripts
            for s in sub:
                if len(s) == 2:
                    if s[0] == rw.IdxToDim[0]:
                        transpose.add(n)
                    elif s[1] == rw.IdxToDim[0]:
                        notranspose.add(n)
                    elif rw.ParDim == 2 and s[0] == rw.IdxToDim[1]:
                        maybetranspose.add(n)
        print transpose, "transpose"
        for n in transpose:
            self.tf.transpose(n)

    def PlaceInReg(self):
        """ Find all array references that can be cached in registers.
        	Then rewrite the code in this fashion.
        """
        rw = self.rw
        optim = dict()
        for n in rw.RefToLoop:
            optim[n] = []
        insideloop = set()
        for n in rw.RefToLoop:
            if n in rw.WriteOnly:
                continue
            ## for (ref, sub) in (rw.RefToLoop[n], rw.SubscriptNoId[n]):
            ref1 = rw.RefToLoop[n]
            sub1 = rw.SubscriptNoId[n]

            for (ref, sub, i) in zip(ref1, sub1, range(len(ref1))):
                ## print rw.GridIndices, sub
                if set(rw.GridIndices) & set(sub):
                    outerloops = set(ref) - set(sub)
                    if outerloops:
                        insideloop |= set(sub) - set(rw.GridIndices)
                        optim[n].append(i)

                        ## print i, n, outerloops, rw.GridIndices, sub, \
                        ##   set(sub) - set(rw.GridIndices)

        insideloop = {k for k in insideloop if k in rw.Loops}
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
                lhs = lan.BinOp(lan.Id(rw.astrepr.UpperLimit[m]), '-', lan.Id(rw.astrepr.LowerLimit[m]))
            else:
                lhs = lan.Constant(1)
            self.PlaceInRegCond = lan.BinOp(lan.BinOp(lhs, '*', lan.Constant(numref)), '<', lan.Constant(40))

    def PlaceInLocalMemory(self):
        """ Find all array references that can be optimized
        	through the use of shared memory.
            Then rewrite the code in this fashion.
        """
        rw = self.rw
        tf = self.tf

        args = dict()
        loopindex = set()
        for k, v in rw.SubscriptNoId.items():
            for i, n in enumerate(v):
                if set(n) & set(rw.GridIndices) and \
                                set(n) & set(rw.Loops.keys()):
                    if rw.ParDim == 2:
                        args[k] = [i]
                        loopindex = loopindex.union(set(n) & set(rw.Loops.keys()))

        loopindex = list(loopindex)
        if args:
            ## print 'Local ' , args
            self.PlaceInLocalArgs.append(args)

        for m in loopindex:
            cond = lan.BinOp(lan.BinOp(lan.BinOp(lan.Id(rw.astrepr.UpperLimit[m]), '-', \
                                                 lan.Id(rw.astrepr.LowerLimit[m])), '%', \
                                       lan.Constant(rw.Local['size'][0])), '==', lan.Constant(0))
            self.PlaceInLocalCond = (cond)

    def GenerateKernels(self, ast, name, fileprefix):
        rw = self.rw
        tf = self.tf
        # Create base version and possible version with Local and
        # Register optimizations
        funcname = name + 'Base'

        if self.PlaceInRegArgs and self.PlaceInLocalArgs:
            raise Exception("""GenerateKernels: Currently unimplemented to perform
                                PlaceInReg and PlaceInLocal together from the analysis""")

        ss = codegen.snippetgen.SnippetGen(rw)

        ss.in_source_kernel(copy.deepcopy(ast), lan.Id('true'), filename=fileprefix + name + '/' + funcname + '.cl',
                            kernelstringname=funcname)
        for (arg, insideloop) in self.PlaceInRegArgs:
            funcname = name + 'PlaceInReg'
            tf.place_in_reg3(arg, list(insideloop))
            ss.in_source_kernel(copy.deepcopy(ast), lan.Id('true'), filename=fileprefix + name + '/' + funcname + '.cl',
                                kernelstringname=funcname)

        for arg in self.PlaceInLocalArgs:
            funcname = name + 'PlaceInLocal'
            tf.local_memory3(arg)
            ss.in_source_kernel(copy.deepcopy(ast), self.PlaceInLocalCond,
                                filename=fileprefix + name + '/' + funcname + '.cl', kernelstringname=funcname)

        MyCond = None
        if self.PlaceInLocalCond:
            MyCond = self.PlaceInLocalCond
        if self.PlaceInRegCond:
            MyCond = self.PlaceInRegCond

        if MyCond:
            name = rw.KernelStringStream[0]['name']
            func = ast_bb.EmptyFuncDecl(name, type=[])
            returnfunc1 = lan.Assignment(lan.Id('return'), func, op='')
            name = rw.KernelStringStream[1]['name']
            func = ast_bb.EmptyFuncDecl(name, type=[])
            returnfunc2 = lan.Assignment(lan.Id('return'), func, op='')
            ifthenelse = lan.IfThenElse(MyCond, \
                                        lan.Compound([returnfunc2]), lan.Compound([returnfunc1]))

            rw.IfThenElse = ifthenelse
        else:
            name = rw.KernelStringStream[0]['name']
            func = ast_bb.EmptyFuncDecl(name, type=[])
            returnfunc1 = lan.Assignment(lan.Id('return'), func, op='')
            rw.IfThenElse = returnfunc1
