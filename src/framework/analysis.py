import copy
import os
from visitor import *
from stringstream import *


class Analysis():
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
        # The transformer
        self.tf = tf

    def DefineArguments(self):
        """ Find all kernel arguments that can be defined
        	at compilation time. Then defines them.
        """
        rw = self.rw
        defines = list()
        for n in rw.KernelArgs:
            if n not in rw.Change and \
                len(rw.Type[n]) < 2:
                defines.append(n)

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
        for n in rw.SubscriptNoId: # Every ArrayRef
            for lsub in rw.SubscriptNoId[n]: # Every list of subscripts
                if len(lsub) == 2:
                    if lsub[0] == rw.IdxToDim[0]:
                        transpose.add(n)                    
                    elif lsub[1] == rw.IdxToDim[0]:
                        notranspose.add(n)
                    elif rw.ParDim == 2 and lsub[0] == rw.IdxToDim[1]:
                        maybetranspose.add(n)     
        ## print notranspose
        ## print transpose
        ## print maybetranspose - notranspose - transpose
        for n in transpose:
            self.tf.transpose(n)


    def PlaceInReg(self):
        """ Find all array references that can be cached in registers.
        	Then rewrite the code in this fashion.
        """
        rw = self.rw
        tf = self.tf
		# subscriptnoid, RefToLoop
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
    
                        
        ## print optim
        insideloop = {k for k in insideloop if k in rw.Loops}
        if len(insideloop) > 1:
            print """ PlaceInReg: array references was inside
    				  two loops. No optimization """
            return
        ## print insideloop
        tf.placeInReg3({k : v for k, v in optim.items() if v}, list(insideloop))
 
    def PlaceInLocalMemory(self):
        """ Find all array references that can be optimized
        	through the use of shared memory.
            Then rewrite the code in this fashion.
        """
        rw = self.rw
        tf = self.tf

        for k, v in rw.SubscriptNoId.items():
            for n in v:
                if set(n) & set(rw.GridIndices) and \
                    set(n) & set(rw.Loops.keys()):
                    if rw.ParDim == 2:
                        print k , n
                
        
