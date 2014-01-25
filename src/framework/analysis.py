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
            if n not in rw.Change:
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
