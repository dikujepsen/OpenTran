

import visitor

class Representation(visitor.NodeVisitor):
    """ Class for rewriting of the original AST. Includes:
    1. the initial small rewritings,
    2. transformation into our representation,
    3. transforming from our representation to C-executable code,
    4. creating our representation of the device kernel code,
    5. creating a C-executable kernel code,
    6. Creating the host code (boilerplate code) 
    """

    
    def __init__(self):
        # List of loop indices
        self.index = list()
        # dict of the upper limit of the loop indices
        self.UpperLimit = dict()
        # dict of the lower limit of the loop indices
        self.LowerLimit = dict()
        # The number of dimensions of each array
        self.NumDims = dict()
        # The Ids of arrays, or pointers
        self.ArrayIds = set()
        # The indices that appear in the subscript of each array
        self.IndexInSubscript = dict()
        # All Ids that are not arrays, or pointers
        self.NonArrayIds = set()
        # The types of the arguments for the kernel
        self.Type = dict()
        # Holds includes for the kernel
        self.Includes = list()

        
    def initOriginal(self, ast):
        loops = visitor.ForLoops()
        loops.visit(ast)
        forLoopAst = loops.ast
        loopIndices = visitor.LoopIndices()
        loopIndices.visit(forLoopAst)
        self.index = loopIndices.index
        self.UpperLimit = loopIndices.end
        self.LowerLimit = loopIndices.start

        norm = visitor.Norm(self.index)
        norm.visit(forLoopAst)
        arrays = visitor.Arrays(self.index)
        arrays.visit(ast)

        for n in arrays.numIndices:
            if arrays.numIndices[n] == 2:
                arrays.numSubscripts[n] = 2
            elif arrays.numIndices[n] > 2:
                arrays.numSubscripts[n] = 1
            
        self.NumDims = arrays.numSubscripts
        self.IndexInSubscript = arrays.indexIds
        typeIds = visitor.TypeIds()
        typeIds.visit(loops.ast)

        
        typeIds2 = visitor.TypeIds()
        typeIds2.visit(ast)
        for n in typeIds.ids:
            typeIds2.dictIds.pop(n)
        self.Type = typeIds2.dictIds
        ids = visitor.Ids()
        ids.visit(ast)

        # print "typeIds.ids ", typeIds.ids
        # print "arrays.ids ", arrays.ids
        # print "ids.ids ", ids.ids
        otherIds = ids.ids - arrays.ids - typeIds.ids
        self.ArrayIds = arrays.ids - typeIds.ids
        self.NonArrayIds = otherIds


    def DataStructures(self):
        print "self.index ", self.index
        print "self.UpperLimit ", self.UpperLimit
        print "self.LowerLimit ", self.LowerLimit
        print "self.NumDims ", self.NumDims
        print "self.ArrayIds ", self.ArrayIds
        print "self.IndexInSubscript ", self.IndexInSubscript
        print "self.NonArrayIds ", self.NonArrayIds
        print "self.Type ", self.Type






           
        


