

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
        self.loop_index = list()
        # dict of the upper limit of the loop indices
        self.UpperLimit = dict()
        # dict of the lower limit of the loop indices
        self.LowerLimit = dict()
        # The number of dimensions of each array
        self.num_array_dims = dict()
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
        self.LoopArrays = dict()

    def __detect_loop_index(self, ast):
        loops = visitor.ForLoops()
        loops.visit(ast)
        self.for_loop_ast = loops.ast
        loop_indices = visitor.LoopIndices()
        loop_indices.visit(self.for_loop_ast)
        self.loop_index = loop_indices.index
        self.UpperLimit = loop_indices.end
        self.LowerLimit = loop_indices.start

    def normalize_subcript(self, ast):
        self.__detect_loop_index(ast)
        norm = visitor.Norm(self.loop_index)
        norm.visit(self.for_loop_ast)

    def init_original(self, ast):

        self.normalize_subcript(ast)
        arrays = visitor.Arrays(self.loop_index)
        arrays.visit(ast)

        for n in arrays.numIndices:
            if arrays.numIndices[n] == 2:
                arrays.numSubscripts[n] = 2
            elif arrays.numIndices[n] > 2:
                arrays.numSubscripts[n] = 1
            
        self.num_array_dims = arrays.numSubscripts

        self.IndexInSubscript = arrays.indexIds
        type_ids = visitor.TypeIds()
        type_ids.visit(self.for_loop_ast)

        type_ids2 = visitor.TypeIds()
        type_ids2.visit(ast)
        for n in type_ids.ids:
            type_ids2.dictIds.pop(n)
        self.Type = type_ids2.dictIds
        ids = visitor.Ids()
        ids.visit(ast)

        # print "typeIds.ids ", typeIds.ids
        # print "arrays.ids ", arrays.ids
        # print "ids.ids ", ids.ids
        other_ids = ids.ids - arrays.ids - type_ids.ids
        self.ArrayIds = arrays.ids - type_ids.ids
        self.NonArrayIds = other_ids

    def data_structures(self):
        print "self.index ", self.loop_index
        print "self.UpperLimit ", self.UpperLimit
        print "self.LowerLimit ", self.LowerLimit
        print "self.NumDims ", self.num_array_dims
        print "self.ArrayIds ", self.ArrayIds
        print "self.IndexInSubscript ", self.IndexInSubscript
        print "self.NonArrayIds ", self.NonArrayIds
        print "self.Type ", self.Type






           
        


