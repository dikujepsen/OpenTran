import visitor
import collect


def print_dict_sorted(mydict):
    keys = sorted(mydict)

    entries = ""
    for key in keys:
        value = mydict[key]
        entries += "'" + key + "': " + value.__repr__() + ","

    return "{" + entries[:-1] + "}"


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

        # print arrays.ids

        for n in arrays.numIndices:
            if arrays.numIndices[n] == 2:
                arrays.numSubscripts[n] = 2
            elif arrays.numIndices[n] > 2:
                arrays.numSubscripts[n] = 1

        self.num_array_dims = arrays.numSubscripts

        self.IndexInSubscript = arrays.indexIds

        # print print_dict_sorted(self.Type)

        mytype_ids = collect.GlobalTypeIds()
        mytype_ids.visit(ast)
        # print print_dict_sorted(mytype_ids.dictIds)
        self.Type = mytype_ids.dictIds

        arrays_ids = collect.GlobalArrayIds()
        arrays_ids.visit(ast)
        self.ArrayIds = arrays_ids.ids
        # print arrays_ids.all_a_ids

        nonarray_ids = collect.GlobalNonArrayIds()
        nonarray_ids.visit(ast)
        self.NonArrayIds = nonarray_ids.ids
        # print nonarray_ids.ids




    def data_structures(self):
        print "self.index ", self.loop_index
        print "self.UpperLimit ", self.UpperLimit
        print "self.LowerLimit ", self.LowerLimit
        print "self.NumDims ", self.num_array_dims
        print "self.ArrayIds ", self.ArrayIds
        print "self.IndexInSubscript ", self.IndexInSubscript
        print "self.NonArrayIds ", self.NonArrayIds
        print "self.Type ", self.Type
