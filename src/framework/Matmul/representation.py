import lan
import collect


def print_dict_sorted(mydict):
    keys = sorted(mydict)

    entries = ""
    for key in keys:
        value = mydict[key]
        entries += "'" + key + "': " + value.__repr__() + ","

    return "{" + entries[:-1] + "}"


class Representation(lan.NodeVisitor):
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

    def __detect_loop_index(self, ast):
        col_li = collect.LoopIndices()
        col_li.visit(ast)
        self.loop_index = col_li.index
        ll = collect.LoopLimit()
        ll.visit(ast)
        self.UpperLimit = ll.upper_limit
        self.LowerLimit = ll.lower_limit

    def normalize_subcript(self, ast):
        self.__detect_loop_index(ast)

        naref = collect.NormArrayRef(ast)

        naref.visit(ast)

    def init_original(self, ast):
        self.normalize_subcript(ast)

        iiar = collect.IndicesInArrayRef(self.loop_index)
        iiar.visit(ast)
        self.IndexInSubscript = iiar.indexIds

        num_array_dim = collect.NumArrayDim(ast)
        num_array_dim.visit(ast)

        self.num_array_dims = num_array_dim.numSubscripts

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
