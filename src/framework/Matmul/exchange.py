import lan


class ExchangeId(lan.NodeVisitor):
    """ Exchanges the Ids that we parallelize with the threadids,
    (or whatever is given in idMap)
    """

    def __init__(self, idMap):
        self.idMap = idMap

    def visit_Id(self, node):
        if node.name in self.idMap:
            node.name = self.idMap[node.name]


class ExchangeIdWithBinOp(lan.NodeVisitor):
    """ Exchanges the Ids that we parallelize with the threadids,
    (or whatever is given in idMap)
    """

    def __init__(self, idMap):
        self.idMap = idMap

    def visit_Assignment(self, node):
        if isinstance(node.rval, lan.Id):
            try:
                node.rval = self.idMap[node.lval.name]
                self.visit(node.lval)
                return
            except KeyError:
                pass
        for c_name, c in node.children():
            self.visit(c)

    def visit_ArrayRef(self, node):
        for i, s in enumerate(node.subscript):
            if isinstance(s, lan.Id):
                try:
                    node.subscript[i] = self.idMap[s.name]
                except KeyError:
                    pass

    def visit_BinOp(self, node):
        if isinstance(node.lval, lan.Id):
            try:
                node.lval = self.idMap[node.lval.name]
                self.visit(node.rval)
                return
            except KeyError:
                pass

        if isinstance(node.rval, lan.Id):
            try:
                node.rval = self.idMap[node.rval.name]
                self.visit(node.lval)
                return
            except KeyError:
                pass
        for c_name, c in node.children():
            self.visit(c)


class ExchangeIndices(lan.NodeVisitor):
    """ Exchanges the indices that we parallelize with the threadids,
    (or whatever is given in idMap)
    ARGS: idMap: a dictionary of Id changes
    	  arrays: A list/set of array names that we change
    """

    def __init__(self, idMap, arrays):
        self.idMap = idMap
        self.arrays = arrays

    def visit_ArrayRef(self, node):
        if node.name.name in self.arrays:
            exchangeId = ExchangeId(self.idMap)
            for n in node.subscript:
                exchangeId.visit(n)


class ExchangeTypes(lan.NodeVisitor):
    """ Exchanges the size_t to unsigned for every TypeId
    """

    def __init__(self):
        pass

    def visit_TypeId(self, node):
        if node.type:
            if node.type[0] == 'size_t':
                node.type[0] = 'unsigned'


class ExchangeArrayId(lan.NodeVisitor):
    """ Exchanges the id of arrays in ArrayRefs with
    what is given in idMap)
    """

    def __init__(self, id_map):
        self.id_map = id_map

    def visit_ArrayRef(self, node):
        try:
            if node.extra['localMemory']:
                return
        except KeyError:
            pass

        try:
            node.name.name = self.id_map[node.name.name]
        except KeyError:
            pass


class RewriteArrayRef(lan.NodeVisitor):
    """ Rewrites the arrays references of form A[i][j] to
    A[i * JDIMSIZE + j]
    """

    def __init__(self, num_array_dims, arrayid_to_dim_name):
        self.ArrayIdToDimName = arrayid_to_dim_name
        self.num_array_dims = num_array_dims

    def visit_ArrayRef(self, node):
        n = node.name.name
        try:
            if len(node.subscript) == 2:

                leftbinop = lan.BinOp(node.subscript[0], '*', lan.Id(self.ArrayIdToDimName[n][0]))
                # Id on first dimension
                topbinop = lan.BinOp(leftbinop, '+', node.subscript[1])
                node.subscript = [topbinop]
        except KeyError:
            pass


class ExchangeArrayIdWithId(lan.NodeVisitor):
    """ Exchanges the array_ref with var_id in ast
    """

    def __init__(self, array_ref, var_id):
        super(ExchangeArrayIdWithId, self).__init__()
        self.array_ref = array_ref
        self.var_id = var_id

    def visit_ArrayRef(self, node):
        if node == self.array_ref:
            setattr(lan.NodeVisitor.current_parent, self.current_child, self.var_id)


