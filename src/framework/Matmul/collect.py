import lan
import copy
import visitor


class GlobalArrayIds(lan.NodeVisitor):
    def __init__(self):
        self.all_a_ids = set()
        self.local_a_ids = set()

    def visit_ArrayRef(self, node):
        name = node.name.name
        self.all_a_ids.add(name)
        for c_name, c in node.children():
            self.visit(c)

    def visit_ArrayTypeId(self, node):
        name = node.name.name
        self.local_a_ids.add(name)
        for c_name, c in node.children():
            self.visit(c)

    @property
    def ids(self):
        return self.all_a_ids - self.local_a_ids


class GlobalNonArrayIds(lan.NodeVisitor):
    def __init__(self):
        self.HasNotVisitedFirst = True
        self.gnai = _NonArrayIdsInLoop()

    def visit_ForLoop(self, node):
        if self.HasNotVisitedFirst:
            self.gnai.visit(node)
            self.HasNotVisitedFirst = False

    @property
    def ids(self):
        return self.gnai.ids


class _NonArrayIdsInLoop(lan.NodeVisitor):
    def __init__(self):
        self.all_na_ids = set()
        self.local_na_ids = set()
        self.local_a_tids = set()

    def visit_ArrayRef(self, node):
        for subnode in node.subscript:
            self.visit(subnode)

    def visit_ArrayTypeId(self, node):
        name = node.name.name
        self.local_a_tids.add(name)
        for subnode in node.subscript:
            self.visit(subnode)

    def visit_TypeId(self, node):
        name = node.name.name
        # if name == "hA":
        #     print self.current_parent
        self.local_na_ids.add(name)

    def visit_FuncDecl(self, node):
        self.visit(node.arglist)
        self.visit(node.compound)

    def visit_Id(self, node):
        name = node.name
        # if name == "db":
        #     print self.current_parent
        self.all_na_ids.add(name)

    @property
    def ids(self):
        return self.all_na_ids - self.local_na_ids - self.local_a_tids


class GlobalTypeIds(lan.NodeVisitor):
    """ Finds type Ids """

    def __init__(self):
        self.ids = set()
        self.dictIds = dict()

    def visit_TypeId(self, node):
        name = node.name.name
        self.ids.add(name)
        self.dictIds[name] = node.type

    def visit_ArrayTypeId(self, node):
        name = node.name.name
        self.ids.add(name)
        self.dictIds[name] = copy.deepcopy(node.type)
        if len(node.type) != 2:
            # "ArrayTypeId: Need to check, type of array is ", node.type
            ## self.dictIds[name].append('*')
            pass

    def visit_ForLoop(self, node):
        pass


class LoopIndices(lan.NodeVisitor):
    """ Finds loop indices, the start and end values of the
        indices and creates a mapping from a loop index to
        the ForLoop AST node that is indexes.
    """

    def __init__(self):
        self.index = list()
        self.depth_limit = 2
        self.depth = 0

    def visit_ForLoop(self, node):
        id_init = visitor.Ids()
        id_init.visit(node.init)
        id_inc = visitor.Ids()
        id_inc.visit(node.inc)
        self.index.extend(id_init.ids.intersection(id_inc.ids))
        self.depth += 1
        if self.depth < self.depth_limit:
            self.visit(node.compound)


class LoopLimit(lan.NodeVisitor):
    """ Finds loop indices, the start and end values of the
        indices and creates a mapping from a loop index to
        the ForLoop AST node that is indexes.
    """

    def __init__(self):
        self.upper_limit = dict()
        self.lower_limit = dict()

    def visit_ForLoop(self, node):
        IdVis = visitor.Ids()
        IdVis.visit(node.init)
        ids = list(IdVis.ids)

        self.visit(node.compound)
        try:
            self.upper_limit[ids[0]] = node.cond.rval.name
            self.lower_limit[ids[0]] = node.init.rval.value
        except AttributeError:
            self.upper_limit[ids[0]] = 'Unknown'
            self.lower_limit[ids[0]] = 'Unknown'


def _is_binop_plus(binop):
    return binop.op == '+'


def _is_binop_times(binop):
    return binop.op == '*'


class _NormBinOp(lan.NodeVisitor):
    def __init__(self, loop_index):
        self.loop_index = loop_index
        self.new_subscript = list()

    def visit_BinOp(self, node):

        p_binop = lan.NodeVisitor.current_parent

        if isinstance(node.lval, lan.Id) and isinstance(node.rval, lan.Id) \
                and _is_binop_times(node):

            if node.lval.name not in self.loop_index and \
                            node.rval.name in self.loop_index:
                (node.lval.name, node.rval.name) = \
                    (node.rval.name, node.lval.name)

        # print "p ", p_binop

        if isinstance(p_binop, lan.BinOp) and _is_binop_times(node) \
                and _is_binop_plus(p_binop):

            if isinstance(p_binop.lval, lan.Id) \
                    and isinstance(p_binop.rval.lval, lan.Id):
                (p_binop.lval, p_binop.rval) = (p_binop.rval, p_binop.lval)
            # print "n ", p_binop
            # binop_di = _BinOpDistinctIndices(self.loop_index, p_binop)
            binop_di = ArrayRefIndices(self.loop_index)
            binop_di.visit(p_binop)

            # if binop_di.has_distinct and isinstance(node.lval, lan.Id):
            if binop_di.num_dims > 0 and isinstance(node.lval, lan.Id):
                self.new_subscript = [lan.Id(p_binop.lval.lval.name, node.coord), p_binop.rval]

        oldparent = lan.NodeVisitor.current_parent
        lan.NodeVisitor.current_parent = node
        self.visit(node.lval)
        lan.NodeVisitor.current_parent = oldparent
        lan.NodeVisitor.current_parent = node
        self.visit(node.rval)
        lan.NodeVisitor.current_parent = oldparent


class _BinOpDistinctIndices(lan.NodeVisitor):
    """ Finds if there is two distinct loop indices
    	in an 1D array reference
    """

    def __init__(self, indices, binop):
        self.indices = indices
        self.yes = False
        self.binop = binop
        self.right = set()
        self.left = set()
        self.tmp = set()

        self.visit(binop.lval)
        self.right = self.tmp
        self.tmp = set()
        self.visit(binop.rval)
        self.left = self.tmp

    @property
    def num_dims(self):
        return len(self.tmp)

    @property
    def loop_indices(self):
        return self.tmp

    @property
    def has_distinct(self):
        return (self.right - self.left) and (self.left - self.right)

    def visit_Id(self, node):
        name = node.name
        if name in self.indices:
            self.tmp.add(name)


class ArrayRefIndices(_BinOpDistinctIndices):
    def __init__(self, indices):
        self.indices = indices
        self.tmp = set()


class IndicesInArrayRef(lan.NodeVisitor):
    def __init__(self, indices):
        self.indices = indices
        self.tmp = set()
        self.indexIds = dict()

    def visit_ArrayRef(self, node):
        name = node.name.name
        ari = ArrayRefIndices(self.indices)
        for n in node.subscript:
            ari.visit(n)
            self.tmp = self.tmp.union(ari.loop_indices)

        if name in self.indexIds:
            self.indexIds[name].update(self.indexIds[name].union(self.tmp))
        else:
            self.indexIds[name] = self.tmp
        self.tmp = set()

        for n in node.subscript:
            self.visit(n)


class NormArrayRef(lan.NodeVisitor):
    """ Normalizes subscripts to the form i * (width of j) + j
        or j + i * (width of j). Never (width of j) * i + j
    """

    def __init__(self, ast):
        self.loop_index = list()
        col_li = LoopIndices()
        col_li.visit(ast)
        self.loop_index = col_li.index

    def visit_ArrayRef(self, node):
        n_binop = _NormBinOp(self.loop_index)
        oldparent = lan.NodeVisitor.current_parent
        lan.NodeVisitor.current_parent = node
        if len(node.subscript) == 1:

            for subnode in node.subscript:
                n_binop.visit(subnode)
            if len(n_binop.new_subscript) > 0:
                node.subscript = n_binop.new_subscript
        # print node.subscript
        lan.NodeVisitor.current_parent = oldparent


class NumArrayDim(lan.NodeVisitor):
    """ Finds array Ids """

    def __init__(self, ast):
        self.loop_index = list()
        col_li = LoopIndices()
        col_li.depth_limit = 99
        col_li.visit(ast)
        self.loop_index = col_li.index

        self.numSubscripts = dict()

    def visit_ArrayRef(self, node):
        name = node.name.name

        binop_di = ArrayRefIndices(self.loop_index)

        if len(node.subscript) == 1:
            for n in node.subscript:
                binop_di.visit(n)
            self.numSubscripts[name] = max(binop_di.num_dims, 1)
        else:
            self.numSubscripts[name] = len(node.subscript)
        if self.numSubscripts[name] > 2:
            self.numSubscripts[name] = 1

        for n in node.subscript:
            self.visit(n)
