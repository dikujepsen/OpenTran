import lan
import collect


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
            binop_di = _ArrayRefIndices(self.loop_index)
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
    def has_distinct(self):
        return (self.right - self.left) and (self.left - self.right)

    def visit_Id(self, node):
        name = node.name
        if name in self.indices:
            self.tmp.add(name)


class _ArrayRefIndices(_BinOpDistinctIndices):
    def __init__(self, indices):
        self.indices = indices
        self.tmp = set()


class NormArrayRef(lan.NodeVisitor):
    """ Normalizes subscripts to the form i * (width of j) + j
        or j + i * (width of j). Never (width of j) * i + j
    """

    def __init__(self, ast):
        self.loop_index = list()
        col_li = collect.LoopIndices()
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

        lan.NodeVisitor.current_parent = oldparent
