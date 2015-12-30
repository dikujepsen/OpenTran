import lan
import copy
import collect_loop as cl


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


class IndicesInArrayRef(lan.NodeVisitor):
    def __init__(self):
        self.indices = list()
        self.tmp = set()
        self.indexIds = dict()

    def collect(self, ast, par_dim=2):
        col_li = cl.LoopIndices(par_dim)
        col_li.visit(ast)
        self.indices = col_li.grid_indices
        self.visit(ast)

    def visit_ArrayRef(self, node):
        name = node.name.name
        ari = _ArrayRefIndices(self.indices)
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


class ArrayNameToRef(lan.NodeVisitor):
    """ Finds array names to array refs """

    def __init__(self):
        self.ids = set()
        self.LoopArrays = dict()
        self.LoopArraysParent = dict()

    def visit_ArrayRef(self, node):
        name = node.name.name
        if name in self.LoopArrays:
            self.LoopArrays[name].append(node)
            self.LoopArraysParent[name].append(lan.NodeVisitor.current_parent)
        else:
            self.LoopArrays[name] = [node]
            self.LoopArraysParent[name] = [lan.NodeVisitor.current_parent]


class ArraySubscripts(lan.NodeVisitor):
    """ Finds array names to subscripts of array refs """

    def __init__(self):
        self.ids = set()
        self.Subscript = dict()
        self.subscript_no_id = dict()

    def collect(self, ast):
        self.visit(ast)

        self.subscript_no_id = copy.deepcopy(self.Subscript)
        for n in self.subscript_no_id.values():
            for m in n:
                for i, k in enumerate(m):
                    if isinstance(k, lan.Id):
                        m[i] = k.name
                    else:
                        m[i] = 'unknown'

    def visit_ArrayRef(self, node):
        name = node.name.name
        if name in self.Subscript:
            self.Subscript[name].append(node.subscript)
        else:
            self.Subscript[name] = [node.subscript]


class NumArrayDim(lan.NodeVisitor):
    """ Finds array Ids """

    def __init__(self, ast):
        self.loop_index = list()
        col_li = cl.LoopIndices()
        col_li.depth_limit = 99
        col_li.visit(ast)
        self.loop_index = col_li.index
        arrays_ids = GlobalArrayIds()
        arrays_ids.visit(ast)
        self.ArrayIds = arrays_ids.ids

        self.numSubscripts = dict()

    def visit_ArrayRef(self, node):
        name = node.name.name
        if name in self.ArrayIds:
            binop_di = _ArrayRefIndices(self.loop_index)

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


class FindReadWrite(lan.NodeVisitor):
    """ Returns a mapping of array to either
    'read'-only, 'write'-only, or 'readwrite'
    """

    def __init__(self, array_ids):
        self.ReadWrite = dict()
        self.ArrayIds = array_ids
        for n in self.ArrayIds:
            self.ReadWrite[n] = set()

    def visit_Assignment(self, node):
        find_read_pattern = _FindReadPattern(self.ArrayIds, self.ReadWrite, True)
        find_read_pattern.visit(node.lval)
        find_read_pattern = _FindReadPattern(self.ArrayIds, self.ReadWrite, False)
        find_read_pattern.visit(node.rval)


class _FindReadPattern(lan.NodeVisitor):
    """ Return whether the an array are read or written """

    def __init__(self, array_ids, read_write, left):
        self.ReadWrite = read_write
        self.ArrayIds = array_ids
        self.left = left

    def visit_ArrayRef(self, node):
        name = node.name.name
        if name in self.ArrayIds:
            if self.left:
                self.ReadWrite[name].add('write')
            else:
                self.ReadWrite[name].add('read')
            find_read_pattern = _FindReadPattern(self.ArrayIds, self.ReadWrite, False)
            for n in node.subscript:
                find_read_pattern.visit(n)


class _ArrayRefIndices(lan.NodeVisitor):
    def __init__(self, indices):
        self.indices = indices
        self.tmp = set()

    @property
    def loop_indices(self):
        return self.tmp

    @property
    def num_dims(self):
        return len(self.tmp)

    def visit_Id(self, node):
        name = node.name
        if name in self.indices:
            self.tmp.add(name)


class FindRefToLoopIndex(lan.NodeVisitor):
    """ Create a dict from array name to list of
    	arrayref list of loop indices that the arrayrefs are inside.
    """

    def __init__(self, par_dim=2):
        self.stack = list()
        self.RefToLoop = dict()
        self.GridIndices = list()
        self.par_dim = par_dim

    def collect(self, ast):
        col_li = cl.LoopIndices(self.par_dim)
        col_li.visit(ast)
        self.GridIndices = col_li.grid_indices
        self.visit(ast)

    def visit_ForLoop(self, node):
        name = node.init.lval.name.name
        if name not in self.GridIndices:
            self.stack.append(name)
        self.visit(node.init)
        self.visit(node.compound)
        if name not in self.GridIndices:
            self.stack.pop()

    def visit_ArrayRef(self, node):
        name = node.name.name
        try:
            self.RefToLoop[name].append(copy.deepcopy(self.stack))
        except KeyError:
            self.RefToLoop[name] = [copy.deepcopy(self.stack)]
