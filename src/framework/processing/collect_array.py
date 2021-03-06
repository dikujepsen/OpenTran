import lan
import copy
import collect_loop as cl
import collect_gen as cg


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


class TransposableArrayIds(lan.NodeVisitor):
    def __init__(self):
        self.trans_a_hst_ids = set()
        self.trans_a_base_ids = set()
        self.a_name_swap = dict()
        self._host_ids = dict()

    def collect(self, ast):
        # self._host_ids = cg.gen_host_ids(ast)
        self.visit(ast)

    def visit_Transpose(self, node):
        name = node.name.name
        base_name = node.base_name.name
        hst_name = node.hst_name.name
        self.trans_a_hst_ids.add(name)
        self.trans_a_base_ids.add(base_name)

        self.a_name_swap[hst_name] = name

    @property
    def trans_ids(self):
        return self.trans_a_hst_ids

    @property
    def base_ids(self):
        return self.trans_a_base_ids

    @property
    def array_name_swap(self):
        return self.a_name_swap


def get_transposable_array_ids(ast):
    transposable_array_ids = TransposableArrayIds()
    transposable_array_ids.collect(ast)
    return transposable_array_ids.trans_ids


def get_transposable_base_ids(ast):
    transposable_array_ids = TransposableArrayIds()
    transposable_array_ids.collect(ast)
    return transposable_array_ids.base_ids


def get_host_array_name_swap(ast):
    transposable_array_ids = TransposableArrayIds()
    transposable_array_ids.collect(ast)
    return transposable_array_ids.a_name_swap


def get_array_dim_swap(ast):
    transposable_array_ids = TransposableArrayIds()
    transposable_array_ids.collect(ast)
    gen_array_dim_names = cg.GenArrayDimNames()
    gen_array_dim_names.collect(ast)
    array_id_to_dim_name = gen_array_dim_names.ArrayIdToDimName
    array_dim_swap = dict()
    base_ids = get_transposable_base_ids(ast)

    for arr_name in base_ids:
        dim_name = array_id_to_dim_name[arr_name]
        array_dim_swap[dim_name[0]] = dim_name[1]

    return array_dim_swap


class IndicesInArrayRef(lan.NodeVisitor):
    def __init__(self):
        self.indices = list()
        self.tmp = set()
        self.indexIds = dict()

    def collect(self, ast):
        par_dim = cl.get_par_dim(ast)
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


def get_indices_in_array_ref(ast):
    indices_in_array_ref = IndicesInArrayRef()
    indices_in_array_ref.collect(ast)
    return indices_in_array_ref.indexIds


class ArrayNameToRef(lan.NodeVisitor):
    """ Finds array names to array refs """

    def __init__(self):
        self.ids = set()
        self.LoopArrays = dict()
        self.LoopArraysParent = dict()

    def collect(self, ast):
        self.visit(ast)

    def visit_ArrayRef(self, node):
        name = node.name.name
        if 'isReg' not in node.extra:
            if name in self.LoopArrays:
                self.LoopArrays[name].append(node)
                self.LoopArraysParent[name].append(lan.NodeVisitor.current_parent)
            else:
                self.LoopArrays[name] = [node]
                self.LoopArraysParent[name] = [lan.NodeVisitor.current_parent]


def get_loop_arrays(ast):
    arr_to_ref = ArrayNameToRef()
    arr_to_ref.collect(ast)
    return arr_to_ref.LoopArrays


def get_loop_arrays_parent(ast):
    arr_to_ref = ArrayNameToRef()
    arr_to_ref.collect(ast)
    return arr_to_ref.LoopArraysParent


class ArraySubscripts(lan.NodeVisitor):
    """ Finds array names to subscripts of array refs """

    def __init__(self):
        self.ids = set()
        self.Subscript = dict()

    def collect(self, ast):
        self.visit(ast)

    def visit_ArrayRef(self, node):
        name = node.name.name
        if name in self.Subscript:
            self.Subscript[name].append(node.subscript)
        else:
            self.Subscript[name] = [node.subscript]


def get_subscript(ast):
    arr_subs = ArraySubscripts()
    arr_subs.collect(ast)
    return arr_subs.Subscript


def get_subscript_no_id(ast):
    subscript = get_subscript(ast)
    subscript_no_id = copy.deepcopy(subscript)
    for n in subscript_no_id.values():
        for m in n:
            for i, k in enumerate(m):
                if isinstance(k, lan.Id):
                    m[i] = k.name
                else:
                    m[i] = 'unknown'
    return subscript_no_id


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


def get_num_array_dims(ast):
    num_array_dim = NumArrayDim(ast)
    num_array_dim.visit(ast)
    return num_array_dim.numSubscripts


def get_array_ids(ast):
    arrays_ids = GlobalArrayIds()
    arrays_ids.visit(ast)
    return arrays_ids.ids


class FindReadWrite(lan.NodeVisitor):
    """ Returns a mapping of array to either
    'read'-only, 'write'-only, or 'readwrite'
    """

    def __init__(self):
        self.ReadWrite = dict()
        self.ArrayIds = set()

    def collect(self, ast):
        self.ArrayIds = get_array_ids(ast)
        for n in self.ArrayIds:
            self.ReadWrite[n] = set()
        self.visit(ast)

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

    def __init__(self):
        self.stack = list()
        self.RefToLoop = dict()
        self.GridIndices = list()

    def collect(self, ast):
        self.GridIndices = cl.get_grid_indices(ast)
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


def get_ref_to_loop(ast):
    find_ref_to_loop_index = FindRefToLoopIndex()
    find_ref_to_loop_index.collect(ast)
    return find_ref_to_loop_index.RefToLoop


def get_read_write(ast):
    find_read_write = FindReadWrite()
    find_read_write.collect(ast)
    return find_read_write.ReadWrite


def get_read_only(ast):
    read_write = get_read_write(ast)
    read_only = list()
    for n in read_write:
        io_set = read_write[n]
        if len(io_set) == 1:
            if 'read' in io_set:
                read_only.append(n)
    return read_only


def get_write_only(ast):
    read_write = get_read_write(ast)
    write_only = list()
    for n in read_write:
        io_set = read_write[n]
        if len(io_set) == 1:
            if 'write' in io_set:
                write_only.append(n)
    return write_only


class LocalMemArrayIdToDimName(lan.NodeVisitor):
    def __init__(self):
        self.ArrayIdToDimName = dict()

    def visit_Stencil(self, node):
        name = node.local_name.name
        self.ArrayIdToDimName[name] = node.size

    def visit_Block(self, node):
        name = node.name.name
        self.ArrayIdToDimName[name] = node.size
