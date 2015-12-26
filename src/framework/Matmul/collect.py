import lan
import copy


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

    def __init__(self, depth_limit=2):
        self.index = list()
        self.depth_limit = depth_limit
        self.depth = 0

    def visit_ForLoop(self, node):
        id_init = Ids()
        id_init.visit(node.init)
        id_inc = Ids()
        id_inc.visit(node.inc)
        self.index.extend(id_init.ids.intersection(id_inc.ids))
        self.depth += 1
        if self.depth < self.depth_limit:
            self.visit(node.compound)

    @property
    def grid_indices(self):
        return self.index


class LoopLimit(lan.NodeVisitor):
    """ Finds loop indices, the start and end values of the
        indices and creates a mapping from a loop index to
        the ForLoop AST node that is indexes.
    """

    def __init__(self):
        self.upper_limit = dict()
        self.lower_limit = dict()

    def visit_ForLoop(self, node):
        IdVis = Ids()
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


class ArrayNameToRef(lan.NodeVisitor):
    """ Finds array names to array refs """

    def __init__(self):
        self.ids = set()
        self.LoopArrays = dict()

    def visit_ArrayRef(self, node):
        name = node.name.name
        if name in self.LoopArrays:
            self.LoopArrays[name].append(node)
        else:
            self.LoopArrays[name] = [node]


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
        col_li = LoopIndices()
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


class GenArrayDimNames(object):
    def __init__(self):
        self.num_array_dims = dict()
        self.ArrayIdToDimName = dict()

    def collect(self, ast):
        num_array_dim = NumArrayDim(ast)
        num_array_dim.visit(ast)
        self.num_array_dims = num_array_dim.numSubscripts

        for array_name, num_dims in num_array_dim.numSubscripts.items():
            tmp = list()
            for i in xrange(num_dims):
                tmp.append('hst_ptr' + array_name + '_dim' + str(i + 1))

            self.ArrayIdToDimName[array_name] = tmp


class FindReadWrite(lan.NodeVisitor):
    """ Returns a mapping of array to either
    'read'-only, 'write'-only, or 'readwrite'
    """

    def __init__(self, ArrayIds):
        self.ReadWrite = dict()
        self.ArrayIds = ArrayIds
        for n in self.ArrayIds:
            self.ReadWrite[n] = set()

    def visit_Assignment(self, node):
        findReadPattern = FindReadPattern(self.ArrayIds, self.ReadWrite, True)
        findReadPattern.visit(node.lval)
        findReadPattern = FindReadPattern(self.ArrayIds, self.ReadWrite, False)
        findReadPattern.visit(node.rval)


class FindReadPattern(lan.NodeVisitor):
    """ Return whether the an array are read or written """

    def __init__(self, ArrayIds, ReadWrite, left):
        self.ReadWrite = ReadWrite
        self.ArrayIds = ArrayIds
        self.left = left

    def visit_ArrayRef(self, node):
        name = node.name.name
        if name in self.ArrayIds:
            if self.left:
                self.ReadWrite[name].add('write')
            else:
                self.ReadWrite[name].add('read')
            findReadPattern = FindReadPattern(self.ArrayIds, self.ReadWrite, False)
            for n in node.subscript:
                findReadPattern.visit(n)


class FindPerfectForLoop(lan.NodeVisitor):
    """ Performs simple checks to decide if we have 1D or 2D
    parallelism, i.e. if we have a perfect loops nest of size one
    or two.
    """

    def __init__(self):
        self.depth = 0
        self.ast = None
        self.inner = None

    def visit_ForLoop(self, node):
        self.ast = node
        self.inner = node
        self.depth += 1
        loopstats = node.compound.statements
        if len(loopstats) == 1:
            if isinstance(loopstats[0], lan.ForLoop):
                self.depth += 1
                self.inner = loopstats[0]

    @property
    def outer(self):
        return self.ast


class FindKernel(lan.NodeVisitor):
    """ Performs simple checks to decide if we have 1D or 2D
    parallelism, i.e. if we have a perfect loops nest of size one
    or two.
    """

    def __init__(self, depth=2):
        self.depth = depth
        self.kernel = None

    def visit_ForLoop(self, node):
        if self.depth > 0:
            self.kernel = node.compound
            self.depth -= 1
            if len(self.kernel.statements) == 1:
                if isinstance(self.kernel.statements[0], lan.ForLoop):
                    if self.depth > 0:
                        self.depth -= 1
                        self.kernel = self.kernel.statements[0].compound


class GenIdxToDim(object):
    def __init__(self):
        self.IdxToDim = dict()

    def collect(self, ast, par_dim=2):
        col_li = LoopIndices(par_dim)
        col_li.visit(ast)

        grid_indices = col_li.grid_indices
        for i, n in enumerate(reversed(grid_indices)):
            self.IdxToDim[i] = n


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
        col_li = LoopIndices(self.par_dim)
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


class FindInnerLoops(lan.NodeVisitor):
    """ Finds loop indices, the start and end values of the
        indices and creates a mapping from a loop index to
        the ForLoop AST node that is indexes.
    """

    def __init__(self, par_dim=2):
        self.Loops = dict()
        self.par_dim = par_dim
        self.GridIndices = list()

    def collect(self, ast):
        col_li = LoopIndices(self.par_dim)
        col_li.visit(ast)
        self.GridIndices = col_li.grid_indices
        self.visit(ast)

    def visit_ForLoop(self, node):
        name = node.init.lval.name.name
        if name not in self.GridIndices:
            self.Loops[name] = node

        self.visit(node.compound)


class GenKernelArgs(object):
    def __init__(self):
        self.kernel_args = dict()

    def collect(self, ast, par_dim=2):
        arrays_ids = GlobalArrayIds()
        arrays_ids.visit(ast)
        array_ids = arrays_ids.ids
        # print self.ArrayIds

        nonarray_ids = GlobalNonArrayIds()
        nonarray_ids.visit(ast)
        non_array_ids = nonarray_ids.ids

        mytype_ids = GlobalTypeIds()
        mytype_ids.visit(ast)
        # print print_dict_sorted(mytype_ids.dictIds)
        types = mytype_ids.dictIds

        gen_removed_ids = GenRemovedIds()
        gen_removed_ids.collect(ast, par_dim)
        removed_ids = gen_removed_ids.removed_ids
        arg_ids = non_array_ids.union(array_ids) - removed_ids

        gen_array_dimnames = GenArrayDimNames()
        gen_array_dimnames.collect(ast)
        num_array_dims = gen_array_dimnames.num_array_dims
        arrayid_to_dimname = gen_array_dimnames.ArrayIdToDimName

        for n in arg_ids:
            tmplist = [n]
            try:
                if num_array_dims[n] == 2:
                    tmplist.append(arrayid_to_dimname[n][0])
            except KeyError:
                pass
            for m in tmplist:
                self.kernel_args[m] = types[m]


class Ids(lan.NodeVisitor):
    """ Finds all unique IDs, excluding function IDs"""

    def __init__(self):
        self.ids = set()

    def visit_FuncDecl(self, node):
        if node.compound.statements:
            self.visit(node.arglist)

    def visit_Id(self, node):
        self.ids.add(node.name)


class GenRemovedIds(object):
    def __init__(self):
        self.removed_ids = set()

    def collect(self, ast, par_dim=2):
        col_li = LoopIndices(par_dim)
        col_li.visit(ast)

        grid_indices = col_li.grid_indices

        col_loop_limit = LoopLimit()
        col_loop_limit.visit(ast)
        upper_limit = col_loop_limit.upper_limit

        upper_limits = set(upper_limit[i] for i in grid_indices)

        find_kernel = FindKernel(par_dim)
        find_kernel.visit(ast)

        ids_still_in_kernel = Ids()
        ids_still_in_kernel.visit(find_kernel.kernel)
        self.removed_ids = upper_limits - ids_still_in_kernel.ids


class FindDeviceArgs(lan.NodeVisitor):
    """ Finds the argument that we transfer from the C code
    to the device.
    """

    def __init__(self, argIds):
        self.argIds = argIds
        self.arglist = list()

    def visit_ArgList(self, node):
        for typeid in node.arglist:
            if isinstance(typeid, lan.TypeId):
                if typeid.name.name in self.argIds:
                    self.argIds.remove(typeid.name.name)
                    if len(typeid.type) == 2:
                        if typeid.type[1] == '*':
                            typeid.type.insert(0, '__global')
                    self.arglist.append(typeid)

class FindFunction(lan.NodeVisitor):
    """ Finds the typeid of the kernel function """

    def __init__(self):
        self.typeid = None

    def visit_FuncDecl(self, node):
        self.visit_TypeId(node.typeid)

    def visit_TypeId(self, node):
        self.typeid = node
