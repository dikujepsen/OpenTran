import lan
import copy


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
        self.types = dict()

    def visit_TypeId(self, node):
        name = node.name.name
        self.ids.add(name)
        self.types[name] = node.type

    def visit_ArrayTypeId(self, node):
        name = node.name.name
        self.ids.add(name)
        self.types[name] = copy.deepcopy(node.type)
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


class Ids(lan.NodeVisitor):
    """ Finds all unique IDs, excluding function IDs"""

    def __init__(self):
        self.ids = set()

    def visit_FuncDecl(self, node):
        if node.compound.statements:
            self.visit(node.arglist)

    def visit_Id(self, node):
        self.ids.add(node.name)


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
