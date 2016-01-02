import lan
import collect_id as ci


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
        id_init = ci.Ids()
        id_init.visit(node.init)
        id_inc = ci.Ids()
        id_inc.visit(node.inc)
        self.index.extend(id_init.ids.intersection(id_inc.ids))
        self.depth += 1
        if self.depth < self.depth_limit:
            self.visit(node.compound)

    @property
    def grid_indices(self):
        return self.index


def get_grid_indices(ast, par_dim):
    loop_indices = LoopIndices(par_dim)
    loop_indices.visit(ast)
    return loop_indices.grid_indices


class LoopLimit(lan.NodeVisitor):
    """ Finds loop indices, the start and end values of the
        indices and creates a mapping from a loop index to
        the ForLoop AST node that is indexes.
    """

    def __init__(self):
        self.upper_limit = dict()
        self.lower_limit = dict()

    def visit_ForLoop(self, node):
        id_vis = ci.Ids()
        id_vis.visit(node.init)
        ids = list(id_vis.ids)

        self.visit(node.compound)
        try:
            self.upper_limit[ids[0]] = node.cond.rval.name
            self.lower_limit[ids[0]] = node.init.rval.value
        except AttributeError:
            self.upper_limit[ids[0]] = 'Unknown'
            self.lower_limit[ids[0]] = 'Unknown'


def get_loop_limits(ast):
    loop_limits = LoopLimit()
    loop_limits.visit(ast)
    return loop_limits.lower_limit, loop_limits.upper_limit


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


def get_inner_loops(ast, par_dim):
    find_inner_loops = FindInnerLoops(par_dim)
    find_inner_loops.collect(ast)
    return find_inner_loops.Loops


def get_inner_loops_indices(ast, par_dim):
    loops = get_inner_loops(ast, par_dim)
    return loops.keys()


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


class FindParDim(lan.NodeVisitor):
    def __init__(self):
        self.par_dim = 0

    def collect(self, ast):
        self.visit(ast)

    def visit_ParDim(self, node):
        self.par_dim = node.par_dim


def get_par_dim(ast):
    perfect_for_loop = FindPerfectForLoop()
    perfect_for_loop.visit(ast)
    find_par_dim = FindParDim()
    find_par_dim.collect(ast)
    if find_par_dim.par_dim is None:
        return perfect_for_loop.depth
    else:
        return find_par_dim.par_dim

