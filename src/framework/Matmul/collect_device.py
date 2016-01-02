import lan
import copy
import collect_loop as cl


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


def get_kernel(ast):
    par_dim = cl.get_par_dim(ast)
    fker = FindKernel(par_dim)
    fker.visit(ast)
    return fker.kernel


class FindDeviceArgs(lan.NodeVisitor):
    """ Finds the argument that we transfer from the C code
    to the device.
    """

    def __init__(self, arg_ids):
        self.argIds = arg_ids
        self.arglist = list()

    def visit_ArgList(self, node):
        for typeid_orig in node.arglist:
            typeid = copy.deepcopy(typeid_orig)
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


class FindIncludes(lan.NodeVisitor):
    def __init__(self):
        self.includes = list()

    def visit_Include(self, node):
        self.includes.append(node)


def get_includes(ast):
    find_includes = FindIncludes()
    find_includes.visit(ast)
    return find_includes.includes
