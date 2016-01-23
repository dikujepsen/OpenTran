import lan
import copy

opencl_builtins = ['CLK_LOCAL_MEM_FENCE']


class Ids(lan.NodeVisitor):
    """ Finds all unique IDs, excluding function IDs"""

    def __init__(self):
        self.ids = set()

    def visit_FuncDecl(self, node):
        if node.compound.statements:
            self.visit(node.arglist)

    def visit_Id(self, node):
        self.ids.add(node.name)


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


def get_non_array_ids(ast):
    non_array_ids = GlobalNonArrayIds()
    non_array_ids.visit(ast)
    return non_array_ids.ids


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
        self.local_na_ids.add(name)

    def visit_FuncDecl(self, node):
        self.visit(node.arglist)
        self.visit(node.compound)

    def visit_Id(self, node):
        name = node.name
        if name not in opencl_builtins:
            self.all_na_ids.add(name)

    @property
    def ids(self):
        return self.all_na_ids - self.local_na_ids - self.local_a_tids


class GlobalTypeIds(lan.NodeVisitor):
    """ Finds type Ids """

    def __init__(self):
        self.ids = set()
        self.types = dict()

    def collect(self, ast):
        self.visit(ast)

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
            # self.dictIds[name].append('*')
            pass

    def visit_ForLoop(self, node):
        self.visit(node.compound)

    def visit_Transpose(self, node):
        name = node.name.name
        self.types[name] = node.type


def get_types(ast):
    mytype_ids = GlobalTypeIds()
    mytype_ids.visit(ast)
    return mytype_ids.types


class FindKernelArgDefine(lan.NodeVisitor):
    def __init__(self):
        self.kernal_arg_ids = set()

    def visit_KernelArgDefine(self, node):
        name = node.name.name
        self.kernal_arg_ids.add(name)


def get_kernel_arg_defines(ast):
    kernel_arg_define = FindKernelArgDefine()
    kernel_arg_define.visit(ast)
    return kernel_arg_define.kernal_arg_ids


class FindLocalSwap(lan.NodeVisitor):
    def __init__(self):
        self.local_swap = dict()

    def visit_Stencil(self, node):
        name = node.name.name
        local_name = node.local_name.name
        self.local_swap[name] = local_name


def get_local_swap(ast):
    local_swap = FindLocalSwap()
    local_swap.visit(ast)
    return local_swap.local_swap


class FindProgramName(lan.NodeVisitor):
    def __init__(self):
        self.name = ""

    def visit_ProgramName(self, node):
        self.name = node.name


def get_program_name(ast):
    program_name = FindProgramName()
    program_name.visit(ast)
    return program_name.name
