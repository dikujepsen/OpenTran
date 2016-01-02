import lan
import copy
import collect_loop as cl
import collect_array as ca
import collect_id as ci


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

    def __init__(self):
        self.argIds = set()
        self.arglist = list()

    def collect(self, ast):
        arrays_ids = ca.get_array_ids(ast)
        non_array_ids = ci.get_non_array_ids(ast)
        self.argIds = arrays_ids.union(non_array_ids)
        self.visit(ast)

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


def get_devices_arg_list(ast):
    find_device_args = FindDeviceArgs()

    find_device_args.collect(ast)

    return find_device_args.arglist


class FindFunction(lan.NodeVisitor):
    """ Finds the typeid of the kernel function """

    def __init__(self):
        self.typeid = None

    def visit_FuncDecl(self, node):
        self.visit_TypeId(node.typeid)

    def visit_TypeId(self, node):
        self.typeid = node


def _get_kernel_base_name(ast):
    find_function = FindFunction()
    find_function.visit(ast)
    dev_func_type_id = find_function.typeid
    kernel_name = dev_func_type_id.name.name
    return kernel_name


def get_kernel_name(ast):
    return _get_kernel_base_name(ast) + 'Kernel'


def get_work_size(ast):
    kernel_name = _get_kernel_base_name(ast)
    work_size = dict()
    work_size['local'] = kernel_name + '_local_worksize'
    work_size['global'] = kernel_name + '_global_worksize'
    work_size['offset'] = kernel_name + '_global_offset'
    return work_size


def get_dev_id(ast):
    array_ids = ca.get_array_ids(ast)
    dev_ids = dict()
    for n in array_ids:
        dev_ids[n] = 'dev_ptr' + n

    return dev_ids


def get_dev_func_id(ast):
    find_function = FindFunction()
    find_function.visit(ast)
    dev_func_type_id = find_function.typeid
    dev_func_id = dev_func_type_id.name.name
    return dev_func_id


class FindIncludes(lan.NodeVisitor):
    def __init__(self):
        self.includes = list()

    def visit_Include(self, node):
        self.includes.append(node)


def get_includes(ast):
    find_includes = FindIncludes()
    find_includes.visit(ast)
    return find_includes.includes
