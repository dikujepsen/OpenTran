import lan
import copy
import stringstream
import exchange
import collect_device as cd
import collect_gen as cg
import collect_boilerplate_info as cbi
import collect_id as ci
import collect_array as ca


def print_dict_sorted(mydict):
    keys = sorted(mydict)

    entries = ""
    for key in keys:
        value = mydict[key]
        entries += "'" + key + "': " + value.__repr__() + ","

    return "{" + entries[:-1] + "}"


class SnippetGen(object):
    def __init__(self):
        self.KernelStringStream = list()
        self.par_dim = None
        self.ast = None

        self.LoopArrays = dict()

        # new
        self.kernel_args = dict()

    def set_datastructure(self,
                          par_dim,
                          ast):
        self.ast = ast
        self.par_dim = par_dim
        fai = cbi.FindLoopArrays()
        fai.collect(ast)
        self.LoopArrays = fai.loop_arrays
        self.kernel_args = cg.get_kernel_args(ast)

    def in_source_kernel(self, ast, cond, filename, kernelstringname):
        self.rewrite_to_device_c_release(ast)

        ssprint = stringstream.SSGenerator()

        ssprint.create_kernel_string_stream(ast, kernelstringname, filename=filename)
        self.KernelStringStream.append({'name': kernelstringname,
                                        'ast': ssprint.newast,
                                        'cond': cond})

    def rewrite_to_device_c_release(self, ast):
        arglist = self._create_arg_list()
        # print arglist

        self._swap_local_array_id()

        my_kernel = self._create_kernel()

        typeid = self._create_function_name()

        newast = NewAST()
        includes = cd.get_includes(ast)
        newast.add_list_statement(copy.deepcopy(includes))

        if self._arg_has_type_double(arglist):
            newast.enable_double_precision()

        newast.add_statement(lan.FuncDecl(typeid, lan.ArgList(arglist), my_kernel))
        ast.ext = list()
        ast.ext.append(newast.ast)

    def _create_arg_list(self):
        arglist = list()

        for n in sorted(self.kernel_args):
            kernel_type = copy.deepcopy(self.kernel_args[n])
            if kernel_type[0] == 'size_t':
                kernel_type[0] = 'unsigned'
            if len(kernel_type) == 2:
                kernel_type.insert(0, '__global')
            arglist.append(lan.TypeId(kernel_type, lan.Id(n)))

        return arglist

    def _swap_local_array_id(self):
        local_swap = ci.get_local_swap(self.ast)
        exchange_array_id = exchange.ExchangeArrayId(local_swap)

        for n in self.LoopArrays.values():
            for m in n:
                exchange_array_id.visit(m)

    def _create_kernel(self):
        my_kernel = copy.deepcopy(cd.get_kernel(self.ast))
        num_array_dims = ca.get_num_array_dims(self.ast)
        array_id_to_dim_name = cg.get_array_id_to_dim_name(self.ast)
        rewrite_array_ref = exchange.RewriteArrayRef(num_array_dims,
                                                     array_id_to_dim_name)
        rewrite_array_ref.visit(my_kernel)

        idx_to_thread_id = cg.GenIdxToThreadId()
        idx_to_thread_id.collect(self.ast, self.par_dim)
        index_to_thread_id = idx_to_thread_id.IndexToThreadId
        exchange_indices = exchange.ExchangeId(index_to_thread_id)
        exchange_indices.visit(my_kernel)

        exchange_types = exchange.ExchangeTypes()
        exchange_types.visit(my_kernel)

        return my_kernel

    def _create_function_name(self):
        find_function = cd.FindFunction()
        find_function.visit(self.ast)
        dev_func_type_id = find_function.typeid

        typeid = copy.deepcopy(dev_func_type_id)
        typeid.type.insert(0, '__kernel')

        return typeid

    def _arg_has_type_double(self, arglist):
        retval = False
        for n in arglist:
            if (len(n.type) == 3 and n.type[1] == 'double') \
                    or (len(n.type) != 3 and n.type[0] == 'double'):
                retval = True
        return retval


class NewAST(object):
    def __init__(self):
        self.ext = list()
        self.ast = lan.FileAST(self.ext)

    def add_list_statement(self, statement):
        for stat in statement:
            self.ext.append(stat)

    def add_statement(self, statement):
        self.ext.append(statement)

    def enable_double_precision(self):
        self.ext.insert(0, lan.Compound([lan.Id("#pragma OPENCL EXTENSION cl_khr_fp64: enable")]))
