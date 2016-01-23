import lan
import boilerplatebase
import collect_gen as cg
import collect_id as ci
import collect_device as cd
import ast_buildingblock as ast_bb


class KernelArgs(boilerplatebase.BoilerplateBase):
    def __init__(self, ast, file_ast):
        super(KernelArgs, self).__init__(ast, file_ast)

    def set_kernel_args(self):

        dev_func_id = cd.get_dev_func_id(self.ast)

        set_arguments_kernel = ast_bb.EmptyFuncDecl(self._set_arguments_name + dev_func_id)
        self.file_ast.append(set_arguments_kernel)
        arg_body = set_arguments_kernel.compound.statements
        self.__set_arg_misc(arg_body)

        kernel_args = cg.get_kernel_args(self.ast)

        kernel_id = self._get_kernel_id()
        types = ci.get_types(self.ast)
        err_name = self._err_name
        dict_n_to_dev_ptr = cd.get_dev_ids(self.ast)

        name_swap = boilerplatebase.BPNameSwap(self.ast)

        lval = lan.Id(err_name)
        op = '|='
        for n in sorted(kernel_args):
            arg_type = types[n]
            if boilerplatebase.is_type_pointer(arg_type):
                rval = self._create_cl_set_kernel_arg(kernel_id, boilerplatebase.count_id(), self._cl_mem_name,
                                                      dict_n_to_dev_ptr[n])
            else:
                n = name_swap.try_swap(n)
                cl_type = arg_type[0]
                if cl_type == 'size_t':
                    cl_type = 'unsigned'
                rval = self._create_cl_set_kernel_arg(kernel_id, boilerplatebase.count_id(), cl_type, n)
            arg_body.append(lan.Assignment(lval, rval, op))

        err_check = self._err_check_function(self._cl_set_kernel_arg_name)
        arg_body.append(err_check)

    def __set_arg_misc(self, arg_body):
        arg_body.append(self._cl_success())

        lval = lan.TypeId(['int'], boilerplatebase.count_id())
        rval = lan.Constant(0)
        arg_body.append(lan.Assignment(lval, rval))

    def _create_cl_set_kernel_arg(self, kernel_id, cnt_name, ctype, var_ref):
        arglist = [kernel_id,
                   lan.Increment(cnt_name, '++'),
                   boilerplatebase.func_call_sizeof(ctype),
                   boilerplatebase.void_pointer_ref(var_ref)]
        return ast_bb.FuncCall(self._cl_set_kernel_arg_name, arglist)
