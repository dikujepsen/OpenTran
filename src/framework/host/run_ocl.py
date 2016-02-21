import boilerplatebase
import lan
import lan.ast_buildingblock as ast_bb
from processing import collect_array as ca
from processing import collect_device as cd
from processing import collect_gen as cg
from processing import collect_id as ci


class RunOCL(boilerplatebase.BoilerplateBase):
    def __init__(self, ast, file_ast):
        super(RunOCL, self).__init__(ast, file_ast)

    def add_runocl_func(self):


        kernel_name = cd.get_kernel_name(self.ast)
        run_ocl = ast_bb.EmptyFuncDecl('RunOCL' + kernel_name)
        self.file_ast.append(run_ocl)

        run_ocl_body = run_ocl.compound.statements

        type_id_list = self.__get_runocl_args()
        run_ocl.arglist = lan.ArgList(type_id_list)

        if_then_list = self.__set_runocl_in_host()

        self.__add_runocl_start_up(if_then_list)

        self.__runocl_compile_kernel(if_then_list)
        self.__runocl_set_kernel_arguments(if_then_list)

        run_ocl_body.append(lan.IfThen(lan.Id(self._first_time_name), lan.Compound(if_then_list)))

        self.__runocl_insert_timing(run_ocl_body)

    def __get_runocl_args(self):
        arg_ids = self.__get_arg_ids()
        type_id_list = []
        array_id_to_dim_name = cg.get_array_id_to_dim_name(self.ast)
        types = ci.get_types(self.ast)

        for n in sorted(arg_ids):
            arg_type = types[n]
            argn = self.__get_arg_id(n)
            type_id_list.append(lan.TypeId(arg_type, argn))
            try:
                for m in sorted(array_id_to_dim_name[n]):
                    arg_type = ['size_t']
                    argm = self.__get_arg_id(m)
                    type_id_list.append(lan.TypeId(arg_type, argm))
            except KeyError:
                pass
        return type_id_list

    def __set_runocl_in_host(self):
        if_then_list = []
        arg_ids = self.__get_arg_ids()
        array_id_to_dim_name = cg.get_array_id_to_dim_name(self.ast)
        my_host_id = cg.get_host_ids(self.ast)

        for n in sorted(arg_ids):
            try:
                newn = my_host_id[n]
            except KeyError:
                newn = n
            lval = lan.Id(newn)
            argn = self.__get_arg_id(n)
            rval = argn
            if_then_list.append(lan.Assignment(lval, rval))
            try:
                for m in sorted(array_id_to_dim_name[n]):
                    argm = self.__get_arg_id(m)
                    lval = lan.Id(m)
                    rval = argm
                    if_then_list.append(lan.Assignment(lval, rval))
            except KeyError:
                pass
        return if_then_list

    def __add_runocl_start_up(self, if_then_list):
        if_then_list.append(ast_bb.FuncCall('StartUpOCL', [lan.Id(self.__get_ocl_type_name)]))
        if_then_list.append(ast_bb.FuncCall(self._allocate_buffers_name))
        if_then_list.append(lan.Cout([lan.Constant('$Defines '), lan.Id(self._kernel_defines_name)]))

    def __runocl_compile_kernel(self, if_then_list):
        dev_func_id = cd.get_dev_func_id(self.ast)
        kernel_name = cd.get_kernel_name(self.ast)
        arglist = [lan.Constant(dev_func_id),
                   lan.Constant(dev_func_id + '.cl'),
                   ast_bb.FuncCall('GetKernelCode'),
                   lan.Id('false'),
                   lan.Ref(kernel_name),
                   lan.Id(self._kernel_defines_name)]
        if_then_list.append(ast_bb.FuncCall('compileKernel', arglist))

    def __runocl_set_kernel_arguments(self, if_then_list):
        dev_func_id = cd.get_dev_func_id(self.ast)
        if_then_list.append(ast_bb.FuncCall(self._set_arguments_name + dev_func_id))

    def __runocl_insert_timing(self, run_ocl_body):
        dev_func_id = cd.get_dev_func_id(self.ast)
        run_ocl_body.append(lan.Id('timer.start();'))
        run_ocl_body.append(ast_bb.FuncCall('Exec' + dev_func_id))
        run_ocl_body.append(lan.Id('cout << "$Time " << timer.stop() << endl;'))

    def __get_arg_ids(self):

        non_array_ids = ci.get_non_array_ids(self.ast)
        array_ids = ca.get_array_ids(self.ast)
        arg_ids = non_array_ids.union(array_ids)
        return arg_ids

    @staticmethod
    def __get_arg_id(var_name):
        return lan.Id('arg_' + var_name)

    @property
    def __get_ocl_type_name(self):
        return 'ocl_type'
