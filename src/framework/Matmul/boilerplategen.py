import lan
import copy
import ast_buildingblock as ast_bb
import collect_gen as cg
import collect_id as ci
import transpose
import collect_array as ca
import define_arguments
import collect_loop as cl
import collect_device as cd
import kernelgen
import global_vars
import boilerplatebase
import buffer_allocation


def print_dict_sorted(mydict):
    keys = sorted(mydict)

    entries = ""
    for key in keys:
        value = mydict[key]
        entries += "'" + key + "': " + value.__repr__() + ","

    return "{" + entries[:-1] + "}"


class Boilerplate(boilerplatebase.BoilerplateBase):
    def __init__(self, ast, no_read_back):
        super(Boilerplate, self).__init__(ast)
        self.NoReadBack = no_read_back
        self.file_ast = lan.FileAST([])

    def generate_code(self):

        self.file_ast = lan.FileAST([])

        globals_vars = global_vars.GlobalVars(self.ast, self.file_ast.ext)
        globals_vars.add_global_vars()

        # Generate the GetKernelCode function
        create_kernels = kernelgen.CreateKernels(self.ast, self.file_ast.ext)
        create_kernels.create_get_kernel_code()

        host_buffer_allocation = buffer_allocation.BufferAllocation(self.ast, self.file_ast.ext)
        host_buffer_allocation.add_buffer_allocation_function()

        self.__set_kernel_args()

        self.__add_exec_kernel_func()

        self.__add_runocl_func()

        return self.file_ast

    @property
    def __set_arguments_name(self):
        return 'SetArguments'

    @property
    def __cl_set_kernel_arg_name(self):
        return 'clSetKernelArg'

    @property
    def __exec_event_name(self):
        return 'GPUExecution'

    @property
    def __command_queue_name(self):
        return 'command_queue'

    @property
    def __cl_exec_kernel_func_name(self):
        return 'clEnqueueNDRangeKernel'

    @property
    def __cl_finish_name(self):
        return 'clFinish'

    def __set_arg_misc(self, arg_body):
        arg_body.append(self._cl_success())

        lval = lan.TypeId(['int'], _count_id())
        rval = lan.Constant(0)
        arg_body.append(lan.Assignment(lval, rval))

    def __set_kernel_args(self):

        dev_func_id = cd.get_dev_func_id(self.ast)

        set_arguments_kernel = ast_bb.EmptyFuncDecl(self.__set_arguments_name + dev_func_id)
        self.file_ast.ext.append(set_arguments_kernel)
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
            if _is_type_pointer(arg_type):
                rval = self._create_cl_set_kernel_arg(kernel_id, _count_id(), self._cl_mem_name, dict_n_to_dev_ptr[n])
            else:
                n = name_swap.try_swap(n)
                cl_type = arg_type[0]
                if cl_type == 'size_t':
                    cl_type = 'unsigned'
                rval = self._create_cl_set_kernel_arg(kernel_id, _count_id(), cl_type, n)
            arg_body.append(lan.Assignment(lval, rval, op))

        err_check = self._err_check_function(self.__cl_set_kernel_arg_name)
        arg_body.append(err_check)

    def __add_exec_misc(self, exec_body):
        exec_body.append(self._cl_success())
        event_name = lan.Id(self.__exec_event_name)
        event = lan.TypeId(['cl_event'], event_name)
        exec_body.append(event)

    def __add_exec_grid_var(self, name, value, exec_body):
        lval = lan.TypeId(['size_t'], lan.Id(name + '[]'))
        rval = lan.ArrayInit(value)
        exec_body.append(lan.Assignment(lval, rval))

    def __add_exec_grid_vars(self, exec_body):
        grid_indices = cl.get_grid_indices(self.ast)
        (lower_limit, upper_limit) = cl.get_loop_limits(self.ast)

        initlist = []
        for m in reversed(grid_indices):
            initlist.append(lan.Id(upper_limit[m] + ' - ' + lower_limit[m]))
        self.__add_exec_grid_var(cd.get_global_work_size(self.ast), initlist, exec_body)

        local = cl.get_local(self.ast)
        local_worksize = [lan.Id(i) for i in local['size']]
        self.__add_exec_grid_var(cd.get_local_work_size(self.ast), local_worksize, exec_body)

        initlist = []
        for m in reversed(grid_indices):
            initlist.append(lan.Id(lower_limit[m]))
        self.__add_exec_grid_var(cd.get_global_grid_offset(self.ast), initlist, exec_body)

    def __add_exec_cl_kernel_func_call(self, exec_body):
        par_dim = cl.get_par_dim(self.ast)
        lval = lan.Id(self._err_name)
        kernel_name = cd.get_kernel_name(self.ast)

        arglist = [lan.Id(self.__command_queue_name),
                   lan.Id(kernel_name),
                   lan.Constant(par_dim),
                   lan.Id(cd.get_global_grid_offset(self.ast)),
                   lan.Id(cd.get_global_work_size(self.ast)),
                   lan.Id(cd.get_local_work_size(self.ast)),
                   lan.Constant(0), lan.Id('NULL'),
                   lan.Ref(self.__exec_event_name)]
        rval = ast_bb.FuncCall(self.__cl_exec_kernel_func_name, arglist)
        exec_body.append(lan.Assignment(lval, rval))

        err_check = self._err_check_function(self.__cl_exec_kernel_func_name)
        exec_body.append(err_check)

    def __add_exec_cl_kernel_finish(self, exec_body):
        finish = ast_bb.FuncCall(self.__cl_finish_name, [lan.Id(self.__command_queue_name)])
        exec_body.append(lan.Assignment(lan.Id(self._err_name), finish))

        err_check = self._err_check_function(self.__cl_finish_name)
        exec_body.append(err_check)

    def __add_exec_read_back(self, exec_body):
        dev_ids = cd.get_dev_ids(self.ast)
        my_host_id = cg.get_host_ids(self.ast)
        name_swap = boilerplatebase.BPNameSwap(self.ast)

        write_only = ca.get_write_only(self.ast)
        mem_names = cg.get_mem_names(self.ast)

        cl_read_back_func_name = 'clEnqueueReadBuffer'
        if not self.NoReadBack:
            for n in sorted(write_only):
                lval = lan.Id(self._err_name)
                hst_nname = my_host_id[n]
                hst_nname = name_swap.try_swap(hst_nname)

                arglist = [lan.Id(self.__command_queue_name),
                           lan.Id(dev_ids[n]),
                           lan.Id('CL_TRUE'),
                           lan.Constant(0),
                           lan.Id(mem_names[n]),
                           lan.Id(hst_nname),
                           lan.Constant(1),
                           lan.Ref(self.__exec_event_name), lan.Id('NULL')]
                rval = ast_bb.FuncCall(cl_read_back_func_name, arglist)
                exec_body.append(lan.Assignment(lval, rval))

                err_check = self._err_check_function(cl_read_back_func_name)
                exec_body.append(err_check)

            # add clFinish statement
            arglist = [lan.Id(self.__command_queue_name)]
            finish = ast_bb.FuncCall(self.__cl_finish_name, arglist)
            exec_body.append(lan.Assignment(lan.Id(self._err_name), finish))

            err_check = self._err_check_function(self.__cl_finish_name)
            exec_body.append(err_check)

    def __add_exec_kernel_func(self):
        dev_func_id = cd.get_dev_func_id(self.ast)
        exec_kernel = ast_bb.EmptyFuncDecl('Exec' + dev_func_id)
        self.file_ast.ext.append(exec_kernel)
        exec_body = exec_kernel.compound.statements
        self.__add_exec_misc(exec_body)

        self.__add_exec_grid_vars(exec_body)

        self.__add_exec_cl_kernel_func_call(exec_body)

        self.__add_exec_cl_kernel_finish(exec_body)

        self.__add_exec_read_back(exec_body)

    def __get_runocl_args(self):
        arg_ids = self.__get_arg_ids()
        type_id_list = []
        array_id_to_dim_name = cg.get_array_id_to_dim_name(self.ast)
        types = ci.get_types(self.ast)

        for n in sorted(arg_ids):
            arg_type = types[n]
            argn = _get_arg_id(n)
            type_id_list.append(lan.TypeId(arg_type, argn))
            try:
                for m in sorted(array_id_to_dim_name[n]):
                    arg_type = ['size_t']
                    argm = _get_arg_id(m)
                    type_id_list.append(lan.TypeId(arg_type, argm))
            except KeyError:
                pass
        return type_id_list

    def __get_arg_ids(self):
        non_array_ids = ci.get_non_array_ids(self.ast)
        array_ids = ca.get_array_ids(self.ast)
        arg_ids = non_array_ids.union(array_ids)
        return arg_ids

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
            argn = _get_arg_id(n)
            rval = argn
            if_then_list.append(lan.Assignment(lval, rval))
            try:
                for m in sorted(array_id_to_dim_name[n]):
                    argm = _get_arg_id(m)
                    lval = lan.Id(m)
                    rval = argm
                    if_then_list.append(lan.Assignment(lval, rval))
            except KeyError:
                pass
        return if_then_list

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
        if_then_list.append(ast_bb.FuncCall(self.__set_arguments_name + dev_func_id))

    def __runocl_insert_timing(self, run_ocl_body):
        dev_func_id = cd.get_dev_func_id(self.ast)
        run_ocl_body.append(lan.Id('timer.start();'))
        run_ocl_body.append(ast_bb.FuncCall('Exec' + dev_func_id))
        run_ocl_body.append(lan.Id('cout << "$Time " << timer.stop() << endl;'))

    def __add_runocl_func(self):
        kernel_name = cd.get_kernel_name(self.ast)
        run_ocl = ast_bb.EmptyFuncDecl('RunOCL' + kernel_name)
        self.file_ast.ext.append(run_ocl)

        run_ocl_body = run_ocl.compound.statements

        type_id_list = self.__get_runocl_args()
        run_ocl.arglist = lan.ArgList(type_id_list)

        if_then_list = self.__set_runocl_in_host()

        self._add_runocl_start_up(if_then_list)

        self.__runocl_compile_kernel(if_then_list)
        self.__runocl_set_kernel_arguments(if_then_list)

        run_ocl_body.append(lan.IfThen(lan.Id(self._first_time_name), lan.Compound(if_then_list)))

        self.__runocl_insert_timing(run_ocl_body)

    def _add_runocl_start_up(self, if_then_list):
        if_then_list.append(ast_bb.FuncCall('StartUpGPU'))
        if_then_list.append(ast_bb.FuncCall(self._allocate_buffers_name))
        if_then_list.append(lan.Cout([lan.Constant('$Defines '), lan.Id(self._kernel_defines_name)]))

    def _create_cl_set_kernel_arg(self, kernel_id, cnt_name, ctype, var_ref):
        arglist = [kernel_id,
                   lan.Increment(cnt_name, '++'),
                   boilerplatebase.func_call_sizeof(ctype),
                   _void_pointer_ref(var_ref)]
        return ast_bb.FuncCall(self.__cl_set_kernel_arg_name, arglist)


def _void_pointer_ref(ptr_name):
    return lan.Id('(void *) &' + ptr_name)


def _is_type_pointer(ctype):
    return len(ctype) == 2


def _count_id():
    return lan.Id('counter')


def _get_arg_id(var_name):
    return lan.Id('arg_' + var_name)
