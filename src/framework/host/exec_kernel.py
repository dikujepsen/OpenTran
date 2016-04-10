import boilerplatebase
import lan
import lan.ast_buildingblock as ast_bb
from processing import collect_array as ca
from processing import collect_device as cd
from processing import collect_gen as cg
from processing import collect_loop as cl


class ExecKernel(boilerplatebase.BoilerplateBase):
    def __init__(self, ast, file_ast, no_read_back):
        super(ExecKernel, self).__init__(ast, file_ast)
        self.NoReadBack = no_read_back

    def add_exec_kernel_func(self):
        dev_func_id = cd.get_dev_func_id(self.ast)
        exec_kernel = ast_bb.EmptyFuncDecl('Exec' + dev_func_id)
        self.file_ast.append(exec_kernel)
        exec_body = exec_kernel.compound.statements
        self.__add_exec_misc(exec_body)

        self.__add_exec_grid_vars(exec_body)

        self.__add_exec_cl_kernel_func_call(exec_body)

        self.__add_exec_cl_kernel_finish(exec_body)

        self.__add_exec_read_back(exec_body)

    def __add_exec_misc(self, exec_body):
        exec_body.append(self._cl_success())
        event_name = lan.Id(self._exec_event_name)
        event = lan.TypeId(['cl_event'], event_name)
        exec_body.append(event)

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

        arglist = [self._command_queue_name_member_func,
                   lan.Id(kernel_name),
                   lan.Constant(par_dim),
                   lan.Id(cd.get_global_grid_offset(self.ast)),
                   lan.Id(cd.get_global_work_size(self.ast)),
                   lan.Id(cd.get_local_work_size(self.ast)),
                   lan.Constant(0), lan.Id('NULL'),
                   lan.Ref(self._exec_event_name)]
        rval = ast_bb.FuncCall(self._cl_exec_kernel_func_name, arglist)
        exec_body.append(lan.Assignment(lval, rval))

        err_check = self._err_check_function(self._cl_exec_kernel_func_name)
        exec_body.append(err_check)

    def __add_exec_cl_kernel_finish(self, exec_body):
        finish = ast_bb.FuncCall(self._cl_finish_name, [self._command_queue_name_member_func])
        exec_body.append(lan.Assignment(lan.Id(self._err_name), finish))

        err_check = self._err_check_function(self._cl_finish_name)
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

                arglist = [self._command_queue_name_member_func,
                           lan.Id(dev_ids[n]),
                           lan.Id('CL_TRUE'),
                           lan.Constant(0),
                           lan.Id(mem_names[n]),
                           lan.Id(hst_nname),
                           lan.Constant(1),
                           lan.Ref(self._exec_event_name), lan.Id('NULL')]
                rval = ast_bb.FuncCall(cl_read_back_func_name, arglist)
                exec_body.append(lan.Assignment(lval, rval))

                err_check = self._err_check_function(cl_read_back_func_name)
                exec_body.append(err_check)

            # add clFinish statement
            arglist = [self._command_queue_name_member_func]
            finish = ast_bb.FuncCall(self._cl_finish_name, arglist)
            exec_body.append(lan.Assignment(lan.Id(self._err_name), finish))

            err_check = self._err_check_function(self._cl_finish_name)
            exec_body.append(err_check)

    def __add_exec_grid_var(self, name, value, exec_body):
        lval = lan.TypeId(['size_t'], lan.Id(name + '[]'))
        rval = lan.ArrayInit(value)
        exec_body.append(lan.Assignment(lval, rval))
