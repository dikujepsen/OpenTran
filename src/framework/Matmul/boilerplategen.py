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
import snippetgen
import stringstream
import place_in_local as piloc
import place_in_reg as pireg
import kernelgen


def print_dict_sorted(mydict):
    keys = sorted(mydict)

    entries = ""
    for key in keys:
        value = mydict[key]
        entries += "'" + key + "': " + value.__repr__() + ","

    return "{" + entries[:-1] + "}"


class Boilerplate(object):
    def __init__(self, ast, no_read_back):
        self.ast = ast
        self.NoReadBack = no_read_back
        self.file_ast = lan.FileAST([])

    def generate_code(self):

        self.file_ast = lan.FileAST([])

        self.__add_util_includes()

        self.__add_global_kernel()

        self.__add_global_device_buffers()

        self.__add_global_hostside_args()

        self.__add_global_mem_sizes()

        self.__add_global_dim_sizes()

        self.__add_global_misc()

        # Generate the GetKernelCode function
        create_kernels = kernelgen.CreateKernels(self.ast, self.file_ast)
        create_kernels.create_get_kernel_code()

        allocate_buffer = ast_bb.EmptyFuncDecl('AllocateBuffers')
        self.file_ast.ext.append(allocate_buffer)

        self.__set_mem_sizes(allocate_buffer)

        self.__set_transpose_arrays(allocate_buffer)

        allocate_buffer.compound.statements.append(lan.GroupCompound([lan.Comment('// Constant Memory')]))

        define_compound = define_arguments.setdefine(self.ast)
        allocate_buffer.compound.statements.append(define_compound)

        self.__add_create_device_buffers(allocate_buffer)

        dev_func_id = cd.get_dev_func_id(self.ast)
        set_arguments_kernel = ast_bb.EmptyFuncDecl('SetArguments' + dev_func_id)

        self.file_ast.ext.append(set_arguments_kernel)
        arg_body = set_arguments_kernel.compound.statements
        arg_body.append(self.__cl_success())
        cnt_name = lan.Id('counter')
        lval = lan.TypeId(['int'], cnt_name)
        rval = lan.Constant(0)
        arg_body.append(lan.Assignment(lval, rval))

        kernel_args = cg.get_kernel_args(self.ast)
        # clSetKernelArg for Arrays
        kernel_id = self.__get_kernel_id()
        types = ci.get_types(self.ast)
        err_name = self.__err_name
        dict_n_to_dev_ptr = cd.get_dev_ids(self.ast)
        name_swap = ca.get_host_array_name_swap(self.ast)
        for n in sorted(kernel_args):
            lval = lan.Id(err_name)
            op = '|='
            arg_type = types[n]
            if len(arg_type) == 2:
                arglist = lan.ArgList([kernel_id,
                                       lan.Increment(cnt_name, '++'),
                                       lan.Id('sizeof(cl_mem)'),
                                       lan.Id('(void *) &' + dict_n_to_dev_ptr[n])])
                rval = lan.FuncDecl(lan.Id('clSetKernelArg'), arglist, lan.Compound([]))
            else:
                try:
                    n = name_swap[n]
                except KeyError:
                    pass
                cl_type = arg_type[0]
                if cl_type == 'size_t':
                    cl_type = 'unsigned'
                arglist = lan.ArgList([kernel_id,
                                       lan.Increment(cnt_name, '++'),
                                       lan.Id('sizeof(' + cl_type + ')'),
                                       lan.Id('(void *) &' + n)])
                rval = lan.FuncDecl(lan.Id('clSetKernelArg'), arglist, lan.Compound([]))
            arg_body.append(lan.Assignment(lval, rval, op))

        arglist = lan.ArgList([lan.Id(err_name), lan.Constant('clSetKernelArg')])
        err_id = lan.Id('oclCheckErr')
        err_check = lan.FuncDecl(err_id, arglist, lan.Compound([]))
        arg_body.append(err_check)

        exec_kernel = ast_bb.EmptyFuncDecl('Exec' + dev_func_id)
        self.file_ast.ext.append(exec_kernel)
        exec_body = exec_kernel.compound.statements
        exec_body.append(self.__cl_success())
        event_name = lan.Id('GPUExecution')
        event = lan.TypeId(['cl_event'], event_name)
        exec_body.append(event)

        work_size = cd.get_work_size(self.ast)
        local = cl.get_local(self.ast)
        grid_indices = cl.get_grid_indices(self.ast)
        (lower_limit, upper_limit) = cl.get_loop_limits(self.ast)

        for n in sorted(work_size):
            lval = lan.TypeId(['size_t'], lan.Id(work_size[n] + '[]'))
            if n == 'local':
                local_worksize = [lan.Id(i) for i in local['size']]
                rval = lan.ArrayInit(local_worksize)
            elif n == 'global':
                initlist = []
                for m in reversed(grid_indices):
                    initlist.append(lan.Id(upper_limit[m] + ' - ' + lower_limit[m]))
                rval = lan.ArrayInit(initlist)
            else:
                initlist = []
                for m in reversed(grid_indices):
                    initlist.append(lan.Id(lower_limit[m]))
                rval = lan.ArrayInit(initlist)

            exec_body.append(lan.Assignment(lval, rval))

        par_dim = cl.get_par_dim(self.ast)
        lval = lan.Id(err_name)
        kernel_name = cd.get_kernel_name(self.ast)
        arglist = lan.ArgList([lan.Id('command_queue'),
                               lan.Id(kernel_name),
                               lan.Constant(par_dim),
                               lan.Id(work_size['offset']),
                               lan.Id(work_size['global']),
                               lan.Id(work_size['local']),
                               lan.Constant(0), lan.Id('NULL'),
                               lan.Id('&' + event_name.name)])
        rval = lan.FuncDecl(lan.Id('clEnqueueNDRangeKernel'), arglist, lan.Compound([]))
        exec_body.append(lan.Assignment(lval, rval))

        arglist = lan.ArgList([lan.Id(err_name), lan.Constant('clEnqueueNDRangeKernel')])
        err_check = lan.FuncDecl(err_id, arglist, lan.Compound([]))
        exec_body.append(err_check)

        arglist = lan.ArgList([lan.Id('command_queue')])
        finish = lan.FuncDecl(lan.Id('clFinish'), arglist, lan.Compound([]))
        exec_body.append(lan.Assignment(lan.Id(err_name), finish))

        arglist = lan.ArgList([lan.Id(err_name), lan.Constant('clFinish')])
        err_check = lan.FuncDecl(err_id, arglist, lan.Compound([]))
        exec_body.append(err_check)
        dev_ids = cd.get_dev_ids(self.ast)
        my_host_id = cg.get_host_ids(self.ast)
        name_swap = ca.get_host_array_name_swap(self.ast)
        write_only = ca.get_write_only(self.ast)
        mem_names = cg.get_mem_names(self.ast)

        if not self.NoReadBack:
            for n in sorted(write_only):
                lval = lan.Id(err_name)
                hst_nname = my_host_id[n]
                try:
                    hst_nname = name_swap[hst_nname]
                except KeyError:
                    pass
                arglist = lan.ArgList([lan.Id('command_queue'),
                                       lan.Id(dev_ids[n]),
                                       lan.Id('CL_TRUE'),
                                       lan.Constant(0),
                                       lan.Id(mem_names[n]),
                                       lan.Id(hst_nname),
                                       lan.Constant(1),
                                       lan.Id('&' + event_name.name), lan.Id('NULL')])
                rval = lan.FuncDecl(lan.Id('clEnqueueReadBuffer'), arglist, lan.Compound([]))
                exec_body.append(lan.Assignment(lval, rval))

                arglist = lan.ArgList([lan.Id(err_name), lan.Constant('clEnqueueReadBuffer')])
                err_check = lan.FuncDecl(err_id, arglist, lan.Compound([]))
                exec_body.append(err_check)

            # add clFinish statement
            arglist = lan.ArgList([lan.Id('command_queue')])
            finish = lan.FuncDecl(lan.Id('clFinish'), arglist, lan.Compound([]))
            exec_body.append(lan.Assignment(lan.Id(err_name), finish))

            arglist = lan.ArgList([lan.Id(err_name), lan.Constant('clFinish')])
            err_check = lan.FuncDecl(err_id, arglist, lan.Compound([]))
            exec_body.append(err_check)

        run_ocl = ast_bb.EmptyFuncDecl('RunOCL' + kernel_name)
        self.file_ast.ext.append(run_ocl)
        run_ocl_body = run_ocl.compound.statements
        non_array_ids = ci.get_non_array_ids(self.ast)
        array_ids = ca.get_array_ids(self.ast)
        arg_ids = non_array_ids.union(array_ids)

        type_id_list = []
        if_then_list = []

        array_id_to_dim_name = cg.get_array_id_to_dim_name(self.ast)
        for n in sorted(arg_ids):
            arg_type = types[n]
            argn = lan.Id('arg_' + n)
            type_id_list.append(lan.TypeId(arg_type, argn))
            try:
                newn = my_host_id[n]
            except KeyError:
                newn = n
            lval = lan.Id(newn)
            rval = argn
            if_then_list.append(lan.Assignment(lval, rval))
            try:

                for m in sorted(array_id_to_dim_name[n]):
                    arg_type = ['size_t']
                    argm = lan.Id('arg_' + m)
                    lval = lan.Id(m)
                    rval = argm
                    if_then_list.append(lan.Assignment(lval, rval))
                    type_id_list.append(lan.TypeId(arg_type, argm))
            except KeyError:
                pass

        arglist = lan.ArgList(type_id_list)
        run_ocl.arglist = arglist

        arglist = lan.ArgList([])
        if_then_list.append(lan.FuncDecl(lan.Id('StartUpGPU'), arglist, lan.Compound([])))
        if_then_list.append(lan.FuncDecl(lan.Id('AllocateBuffers'), arglist, lan.Compound([])))
        use_file = 'false'

        if_then_list.append(lan.Id('cout << "$Defines " << KernelDefines << endl;'))
        arglist = lan.ArgList([lan.Constant(dev_func_id),
                               lan.Constant(dev_func_id + '.cl'),
                               lan.Id('GetKernelCode()'),
                               lan.Id(use_file),
                               lan.Id('&' + kernel_name),
                               lan.Id('KernelDefines')])
        if_then_list.append(lan.FuncDecl(lan.Id('compileKernel'), arglist, lan.Compound([])))
        if_then_list.append(
            lan.FuncDecl(lan.Id('SetArguments' + dev_func_id), lan.ArgList([]), lan.Compound([])))

        run_ocl_body.append(lan.IfThen(lan.Id('isFirstTime'), lan.Compound(if_then_list)))
        arglist = lan.ArgList([])

        # Insert timing
        run_ocl_body.append(lan.Id('timer.start();'))
        run_ocl_body.append(lan.FuncDecl(lan.Id('Exec' + dev_func_id), arglist, lan.Compound([])))
        run_ocl_body.append(lan.Id('cout << "$Time " << timer.stop() << endl;'))

        return self.file_ast

    def __add_util_includes(self):
        self.file_ast.ext.append(lan.RawCpp('#include \"../../../utils/StartUtil.cpp\"'))
        self.file_ast.ext.append(lan.RawCpp('using namespace std;'))

    def __get_kernel_id(self):
        kernel_name = cd.get_kernel_name(self.ast)
        kernel_id = lan.Id(kernel_name)
        return kernel_id

    def __add_global_kernel(self):
        kernel_id = self.__get_kernel_id()
        kernel_type_id = lan.TypeId(['cl_kernel'], kernel_id, 0)
        self.file_ast.ext.append(kernel_type_id)

    def __add_global_device_buffers(self):
        list_dev_buffers = []

        array_ids = ca.get_array_ids(self.ast)
        dev_ids = cd.get_dev_ids(self.ast)

        for n in sorted(array_ids):
            try:
                name = dev_ids[n]
                list_dev_buffers.append(lan.TypeId(['cl_mem'], lan.Id(name)))
            except KeyError:
                pass

        list_dev_buffers = lan.GroupCompound(list_dev_buffers)

        self.file_ast.ext.append(list_dev_buffers)

    def __add_global_hostside_args(self):
        dev_arg_list = cd.get_devices_arg_list(self.ast)
        types = ci.get_types(self.ast)

        list_host_ptrs = []

        my_host_id = cg.get_host_ids(self.ast)
        for n in sorted(dev_arg_list, key=lambda type_id: type_id.name.name.lower()):
            name = n.name.name
            arg_type = types[name]
            try:
                name = my_host_id[name]
            except KeyError:
                pass
            list_host_ptrs.append(lan.TypeId(arg_type, lan.Id(name), 0))

        transposable_host_id = cg.gen_transposable_host_ids(self.ast)
        for n in sorted(transposable_host_id):
            arg_type = types[n]
            name = my_host_id[n]
            list_host_ptrs.append(lan.TypeId(arg_type, lan.Id(name), 0))

        list_host_ptrs = lan.GroupCompound(list_host_ptrs)
        self.file_ast.ext.append(list_host_ptrs)

    def __add_global_mem_sizes(self):
        list_mem_size = []
        mem_names = cg.get_mem_names(self.ast)
        for n in sorted(mem_names):
            size_name = mem_names[n]
            list_mem_size.append(lan.TypeId(['size_t'], lan.Id(size_name)))

        self.file_ast.ext.append(lan.GroupCompound(list_mem_size))

    def __add_global_dim_sizes(self):
        array_id_to_dim_name = cg.get_array_id_to_dim_name(self.ast)
        array_ids = ca.get_array_ids(self.ast)

        list_dim_size = []
        for n in sorted(array_ids):
            for dimName in array_id_to_dim_name[n]:
                list_dim_size.append(lan.TypeId(['size_t'], lan.Id(dimName)))

        self.file_ast.ext.append(lan.GroupCompound(list_dim_size))

    def __add_global_misc(self):
        misc = []
        lval = lan.TypeId(['size_t'], lan.Id('isFirstTime'))
        rval = lan.Constant(1)
        misc.append(lan.Assignment(lval, rval))

        lval = lan.TypeId(['std::string'], lan.Id('KernelDefines'))
        rval = lan.Constant('""')
        misc.append(lan.Assignment(lval, rval))

        lval = lan.TypeId(['Stopwatch'], lan.Id('timer'))
        misc.append(lval)

        self.file_ast.ext.append(lan.GroupCompound(misc))

    def __set_mem_sizes(self, allocate_buffer):
        types = ci.get_types(self.ast)
        list_set_mem_size = []
        mem_names = cg.get_mem_names(self.ast)
        array_id_to_dim_name = cg.get_array_id_to_dim_name(self.ast)
        array_ids = ca.get_array_ids(self.ast)

        for entry in sorted(array_ids):
            n = array_id_to_dim_name[entry]
            lval = lan.Id(mem_names[entry])
            rval = lan.BinOp(lan.Id(n[0]), '*', ast_bb.FuncCall('sizeof', [lan.Type(types[entry][0])]))
            if len(n) == 2:
                rval = lan.BinOp(lan.Id(n[1]), '*', rval)
            list_set_mem_size.append(lan.Assignment(lval, rval))

        allocate_buffer.compound.statements.append(lan.GroupCompound(list_set_mem_size))

    def __set_transpose_arrays(self, allocate_buffer):
        transpose_transformation = transpose.Transpose(self.ast)
        transpose_arrays = ca.get_transposable_base_ids(self.ast)
        my_transposition = lan.GroupCompound([lan.Comment('// Transposition')])
        for n in transpose_arrays:
            my_transposition.statements.extend(transpose_transformation.create_transposition_func(n))

        allocate_buffer.compound.statements.append(my_transposition)

    @property
    def __err_name(self):
        return 'oclErrNum'

    def __cl_success(self):
        lval = lan.TypeId(['cl_int'], lan.Id(self.__err_name))
        rval = lan.Id('CL_SUCCESS')
        return lan.Assignment(lval, rval)

    def __add_create_device_buffers(self, allocate_buffer):
        cl_suc = self.__cl_success()
        allocate_buffer.compound.statements.extend([lan.GroupCompound([cl_suc])])

        name_swap = ca.get_host_array_name_swap(self.ast)
        write_only = ca.get_write_only(self.ast)
        read_only = ca.get_read_only(self.ast)

        dict_n_to_dev_ptr = cd.get_dev_ids(self.ast)
        dict_n_to_hst_ptr = cg.get_host_ids(self.ast)
        mem_names = cg.get_mem_names(self.ast)
        for n in sorted(dict_n_to_dev_ptr):
            lval = lan.Id(dict_n_to_dev_ptr[n])
            arrayn = dict_n_to_hst_ptr[n]
            try:
                arrayn = name_swap[arrayn]
            except KeyError:
                pass
            if n in write_only:
                flag = lan.Id('CL_MEM_WRITE_ONLY')
                arrayn_id = lan.Id('NULL')
            elif n in read_only:
                flag = lan.Id('CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY')
                arrayn_id = lan.Id(arrayn)
            else:
                flag = lan.Id('CL_MEM_USE_HOST_PTR')
                arrayn_id = lan.Id(arrayn)

            arglist = [lan.Id('context'),
                       flag,
                       lan.Id(mem_names[n]),
                       arrayn_id,
                       lan.Id('&' + self.__err_name)]

            rval = ast_bb.FuncCall('clCreateBuffer', arglist)
            allocate_buffer.compound.statements.append(lan.Assignment(lval, rval))

            arglist = [lan.Id(self.__err_name), lan.Constant("clCreateBuffer " + lval.name)]
            err_check = ast_bb.FuncCall('oclCheckErr', arglist)
            allocate_buffer.compound.statements.append(err_check)
