import lan
import copy
import ast_buildingblock as ast_bb
import collect_transformation_info as cti
import struct
import collect_boilerplate_info as cbi
import collect_gen as cg
import collect_id as ci
import transpose
import collect_array as ca
import define_arguments

def print_dict_sorted(mydict):
    keys = sorted(mydict)

    entries = ""
    for key in keys:
        value = mydict[key]
        entries += "'" + key + "': " + value.__repr__() + ","

    return "{" + entries[:-1] + "}"


class Boilerplate(object):
    def __init__(self):
        self.ks = None
        self.bps = None
        self.kgen_strt = None

        self.bps_static = None
        self.Local = dict()
        self.GridIndices = list()
        self.UpperLimit = dict()
        self.ArrayIds = set()

        # new
        self.HstId = dict()
        self.transposable_host_id = list()
        self.Type = dict()
        self.ast = None
        self.par_dim = None
        self.NameSwap = dict()
        self.kernel_args = dict()

    def set_struct(self, kernelstruct, boilerplatestruct, kgen_strt, ast):
        self.ks = kernelstruct
        self.bps = boilerplatestruct
        self.kgen_strt = kgen_strt
        self.ast = ast
        self.par_dim = self.ks.ParDim

        fl = cti.FindLocal()
        fl.ParDim = self.ks.ParDim
        fl.collect(ast)
        self.Local = fl.Local

        fpl = cti.FindGridIndices()
        fpl.ParDim = self.ks.ParDim
        fpl.collect(ast)
        self.GridIndices = fpl.GridIndices

        fai = cti.FindReadWrite()
        fai.ParDim = self.ks.ParDim
        fai.collect(ast)

        self.UpperLimit = fai.upper_limit
        self.ArrayIds = fai.ArrayIds

        self.bps_static = struct.BoilerPlateStruct()
        self.bps_static.set_datastructure(ast, self.ks.ParDim)

        # new
        self.HstId = cg.gen_host_ids(ast)
        self.transposable_host_id = cg.gen_transposable_host_ids(ast)
        self.Type = ci.get_types(ast)
        self.NameSwap = ca.get_host_array_name_swap(ast)
        self.kernel_args = cg.get_kernel_args(ast, self.ks.ParDim)

    def generate_code(self):

        my_host_id = self.HstId
        dict_n_to_dim_names = self.ks.ArrayIdToDimName

        non_array_ids = copy.deepcopy(self.bps_static.NonArrayIds)

        file_ast = lan.FileAST([])

        file_ast.ext.append(lan.Id('#include \"../../../utils/StartUtil.cpp\"'))
        file_ast.ext.append(lan.Id('using namespace std;'))

        kernel_id = lan.Id(self.bps_static.KernelName)
        kernel_type_id = lan.TypeId(['cl_kernel'], kernel_id, 0)
        file_ast.ext.append(kernel_type_id)

        list_dev_buffers = []

        for n in sorted(self.ArrayIds):
            try:
                name = self.bps_static.DevId[n]
                list_dev_buffers.append(lan.TypeId(['cl_mem'], lan.Id(name)))
            except KeyError:
                pass

        dict_n_to_dev_ptr = self.bps_static.DevId
        list_dev_buffers = lan.GroupCompound(list_dev_buffers)

        file_ast.ext.append(list_dev_buffers)

        list_host_ptrs = []
        for n in self.bps_static.DevArgList:
            name = n.name.name
            arg_type = self.Type[name]
            try:
                name = my_host_id[name]
            except KeyError:
                pass
            list_host_ptrs.append(lan.TypeId(arg_type, lan.Id(name), 0))

        for n in sorted(self.transposable_host_id):
            arg_type = self.Type[n]
            name = my_host_id[n]
            list_host_ptrs.append(lan.TypeId(arg_type, lan.Id(name), 0))

        dict_n_to_hst_ptr = my_host_id
        dict_type_host_ptrs = copy.deepcopy(self.Type)
        list_host_ptrs = lan.GroupCompound(list_host_ptrs)
        file_ast.ext.append(list_host_ptrs)

        list_mem_size = []
        list_dim_size = []
        dict_n_to_size = self.bps_static.Mem

        for n in sorted(self.bps_static.Mem):
            size_name = self.bps_static.Mem[n]
            list_mem_size.append(lan.TypeId(['size_t'], lan.Id(size_name)))

        for n in sorted(self.ArrayIds):
            for dimName in self.ks.ArrayIdToDimName[n]:
                list_dim_size.append(lan.TypeId(['size_t'], lan.Id(dimName)))

        file_ast.ext.append(lan.GroupCompound(list_mem_size))
        file_ast.ext.append(lan.GroupCompound(list_dim_size))
        misc = []
        lval = lan.TypeId(['size_t'], lan.Id('isFirstTime'))
        rval = lan.Constant(1)
        misc.append(lan.Assignment(lval, rval))

        lval = lan.TypeId(['std::string'], lan.Id('KernelDefines'))
        rval = lan.Constant('""')
        misc.append(lan.Assignment(lval, rval))

        lval = lan.TypeId(['Stopwatch'], lan.Id('timer'))
        misc.append(lval)

        file_ast.ext.append(lan.GroupCompound(misc))

        # Generate the GetKernelCode function
        for optim in self.kgen_strt.KernelStringStream:
            file_ast.ext.append(optim['ast'])

        get_kernel_code = ast_bb.EmptyFuncDecl('GetKernelCode', type=['std::string'])
        get_kernel_stats = []
        get_kernel_code.compound.statements = get_kernel_stats
        get_kernel_stats.append(self.kgen_strt.IfThenElse)
        file_ast.ext.append(get_kernel_code)

        allocate_buffer = ast_bb.EmptyFuncDecl('AllocateBuffers')
        file_ast.ext.append(allocate_buffer)

        list_set_mem_size = []
        for entry in sorted(self.ArrayIds):
            n = self.ks.ArrayIdToDimName[entry]
            lval = lan.Id(self.bps_static.Mem[entry])
            rval = lan.BinOp(lan.Id(n[0]), '*', lan.Id('sizeof(' + self.Type[entry][0] + ')'))
            if len(n) == 2:
                rval = lan.BinOp(lan.Id(n[1]), '*', rval)
            list_set_mem_size.append(lan.Assignment(lval, rval))

        allocate_buffer.compound.statements.append(lan.GroupCompound(list_set_mem_size))

        transpose_transformation = transpose.Transpose()

        transpose_transformation.set_datastructures(self.ast, self.par_dim)
        transpose_arrays = ca.get_transposable_base_ids(self.ast)
        my_transposition = lan.GroupCompound([lan.Comment('// Transposition')])
        for n in transpose_arrays:
            my_transposition.statements.extend(transpose_transformation.create_transposition_func(n))

        allocate_buffer.compound.statements.append(my_transposition)

        allocate_buffer.compound.statements.append(lan.GroupCompound([lan.Comment('// Constant Memory')]))

        define_compound = define_arguments.setdefine(self.ast)
        allocate_buffer.compound.statements.append(define_compound)

        err_name = 'oclErrNum'
        lval = lan.TypeId(['cl_int'], lan.Id(err_name))
        rval = lan.Id('CL_SUCCESS')
        cl_suc = lan.Assignment(lval, rval)
        allocate_buffer.compound.statements.extend([lan.GroupCompound([cl_suc])])

        for n in sorted(dict_n_to_dev_ptr):
            lval = lan.Id(dict_n_to_dev_ptr[n])
            arrayn = dict_n_to_hst_ptr[n]
            try:
                arrayn = self.NameSwap[arrayn]
            except KeyError:
                pass
            if n in self.bps_static.WriteOnly:
                flag = lan.Id('CL_MEM_WRITE_ONLY')
                arrayn_id = lan.Id('NULL')
            elif n in self.bps_static.ReadOnly:
                flag = lan.Id('CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY')
                arrayn_id = lan.Id(arrayn)
            else:
                flag = lan.Id('CL_MEM_USE_HOST_PTR')
                arrayn_id = lan.Id(arrayn)

            arglist = lan.ArgList([lan.Id('context'),
                                   flag,
                                   lan.Id(dict_n_to_size[n]),
                                   arrayn_id,
                                   lan.Id('&' + err_name)])
            rval = lan.FuncDecl(lan.Id('clCreateBuffer'), arglist, lan.Compound([]))
            allocate_buffer.compound.statements.append(lan.Assignment(lval, rval))

            arglist = lan.ArgList([lan.Id(err_name), lan.Constant("clCreateBuffer " + lval.name)])
            err_check = lan.FuncDecl(lan.Id('oclCheckErr'), arglist, lan.Compound([]))
            allocate_buffer.compound.statements.append(err_check)

        set_arguments_kernel = ast_bb.EmptyFuncDecl('SetArguments' + self.bps_static.DevFuncId)
        file_ast.ext.append(set_arguments_kernel)
        arg_body = set_arguments_kernel.compound.statements
        arg_body.append(cl_suc)
        cnt_name = lan.Id('counter')
        lval = lan.TypeId(['int'], cnt_name)
        rval = lan.Constant(0)
        arg_body.append(lan.Assignment(lval, rval))

        for n in sorted(dict_n_to_dim_names):
            # add dim arguments to set of ids
            non_array_ids.add(dict_n_to_dim_names[n][0])
            # Add types of dimensions for size arguments
            dict_type_host_ptrs[dict_n_to_dim_names[n][0]] = ['size_t']

        for n in self.bps_static.RemovedIds:
            dict_type_host_ptrs.pop(n, None)

        # clSetKernelArg for Arrays
        for n in sorted(self.kernel_args):
            lval = lan.Id(err_name)
            op = '|='
            arg_type = self.Type[n]
            if len(arg_type) == 2:
                arglist = lan.ArgList([kernel_id,
                                       lan.Increment(cnt_name, '++'),
                                       lan.Id('sizeof(cl_mem)'),
                                       lan.Id('(void *) &' + dict_n_to_dev_ptr[n])])
                rval = lan.FuncDecl(lan.Id('clSetKernelArg'), arglist, lan.Compound([]))
            else:
                try:
                    n = self.NameSwap[n]
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

        exec_kernel = ast_bb.EmptyFuncDecl('Exec' + self.bps_static.DevFuncTypeId.name.name)
        file_ast.ext.append(exec_kernel)
        exec_body = exec_kernel.compound.statements
        exec_body.append(cl_suc)
        event_name = lan.Id('GPUExecution')
        event = lan.TypeId(['cl_event'], event_name)
        exec_body.append(event)

        for n in self.bps_static.Worksize:
            lval = lan.TypeId(['size_t'], lan.Id(self.bps_static.Worksize[n] + '[]'))
            if n == 'local':
                local_worksize = [lan.Id(i) for i in self.Local['size']]
                rval = lan.ArrayInit(local_worksize)
            elif n == 'global':
                initlist = []
                for m in reversed(self.GridIndices):
                    initlist.append(lan.Id(self.UpperLimit[m] + ' - ' + self.bps_static.LowerLimit[m]))
                rval = lan.ArrayInit(initlist)
            else:
                initlist = []
                for m in reversed(self.GridIndices):
                    initlist.append(lan.Id(self.bps_static.LowerLimit[m]))
                rval = lan.ArrayInit(initlist)

            exec_body.append(lan.Assignment(lval, rval))

        lval = lan.Id(err_name)
        arglist = lan.ArgList([lan.Id('command_queue'),
                               lan.Id(self.bps_static.KernelName),
                               lan.Constant(self.ks.ParDim),
                               lan.Id(self.bps_static.Worksize['offset']),
                               lan.Id(self.bps_static.Worksize['global']),
                               lan.Id(self.bps_static.Worksize['local']),
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

        if not self.bps.NoReadBack:
            for n in self.bps_static.WriteOnly:
                lval = lan.Id(err_name)
                hst_nname = my_host_id[n]
                try:
                    hst_nname = self.NameSwap[hst_nname]
                except KeyError:
                    pass
                arglist = lan.ArgList([lan.Id('command_queue'),
                                       lan.Id(self.bps_static.DevId[n]),
                                       lan.Id('CL_TRUE'),
                                       lan.Constant(0),
                                       lan.Id(self.bps_static.Mem[n]),
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

        run_ocl = ast_bb.EmptyFuncDecl('RunOCL' + self.bps_static.KernelName)
        file_ast.ext.append(run_ocl)
        run_ocl_body = run_ocl.compound.statements

        arg_ids = self.bps_static.NonArrayIds.union(self.ArrayIds)  #

        type_id_list = []
        if_then_list = []
        for n in arg_ids:
            arg_type = self.Type[n]
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
                for m in self.ks.ArrayIdToDimName[n]:
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
        use_file = 'true'
        if self.kgen_strt.KernelStringStream:
            use_file = 'false'

        if_then_list.append(lan.Id('cout << "$Defines " << KernelDefines << endl;'))
        arglist = lan.ArgList([lan.Constant(self.bps_static.DevFuncId),
                               lan.Constant(self.bps_static.DevFuncId + '.cl'),
                               lan.Id('GetKernelCode()'),
                               lan.Id(use_file),
                               lan.Id('&' + self.bps_static.KernelName),
                               lan.Id('KernelDefines')])
        if_then_list.append(lan.FuncDecl(lan.Id('compileKernel'), arglist, lan.Compound([])))
        if_then_list.append(
            lan.FuncDecl(lan.Id('SetArguments' + self.bps_static.DevFuncId), lan.ArgList([]), lan.Compound([])))

        run_ocl_body.append(lan.IfThen(lan.Id('isFirstTime'), lan.Compound(if_then_list)))
        arglist = lan.ArgList([])

        # Insert timing
        run_ocl_body.append(lan.Id('timer.start();'))
        run_ocl_body.append(lan.FuncDecl(lan.Id('Exec' + self.bps_static.DevFuncId), arglist, lan.Compound([])))
        run_ocl_body.append(lan.Id('cout << "$Time " << timer.stop() << endl;'))

        return file_ast
