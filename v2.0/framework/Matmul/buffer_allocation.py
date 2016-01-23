import lan
import boilerplatebase
import define_arguments
import ast_buildingblock as ast_bb
import collect_id as ci
import collect_gen as cg
import collect_array as ca
import transpose
import collect_device as cd

class BufferAllocation(boilerplatebase.BoilerplateBase):
    def __init__(self, ast, file_ast):
        super(BufferAllocation, self).__init__(ast, file_ast)

    def add_buffer_allocation_function(self):
        allocate_buffer = ast_bb.EmptyFuncDecl(self._allocate_buffers_name)
        self.file_ast.append(allocate_buffer)

        self.__set_mem_sizes(allocate_buffer)

        self.__set_transpose_arrays(allocate_buffer)

        allocate_buffer.compound.statements.append(lan.GroupCompound([lan.Comment('// Constant Memory')]))

        define_compound = define_arguments.setdefine(self.ast)
        allocate_buffer.compound.statements.append(define_compound)

        self.__add_create_device_buffers(allocate_buffer)

    def __set_mem_sizes(self, allocate_buffer):
        types = ci.get_types(self.ast)
        list_set_mem_size = []
        mem_names = cg.get_mem_names(self.ast)
        array_id_to_dim_name = cg.get_array_id_to_dim_name(self.ast)
        array_ids = ca.get_array_ids(self.ast)

        for entry in sorted(array_ids):
            n = array_id_to_dim_name[entry]
            lval = lan.Id(mem_names[entry])
            rval = lan.BinOp(lan.Id(n[0]), '*', boilerplatebase.func_call_sizeof(types[entry][0]))
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

    def __add_create_device_buffers(self, allocate_buffer):
        cl_suc = self._cl_success()
        allocate_buffer.compound.statements.extend([lan.GroupCompound([cl_suc])])

        name_swap = boilerplatebase.BPNameSwap(self.ast)
        write_only = ca.get_write_only(self.ast)
        read_only = ca.get_read_only(self.ast)

        dict_n_to_dev_ptr = cd.get_dev_ids(self.ast)
        dict_n_to_hst_ptr = cg.get_host_ids(self.ast)
        mem_names = cg.get_mem_names(self.ast)
        for n in sorted(dict_n_to_dev_ptr):
            lval = lan.Id(dict_n_to_dev_ptr[n])
            arrayn = dict_n_to_hst_ptr[n]
            arrayn = name_swap.try_swap(arrayn)
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
                       lan.Ref(self._err_name)]

            cl_create_buffer_name = 'clCreateBuffer'
            rval = ast_bb.FuncCall(cl_create_buffer_name, arglist)
            allocate_buffer.compound.statements.append(lan.Assignment(lval, rval))

            err_check = self._err_check_function(cl_create_buffer_name, var_name=lval.name)
            allocate_buffer.compound.statements.append(err_check)
