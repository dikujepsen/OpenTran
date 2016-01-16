import lan
import boilerplatebase
import collect_array as ca
import collect_device as cd
import collect_id as ci
import collect_gen as cg


class GlobalVars(boilerplatebase.BoilerplateBase):
    def __init__(self, ast, file_ast):
        super(GlobalVars, self).__init__(ast)
        self.file_ast = file_ast

    def __add_util_includes(self):
        self.file_ast.append(lan.RawCpp('#include \"../../../utils/StartUtil.cpp\"'))
        self.file_ast.append(lan.RawCpp('using namespace std;'))

    def __add_global_kernel(self):
        kernel_id = self._get_kernel_id()
        kernel_type_id = lan.TypeId(['cl_kernel'], kernel_id, 0)
        self.file_ast.append(kernel_type_id)

    def __add_global_device_buffers(self):
        list_dev_buffers = []

        array_ids = ca.get_array_ids(self.ast)
        dev_ids = cd.get_dev_ids(self.ast)

        for n in sorted(array_ids):
            try:
                name = dev_ids[n]
                list_dev_buffers.append(lan.TypeId([self._cl_mem_name], lan.Id(name)))
            except KeyError:
                pass

        list_dev_buffers = lan.GroupCompound(list_dev_buffers)

        self.file_ast.append(list_dev_buffers)

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
        self.file_ast.append(list_host_ptrs)

    def __add_global_mem_sizes(self):
        list_mem_size = []
        mem_names = cg.get_mem_names(self.ast)
        for n in sorted(mem_names):
            size_name = mem_names[n]
            list_mem_size.append(lan.TypeId(['size_t'], lan.Id(size_name)))

        self.file_ast.append(lan.GroupCompound(list_mem_size))

    def __add_global_dim_sizes(self):
        array_id_to_dim_name = cg.get_array_id_to_dim_name(self.ast)
        array_ids = ca.get_array_ids(self.ast)

        list_dim_size = []
        for n in sorted(array_ids):
            for dimName in array_id_to_dim_name[n]:
                list_dim_size.append(lan.TypeId(['size_t'], lan.Id(dimName)))

        self.file_ast.append(lan.GroupCompound(list_dim_size))

    def __add_global_misc(self):
        misc = []
        lval = lan.TypeId(['size_t'], lan.Id(self._first_time_name))
        rval = lan.Constant(1)
        misc.append(lan.Assignment(lval, rval))

        lval = lan.TypeId(['std::string'], lan.Id(self._kernel_defines_name))
        rval = lan.Constant('""')
        misc.append(lan.Assignment(lval, rval))

        lval = lan.TypeId(['Stopwatch'], lan.Id('timer'))
        misc.append(lval)

        self.file_ast.append(lan.GroupCompound(misc))

    def add_global_vars(self):
        self.__add_util_includes()

        self.__add_global_kernel()

        self.__add_global_device_buffers()

        self.__add_global_hostside_args()

        self.__add_global_mem_sizes()

        self.__add_global_dim_sizes()

        self.__add_global_misc()
