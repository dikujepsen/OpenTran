import boilerplatebase
import buffer_allocation
import exec_kernel
import global_vars
import host.kernelgen
import kernel_args
import lan
import run_ocl
from processing import collect_device as cd
from processing import collect_id as ci

def print_dict_sorted(mydict):
    keys = sorted(mydict)

    entries = ""
    for key in keys:
        value = mydict[key]
        entries += "'" + key + "': " + value.__repr__() + ","

    return "{" + entries[:-1] + "}"


class Boilerplate(boilerplatebase.BoilerplateBase):
    def __init__(self, ast, no_read_back, is_debug):
        super(Boilerplate, self).__init__(ast, lan.FileAST([]))
        self._is_debug = is_debug
        self.NoReadBack = no_read_back

    def generate_code(self):
        self.file_ast = lan.FileAST([])

        self.__add_util_includes(self.file_ast.ext)

        program_name = 'OCL' + ci.get_program_name(self.ast) + 'Task'
        # GroupCompound
        global_vars_list = []
        public_list = []
        protected_list = []
        private_list = []

        globals_vars = global_vars.GlobalVars(self.ast, global_vars_list)
        globals_vars.add_global_vars()

        # Generate the GetKernelCode function
        create_kernels = host.kernelgen.CreateKernels(self.ast, private_list)
        create_kernels.create_get_kernel_code()

        host_buffer_allocation = buffer_allocation.BufferAllocation(self.ast, private_list, self._is_debug)
        host_buffer_allocation.add_buffer_allocation_function()

        host_kernel_args = kernel_args.KernelArgs(self.ast, private_list)
        host_kernel_args.set_kernel_args()

        host_exec_kernel = exec_kernel.ExecKernel(self.ast, private_list, self.NoReadBack)
        host_exec_kernel.add_exec_kernel_func()

        self.__add_class_init(public_list, program_name)

        host_run_ocl = run_ocl.RunOCL(self.ast, public_list)
        host_run_ocl.add_runocl_func()

        self.file_ast.ext.append(lan.CppClass(lan.Id(program_name), lan.GroupCompound(global_vars_list),
                                 lan.GroupCompound(public_list), lan.GroupCompound(protected_list),
                                              lan.GroupCompound(private_list)))

        return self.file_ast

    def __add_util_includes(self, file_ast):
        stat_list = [lan.RawCpp('#include \"../../../utils/StartUtil.cpp\"'),
                     lan.RawCpp('#include \"../../../utils/helper.hpp\"'), lan.RawCpp('using namespace std;')]
        file_ast.append(lan.GroupCompound(stat_list))

    def __add_class_init(self, file_ast, program_name):
        stat_list = [lan.Assignment(lan.Id(self._first_time_name), lan.Constant(1)),
                     lan.Assignment(lan.Id(self._kernel_defines_name), lan.Constant(""))]
        file_ast.append(lan.ClassConstructor(lan.Id(program_name), lan.ArgList([]), lan.Compound(stat_list)))
