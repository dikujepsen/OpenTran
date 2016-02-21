import boilerplatebase
import buffer_allocation
import exec_kernel
import global_vars
import host.kernelgen
import kernel_args
import lan
import run_ocl


def print_dict_sorted(mydict):
    keys = sorted(mydict)

    entries = ""
    for key in keys:
        value = mydict[key]
        entries += "'" + key + "': " + value.__repr__() + ","

    return "{" + entries[:-1] + "}"


class Boilerplate(boilerplatebase.BoilerplateBase):
    def __init__(self, ast, no_read_back):
        super(Boilerplate, self).__init__(ast, lan.FileAST([]))
        self.NoReadBack = no_read_back

    def generate_code(self):
        self.file_ast = lan.FileAST([])

        globals_vars = global_vars.GlobalVars(self.ast, self.file_ast.ext)
        globals_vars.add_global_vars()

        # Generate the GetKernelCode function
        create_kernels = host.kernelgen.CreateKernels(self.ast, self.file_ast.ext)
        create_kernels.create_get_kernel_code()

        host_buffer_allocation = buffer_allocation.BufferAllocation(self.ast, self.file_ast.ext)
        host_buffer_allocation.add_buffer_allocation_function()

        host_kernel_args = kernel_args.KernelArgs(self.ast, self.file_ast.ext)
        host_kernel_args.set_kernel_args()

        host_exec_kernel = exec_kernel.ExecKernel(self.ast, self.file_ast.ext, self.NoReadBack)
        host_exec_kernel.add_exec_kernel_func()

        host_run_ocl = run_ocl.RunOCL(self.ast, self.file_ast.ext)
        host_run_ocl.add_runocl_func()

        return self.file_ast
