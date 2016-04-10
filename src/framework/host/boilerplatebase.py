import lan
import lan.ast_buildingblock as ast_bb
from processing import collect_array as ca
from processing import collect_device as cd


class BoilerplateBase(object):
    def __init__(self, ast, file_ast):
        self.ast = ast
        self.file_ast = file_ast

        # Properties
        # Used in globals_vars
        self._cl_mem_name = 'cl_mem'
        self._first_time_name = 'isFirstTime'
        self._kernel_defines_name = 'KernelDefines'
        self._ocl_context_name = 'ocl_context'
        self._ocl_context_class_name = 'OCLContext'

        # Used in BufferAllocation
        self._allocate_buffers_name = 'AllocateBuffers'
        self._context_member_func = ast_bb.ClassMemberFuncCall(self._ocl_context_name, 'getContext')
        self._err_name = 'oclErrNum'

        # Used in KernelArgs
        self._set_arguments_name = 'SetArguments'
        self._cl_set_kernel_arg_name = 'clSetKernelArg'

        # Used in ExecKernel
        self._exec_event_name = 'GPUExecution'
        self._command_queue_name = 'command_queue'
        self._command_queue_name_member_func = ast_bb.ClassMemberFuncCall(self._ocl_context_name, 'getCommandQueue')
        self._cl_exec_kernel_func_name = 'clEnqueueNDRangeKernel'
        self._cl_finish_name = 'clFinish'

        # Debug purposes
        self._is_debug = False

    def _get_kernel_id(self):
        kernel_name = cd.get_kernel_name(self.ast)
        kernel_id = lan.Id(kernel_name)
        return kernel_id

    def _err_check_function(self, cl_function_name, var_name=''):
        if not var_name == '':
            var_name = ' ' + var_name
        arglist = [lan.Id(self._err_name), lan.Constant(cl_function_name + var_name)]
        return ast_bb.FuncCall('helper::oclCheckErr', arglist)

    def _cl_success(self):
        lval = lan.TypeId(['cl_int'], lan.Id(self._err_name))
        rval = lan.Id('CL_SUCCESS')
        return lan.Assignment(lval, rval)


def func_call_sizeof(ctype):
    return ast_bb.FuncCall('sizeof', [lan.Type(ctype)])


def is_type_pointer(ctype):
    return len(ctype) == 2


def count_id():
    return lan.Id('counter')


def void_pointer_ref(ptr_name):
    return lan.Id('(void *) &' + ptr_name)


class BPNameSwap(object):
    def __init__(self, ast):
        self.name_swap = ca.get_host_array_name_swap(ast)

    def try_swap(self, key):
        try:
            key = self.name_swap[key]
        except KeyError:
            pass
        return key
