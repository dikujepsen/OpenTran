import lan
import collect_device as cd
import ast_buildingblock as ast_bb
import collect_array as ca


class BoilerplateBase(object):
    def __init__(self, ast):
        self.ast = ast

        # Properties
        # Used in globals_vars
        self._cl_mem_name = 'cl_mem'
        self._first_time_name = 'isFirstTime'
        self._kernel_defines_name = 'KernelDefines'

        # Used in BufferAllocation
        self._allocate_buffers_name = 'AllocateBuffers'
        self._err_name = 'oclErrNum'

    def _get_kernel_id(self):
        kernel_name = cd.get_kernel_name(self.ast)
        kernel_id = lan.Id(kernel_name)
        return kernel_id

    def _err_check_function(self, cl_function_name, var_name=''):
        if not var_name == '':
            var_name = ' ' + var_name
        arglist = [lan.Id(self._err_name), lan.Constant(cl_function_name + var_name)]
        return ast_bb.FuncCall('oclCheckErr', arglist)

    def _cl_success(self):
        lval = lan.TypeId(['cl_int'], lan.Id(self._err_name))
        rval = lan.Id('CL_SUCCESS')
        return lan.Assignment(lval, rval)


def func_call_sizeof(ctype):
    return ast_bb.FuncCall('sizeof', [lan.Type(ctype)])


class BPNameSwap(object):
    def __init__(self, ast):
        self.name_swap = ca.get_host_array_name_swap(ast)

    def try_swap(self, key):
        try:
            key = self.name_swap[key]
        except KeyError:
            pass
        return key
