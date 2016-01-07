import collect_transformation_info as cti
import collect_device as cd
import collect_array as ca


class FindLoopArrays(object):
    def __init__(self):
        self.loop_arrays = dict()
        self.loop_arrays_parent = dict()

    def collect(self, ast):
        arr_to_ref = ca.ArrayNameToRef()
        arr_to_ref.visit(ast)
        self.loop_arrays = arr_to_ref.LoopArrays
        self.loop_arrays_parent = arr_to_ref.LoopArraysParent


class FindKernelName(cti.FindArrayIdsKernel):
    def __init__(self):
        super(FindKernelName, self).__init__()
        self.KernelName = None
        self.DevId = dict()
        self.DevFuncId = None
        self.DevFuncTypeId = None
        self.DevArgList = list()
        self.Mem = dict()
        self.Worksize = dict()

    def collect(self, ast):
        super(FindKernelName, self).collect(ast)

        find_device_args = cd.FindDeviceArgs()

        find_device_args.visit(ast)

        self.DevArgList = find_device_args.arglist

        find_function = cd.FindFunction()

        find_function.visit(ast)

        self.DevFuncTypeId = find_function.typeid
        self.DevFuncId = self.DevFuncTypeId.name.name
        kernel_name = self.DevFuncTypeId.name.name

        for n in self.ArrayIds:
            self.DevId[n] = 'dev_ptr' + n
            self.Mem[n] = 'hst_ptr' + n + '_mem_size'

        self.KernelName = kernel_name + 'Kernel'
        self.Worksize['local'] = kernel_name + '_local_worksize'
        self.Worksize['global'] = kernel_name + '_global_worksize'
        self.Worksize['offset'] = kernel_name + '_global_offset'
