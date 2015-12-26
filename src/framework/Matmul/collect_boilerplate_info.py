import collect_transformation_info as cti
import collect


class GenReverseIdx(object):
    def __init__(self):
        self.ReverseIdx = dict()
        self.ReverseIdx[0] = 1
        self.ReverseIdx[1] = 0


class FindLoopArrays(object):
    def __init__(self):
        self.loop_arrays = dict()

    def collect(self, ast):
        arr_to_ref = collect.ArrayNameToRef()
        arr_to_ref.visit(ast)
        self.loop_arrays = arr_to_ref.LoopArrays


class FindKernelName(cti.FindArrayIds):
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
        other_ids = self.ArrayIds.union(self.NonArrayIds)
        find_device_args = collect.FindDeviceArgs(other_ids)

        find_device_args.visit(ast)

        self.DevArgList = find_device_args.arglist
        find_function = collect.FindFunction()

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
