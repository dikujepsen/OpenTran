import collect_transformation_info as cti
import transf_visitor as tvisitor

class GenReverseIdx(object):
    def __init__(self):
        self.ReverseIdx = dict()
        self.ReverseIdx[0] = 1
        self.ReverseIdx[1] = 0


class FindLoopArrays(cti.FindLoops):
    def __init__(self):
        super(FindLoopArrays, self).__init__()

    def collect(self, ast):
        super(FindLoopArrays, self).collect(ast)

    @property
    def loop_arrays(self):
        return self.arrays.LoopArrays


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
        find_device_args = tvisitor.FindDeviceArgs(other_ids)
        find_device_args.visit(ast)
        self.DevArgList = find_device_args.arglist
        find_function = tvisitor.FindFunction()
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