import lan
import copy
import stringstream
import exchange
import collect_device as cd
import collect_gen as cg
import collect_transformation_info as cti
import collect_array as ca


class SnippetGen(object):
    def __init__(self):
        self.KernelStruct = None
        self.KernelStringStream = list()
        self.IndexToThreadId = dict()
        self.DevFuncTypeId = None
        self.ArrayIds = set()
        self.types = dict()

    def set_datastructure(self,
                          kernel_struct,
                          ast):
        self.KernelStruct = kernel_struct

        par_dim = self.KernelStruct.ParDim
        idx_to_thread_id = cg.GenIdxToThreadId()
        idx_to_thread_id.collect(ast, par_dim)

        self.IndexToThreadId = idx_to_thread_id.IndexToThreadId

        find_function = cd.FindFunction()
        find_function.visit(ast)
        self.DevFuncTypeId = find_function.typeid

        fai = cti.FindArrayIdsKernel()
        fai.ParDim = par_dim
        fai.collect(ast)
        self.ArrayIds = fai.ArrayIds
        self.types = fai.type

    def in_source_kernel(self, ast, cond, filename, kernelstringname):
        self.rewrite_to_device_c_release(ast)

        ssprint = stringstream.SSGenerator()
        emptyast = lan.FileAST([])
        ssprint.createKernelStringStream(ast, emptyast, kernelstringname, filename=filename)
        self.KernelStringStream.append({'name': kernelstringname,
                                        'ast': emptyast,
                                        'cond': cond})

    def rewrite_to_device_c_release(self, ast):
        # The list of arguments for the kernel
        dict_type_host_ptrs = copy.deepcopy(self.types)
        for n in self.ArrayIds:
            dict_type_host_ptrs[self.KernelStruct.ArrayIdToDimName[n][0]] = ['size_t']

        arglist = list()
        for n in self.KernelStruct.KernelArgs:
            kernel_type = copy.deepcopy(self.KernelStruct.KernelArgs[n])
            if kernel_type[0] == 'size_t':
                kernel_type[0] = 'unsigned'
            if len(kernel_type) == 2:
                kernel_type.insert(0, '__global')
            arglist.append(lan.TypeId(kernel_type, lan.Id(n)))

        exchange_array_id = exchange.ExchangeArrayId(self.KernelStruct.LocalSwap)

        for n in self.KernelStruct.LoopArrays.values():
            for m in n:
                exchange_array_id.visit(m)

        my_kernel = copy.deepcopy(self.KernelStruct.Kernel)
        # print self.astrepr.ArrayIdToDimName
        rewrite_array_ref = exchange.RewriteArrayRef(self.KernelStruct.num_array_dims,
                                                     self.KernelStruct.ArrayIdToDimName,
                                                     self.KernelStruct.SubSwap)
        rewrite_array_ref.visit(my_kernel)

        # print MyKernel
        exchange_indices = exchange.ExchangeId(self.IndexToThreadId)
        exchange_indices.visit(my_kernel)

        exchange_types = exchange.ExchangeTypes()
        exchange_types.visit(my_kernel)

        typeid = copy.deepcopy(self.DevFuncTypeId)
        typeid.type.insert(0, '__kernel')

        ext = copy.deepcopy(self.KernelStruct.Includes)
        newast = lan.FileAST(ext)
        for n in arglist:
            if (len(n.type) == 3 and n.type[1] == 'double') \
                    or (len(n.type) != 3 and n.type[0] == 'double'):
                ext.insert(0, lan.Compound([lan.Id("#pragma OPENCL EXTENSION cl_khr_fp64: enable")]))
                break

        ext.append(lan.FuncDecl(typeid, lan.ArgList(arglist), my_kernel))
        ast.ext = list()
        ast.ext.append(newast)
