import lan
import copy
import transf_visitor
import stringstream
import visitor
import transf_visitor as tvisitor



class SnippetGen(object):
    def __init__(self):
        self.KernelStruct = None
        self.KernelStringStream = list()
        self.IndexToThreadId = dict()
        self.DevFuncTypeId = None

    def set_datastructure(self,
                          KernelStruct,
                          ast):
        self.KernelStruct = KernelStruct
        perfect_for_loop = tvisitor.PerfectForLoop()
        perfect_for_loop.visit(ast)

        self.ParDim = self.KernelStruct.ParDim

        init_ids = tvisitor.InitIds()
        init_ids.visit(perfect_for_loop.ast.init)
        first_idx = init_ids.index[0]

        grid_ids = list()
        id_map = dict()
        id_map[first_idx] = 'get_global_id(0)'
        grid_ids.extend(init_ids.index)
        kernel = perfect_for_loop.ast.compound
        if self.ParDim == 2:
            init_ids = tvisitor.InitIds()
            init_ids.visit(kernel.statements[0].init)
            second_idx = init_ids.index[0]
            id_map[second_idx] = 'get_global_id(1)'
            grid_ids.extend(init_ids.index)
            (id_map[grid_ids[0]], id_map[grid_ids[1]]) = (id_map[grid_ids[1]], id_map[grid_ids[0]])

        self.IndexToThreadId = id_map

        find_function = tvisitor.FindFunction()
        find_function.visit(ast)
        self.DevFuncTypeId = find_function.typeid

    def InSourceKernel(self, ast, cond, filename, kernelstringname):
        self.rewriteToDeviceCRelease(ast)

        ssprint = stringstream.SSGenerator()
        emptyast = lan.FileAST([])
        ssprint.createKernelStringStream(ast, emptyast, kernelstringname, filename=filename)
        self.KernelStringStream.append({'name': kernelstringname, \
                                        'ast': emptyast,
                                        'cond': cond})

    def rewriteToDeviceCRelease(self, ast):
        arglist = list()
        # The list of arguments for the kernel
        dictTypeHostPtrs = copy.deepcopy(self.KernelStruct.Type)
        for n in self.KernelStruct.ArrayIds:
            dictTypeHostPtrs[self.KernelStruct.ArrayIdToDimName[n][0]] = ['size_t']

        for n in self.KernelStruct.KernelArgs:
            type = copy.deepcopy(self.KernelStruct.KernelArgs[n])
            if type[0] == 'size_t':
                type[0] = 'unsigned'
            if len(type) == 2:
                type.insert(0, '__global')
            arglist.append(lan.TypeId(type, lan.Id(n)))

        exchangeArrayId = transf_visitor.ExchangeArrayId(self.KernelStruct.LocalSwap)

        for n in self.KernelStruct.LoopArrays.values():
            for m in n:
                exchangeArrayId.visit(m)

        MyKernel = copy.deepcopy(self.KernelStruct.Kernel)
        # print self.astrepr.ArrayIdToDimName
        rewriteArrayRef = transf_visitor.RewriteArrayRef(self.KernelStruct.num_array_dims,
                                                         self.KernelStruct.ArrayIdToDimName,
                                                         self.KernelStruct.SubSwap)
        rewriteArrayRef.visit(MyKernel)

        # print MyKernel
        exchangeIndices = transf_visitor.ExchangeId(self.IndexToThreadId)
        exchangeIndices.visit(MyKernel)

        exchangeTypes = transf_visitor.ExchangeTypes()
        exchangeTypes.visit(MyKernel)

        typeid = copy.deepcopy(self.DevFuncTypeId)
        typeid.type.insert(0, '__kernel')

        ext = copy.deepcopy(self.KernelStruct.Includes)
        newast = lan.FileAST(ext)
        for n in arglist:
            if (len(n.type) == 3 and n.type[1] == 'double') \
                    or (len(n.type) != 3 and n.type[0] == 'double'):
                ext.insert(0, lan.Compound([lan.Id("#pragma OPENCL EXTENSION cl_khr_fp64: enable")]))
                break

        ext.append(lan.FuncDecl(typeid, lan.ArgList(arglist), MyKernel))
        ast.ext = list()
        ast.ext.append(newast)
