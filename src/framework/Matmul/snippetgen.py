import lan
import copy
import transf_visitor
import stringstream


class SnippetGen(object):
    def __init__(self):
        self.KernelStringStream = list()
        self.RemovedIds = dict()
        self.ArrayIdToDimName = dict()
        self.NonArrayIds = set()
        self.Type = dict()
        self.ArrayIds = set()
        self.KernelArgs = dict()
        self.LocalSwap = dict()
        self.LoopArrays = dict()
        self.Kernel = None
        self.IndexToThreadId = dict()
        self.DevFuncTypeId = None
        self.Includes = list()
        self.num_array_dims = dict()
        self.SubSwap = dict()

    def set_datastructure(self, KernelStringStream,
                          RemovedIds,
                          ArrayIdToDimName,
                          NonArrayIds,
                          Type,
                          ArrayIds,
                          KernelArgs,
                          LocalSwap,
                          LoopArrays,
                          Kernel,
                          IndexToThreadId,
                          DevFuncTypeId,
                          Includes,
                          num_array_dims,
                          SubSwap):
        self.KernelStringStream = KernelStringStream
        self.RemovedIds = RemovedIds
        self.ArrayIdToDimName = ArrayIdToDimName
        self.NonArrayIds = NonArrayIds
        self.Type = Type
        self.ArrayIds = ArrayIds
        self.KernelArgs = KernelArgs
        self.LocalSwap = LocalSwap
        self.LoopArrays = LoopArrays
        self.Kernel = Kernel
        self.IndexToThreadId = IndexToThreadId
        self.DevFuncTypeId = DevFuncTypeId
        self.Includes = Includes
        self.num_array_dims = num_array_dims
        self.SubSwap = SubSwap

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
        dictTypeHostPtrs = copy.deepcopy(self.Type)
        for n in self.ArrayIds:
            dictTypeHostPtrs[self.ArrayIdToDimName[n][0]] = ['size_t']

        for n in self.KernelArgs:
            type = copy.deepcopy(self.KernelArgs[n])
            if type[0] == 'size_t':
                type[0] = 'unsigned'
            if len(type) == 2:
                type.insert(0, '__global')
            arglist.append(lan.TypeId(type, lan.Id(n)))

        exchangeArrayId = transf_visitor.ExchangeArrayId(self.LocalSwap)

        for n in self.LoopArrays.values():
            for m in n:
                exchangeArrayId.visit(m)

        MyKernel = copy.deepcopy(self.Kernel)
        # print self.astrepr.ArrayIdToDimName
        rewriteArrayRef = transf_visitor.RewriteArrayRef(self.num_array_dims,
                                                         self.ArrayIdToDimName,
                                                         self.SubSwap)
        rewriteArrayRef.visit(MyKernel)

        # print MyKernel
        exchangeIndices = transf_visitor.ExchangeId(self.IndexToThreadId)
        exchangeIndices.visit(MyKernel)

        exchangeTypes = transf_visitor.ExchangeTypes()
        exchangeTypes.visit(MyKernel)

        typeid = copy.deepcopy(self.DevFuncTypeId)
        typeid.type.insert(0, '__kernel')

        ext = copy.deepcopy(self.Includes)
        newast = lan.FileAST(ext)
        for n in arglist:
            if (len(n.type) == 3 and n.type[1] == 'double') \
                    or (len(n.type) != 3 and n.type[0] == 'double'):
                ext.insert(0, lan.Compound([lan.Id("#pragma OPENCL EXTENSION cl_khr_fp64: enable")]))
                break

        ext.append(lan.FuncDecl(typeid, lan.ArgList(arglist), MyKernel))
        ast.ext = list()
        ast.ext.append(newast)
