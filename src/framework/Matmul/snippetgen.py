
import lan
import copy
import transf_visitor
import stringstream

class SnippetGen(object):

    def __init__(self, astrepr):
        self.astrepr = astrepr

    def InSourceKernel(self, ast, cond, filename, kernelstringname):
        self.rewriteToDeviceCRelease(ast)

        ssprint = stringstream.SSGenerator()
        newast = lan.FileAST([])
        ssprint.createKernelStringStream(ast, newast, self.astrepr.UnrollLoops, kernelstringname, filename = filename)
        self.astrepr.KernelStringStream.append({'name' : kernelstringname, \
                                        'ast' : newast,
                                        'cond' : cond})

    def rewriteToDeviceCRelease(self, ast):
        initrepr = self.astrepr.astrepr
        arglist = list()
        argIds = initrepr.NonArrayIds.union(initrepr.ArrayIds) - self.astrepr.RemovedIds
        # The list of arguments for the kernel
        dictTypeHostPtrs = copy.deepcopy(initrepr.Type)
        for n in initrepr.ArrayIds:
            dictTypeHostPtrs[self.astrepr.ArrayIdToDimName[n][0]] = ['size_t']

        for n in self.astrepr.KernelArgs:
            type = copy.deepcopy(self.astrepr.KernelArgs[n])
            if type[0] == 'size_t':
                type[0] = 'unsigned'
            if len(type) == 2:
                type.insert(0, '__global')
            arglist.append(lan.TypeId(type, lan.Id(n)))


        exchangeArrayId = transf_visitor.ExchangeArrayId(self.astrepr.LocalSwap)

        for n in self.astrepr.LoopArrays.values():
            for m in n:
                exchangeArrayId.visit(m)


        MyKernel = copy.deepcopy(self.astrepr.Kernel)
        rewriteArrayRef = transf_visitor.RewriteArrayRef(initrepr.num_array_dims,
                                                         self.astrepr.ArrayIdToDimName, self.astrepr)
        rewriteArrayRef.visit(MyKernel)


        exchangeIndices = transf_visitor.ExchangeId(self.astrepr.IndexToThreadId)
        exchangeIndices.visit(MyKernel)


        exchangeTypes = transf_visitor.ExchangeTypes()
        exchangeTypes.visit(MyKernel)


        typeid = copy.deepcopy(self.astrepr.DevFuncTypeId)
        typeid.type.insert(0, '__kernel')

        ext = copy.deepcopy(initrepr.Includes)
        newast = lan.FileAST(ext)
        for n in arglist:
            if (len(n.type) == 3 and n.type[1] == 'double') \
             or (len(n.type) != 3 and n.type[0] == 'double'):
                ext.insert(0, lan.Compound([lan.Id("#pragma OPENCL EXTENSION cl_khr_fp64: enable")]))
                break

        ext.append(lan.FuncDecl(typeid, lan.ArgList(arglist), MyKernel))
        ast.ext = list()
        ast.ext.append(newast)

