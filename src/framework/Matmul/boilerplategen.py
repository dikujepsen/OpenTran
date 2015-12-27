import lan
import copy
import ast_buildingblock as ast_bb
import collect_transformation_info as cti


class Boilerplate(object):
    def __init__(self):
        self.ks = None
        self.bps = None
        self.kgen_strt = None

        self.Local = list()
        self.GridIndices = list()
        self.UpperLimit = dict()

    def set_struct(self, kernelstruct, boilerplatestruct, kgen_strt, ast):
        self.ks = kernelstruct
        self.bps = boilerplatestruct
        self.kgen_strt = kgen_strt

        fl = cti.FindLocal()
        fl.ParDim = self.ks.ParDim
        fl.collect(ast)
        self.Local = fl.Local

        fpl = cti.FindGridIndices()
        fpl.ParDim = self.ks.ParDim
        fpl.collect(ast)
        self.GridIndices = fpl.GridIndices

        fai = cti.FindReadWrite()
        fai.ParDim = self.ks.ParDim
        fai.collect(ast)

        self.UpperLimit = fai.upper_limit



    def generate_code(self):

        dictNToDimNames = self.ks.ArrayIdToDimName

        NonArrayIds = copy.deepcopy(self.bps.NonArrayIds)

        fileAST = lan.FileAST([])

        fileAST.ext.append(lan.Id('#include \"../../../utils/StartUtil.cpp\"'))
        fileAST.ext.append(lan.Id('using namespace std;'))

        kernelId = lan.Id(self.bps.KernelName)
        kernelTypeid = lan.TypeId(['cl_kernel'], kernelId, 0)
        fileAST.ext.append(kernelTypeid)

        listDevBuffers = []

        for n in self.ks.ArrayIds:
            try:
                name = self.bps.DevId[n]
                listDevBuffers.append(lan.TypeId(['cl_mem'], lan.Id(name)))
            except KeyError:
                pass


        dictNToDevPtr = self.bps.DevId
        listDevBuffers = lan.GroupCompound(listDevBuffers)

        fileAST.ext.append(listDevBuffers)

        listHostPtrs = []
        for n in self.bps.DevArgList:
            name = n.name.name
            type = self.ks.Type[name]
            try:
                name = self.bps.HstId[name]
            except KeyError:
                pass
            listHostPtrs.append(lan.TypeId(type, lan.Id(name), 0))

        for n in self.bps.GlobalVars:
            type = self.ks.Type[n]
            name = self.bps.HstId[n]
            listHostPtrs.append(lan.TypeId(type, lan.Id(name), 0))

        dictNToHstPtr = self.bps.HstId
        dictTypeHostPtrs = copy.deepcopy(self.ks.Type)
        listHostPtrs = lan.GroupCompound(listHostPtrs)
        fileAST.ext.append(listHostPtrs)

        listMemSize = []
        listDimSize = []
        dictNToSize = self.bps.Mem
        for n in self.bps.Mem:
            sizeName = self.bps.Mem[n]
            listMemSize.append(lan.TypeId(['size_t'], lan.Id(sizeName)))

        for n in self.ks.ArrayIds:
            for dimName in self.ks.ArrayIdToDimName[n]:
                listDimSize.append( \
                    lan.TypeId(['size_t'], lan.Id(dimName)))

        fileAST.ext.append(lan.GroupCompound(listMemSize))
        fileAST.ext.append(lan.GroupCompound(listDimSize))
        misc = []
        lval = lan.TypeId(['size_t'], lan.Id('isFirstTime'))
        rval = lan.Constant(1)
        misc.append(lan.Assignment(lval, rval))

        lval = lan.TypeId(['std::string'], lan.Id('KernelDefines'))
        rval = lan.Constant('""')
        misc.append(lan.Assignment(lval, rval))

        lval = lan.TypeId(['Stopwatch'], lan.Id('timer'))
        misc.append(lval)

        fileAST.ext.append(lan.GroupCompound(misc))

        # Generate the GetKernelCode function
        for optim in self.kgen_strt.KernelStringStream:
            fileAST.ext.append(optim['ast'])

        getKernelCode = ast_bb.EmptyFuncDecl('GetKernelCode', type=['std::string'])
        getKernelStats = []
        getKernelCode.compound.statements = getKernelStats
        getKernelStats.append(self.kgen_strt.IfThenElse)
        fileAST.ext.append(getKernelCode)

        allocateBuffer = ast_bb.EmptyFuncDecl('AllocateBuffers')
        fileAST.ext.append(allocateBuffer)

        listSetMemSize = []
        for entry in self.ks.ArrayIds:
            n = self.ks.ArrayIdToDimName[entry]
            lval = lan.Id(self.bps.Mem[entry])
            rval = lan.BinOp(lan.Id(n[0]), '*', lan.Id('sizeof(' + \
                                                       self.ks.Type[entry][0] + ')'))
            if len(n) == 2:
                rval = lan.BinOp(lan.Id(n[1]), '*', rval)
            listSetMemSize.append(lan.Assignment(lval, rval))

        allocateBuffer.compound.statements.append( \
            lan.GroupCompound(listSetMemSize))

        allocateBuffer.compound.statements.append( \
            self.bps.Transposition)

        allocateBuffer.compound.statements.append( \
            lan.GroupCompound([lan.Comment('// Constant Memory')]))

        allocateBuffer.compound.statements.append( \
            self.bps.define_compound)

        ErrName = 'oclErrNum'
        lval = lan.TypeId(['cl_int'], lan.Id(ErrName))
        rval = lan.Id('CL_SUCCESS')
        clSuc = lan.Assignment(lval, rval)
        allocateBuffer.compound.statements.extend( \
            [lan.GroupCompound([clSuc])])

        for n in dictNToDevPtr:
            lval = lan.Id(dictNToDevPtr[n])
            op = '='
            arrayn = dictNToHstPtr[n]
            try:
                arrayn = self.bps.NameSwap[arrayn]
            except KeyError:
                pass
            if n in self.bps.WriteOnly:
                flag = lan.Id('CL_MEM_WRITE_ONLY')
                arraynId = lan.Id('NULL')
            elif n in self.bps.ReadOnly:
                flag = lan.Id('CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY')
                arraynId = lan.Id(arrayn)
            else:
                flag = lan.Id('CL_MEM_USE_HOST_PTR')
                arraynId = lan.Id(arrayn)

            arglist = lan.ArgList([lan.Id('context'), \
                                   flag, \
                                   lan.Id(dictNToSize[n]), \
                                   arraynId, \
                                   lan.Id('&' + ErrName)])
            rval = lan.FuncDecl(lan.Id('clCreateBuffer'), arglist, lan.Compound([]))
            allocateBuffer.compound.statements.append( \
                lan.Assignment(lval, rval))
            arglist = lan.ArgList([lan.Id(ErrName), lan.Constant("clCreateBuffer " + lval.name)])
            ErrCheck = lan.FuncDecl(lan.Id('oclCheckErr'), arglist, lan.Compound([]))
            allocateBuffer.compound.statements.append(ErrCheck)

        setArgumentsKernel = ast_bb.EmptyFuncDecl('SetArguments' + self.bps.DevFuncId)
        fileAST.ext.append(setArgumentsKernel)
        ArgBody = setArgumentsKernel.compound.statements
        ArgBody.append(clSuc)
        cntName = lan.Id('counter')
        lval = lan.TypeId(['int'], cntName)
        rval = lan.Constant(0)
        ArgBody.append(lan.Assignment(lval, rval))

        for n in dictNToDimNames:
            ## add dim arguments to set of ids
            NonArrayIds.add(dictNToDimNames[n][0])
            # Add types of dimensions for size arguments
            dictTypeHostPtrs[dictNToDimNames[n][0]] = ['size_t']

        for n in self.bps.RemovedIds:
            dictTypeHostPtrs.pop(n, None)

        ## clSetKernelArg for Arrays
        for n in self.ks.KernelArgs:
            lval = lan.Id(ErrName)
            op = '|='
            type = self.ks.Type[n]
            if len(type) == 2:
                arglist = lan.ArgList([kernelId, \
                                       lan.Increment(cntName, '++'), \
                                       lan.Id('sizeof(cl_mem)'), \
                                       lan.Id('(void *) &' + dictNToDevPtr[n])])
                rval = lan.FuncDecl(lan.Id('clSetKernelArg'), arglist, lan.Compound([]))
            else:
                try:
                    n = self.bps.NameSwap[n]
                except KeyError:
                    pass
                cl_type = type[0]
                if cl_type == 'size_t':
                    cl_type = 'unsigned'
                arglist = lan.ArgList([kernelId, \
                                       lan.Increment(cntName, '++'), \
                                       lan.Id('sizeof(' + cl_type + ')'), \
                                       lan.Id('(void *) &' + n)])
                rval = lan.FuncDecl(lan.Id('clSetKernelArg'), arglist, lan.Compound([]))
            ArgBody.append(lan.Assignment(lval, rval, op))

        arglist = lan.ArgList([lan.Id(ErrName), lan.Constant('clSetKernelArg')])
        ErrId = lan.Id('oclCheckErr')
        ErrCheck = lan.FuncDecl(ErrId, arglist, lan.Compound([]))
        ArgBody.append(ErrCheck)

        execKernel = ast_bb.EmptyFuncDecl('Exec' + self.bps.DevFuncTypeId.name.name)
        fileAST.ext.append(execKernel)
        execBody = execKernel.compound.statements
        execBody.append(clSuc)
        eventName = lan.Id('GPUExecution')
        event = lan.TypeId(['cl_event'], eventName)
        execBody.append(event)

        for n in self.bps.Worksize:
            lval = lan.TypeId(['size_t'], lan.Id(self.bps.Worksize[n] + '[]'))
            if n == 'local':
                local_worksize = [lan.Id(i) for i in self.Local['size']]
                rval = lan.ArrayInit(local_worksize)
            elif n == 'global':
                initlist = []
                for m in reversed(self.GridIndices):
                    initlist.append(lan.Id(self.UpperLimit[m] \
                                           + ' - ' + self.bps.LowerLimit[m]))
                rval = lan.ArrayInit(initlist)
            else:
                initlist = []
                for m in reversed(self.GridIndices):
                    initlist.append(lan.Id(self.bps.LowerLimit[m]))
                rval = lan.ArrayInit(initlist)

            execBody.append(lan.Assignment(lval, rval))

        lval = lan.Id(ErrName)
        arglist = lan.ArgList([lan.Id('command_queue'), \
                               lan.Id(self.bps.KernelName), \
                               lan.Constant(self.ks.ParDim), \
                               lan.Id(self.bps.Worksize['offset']), \
                               lan.Id(self.bps.Worksize['global']), \
                               lan.Id(self.bps.Worksize['local']), \
                               lan.Constant(0), lan.Id('NULL'), \
                               lan.Id('&' + eventName.name)])
        rval = lan.FuncDecl(lan.Id('clEnqueueNDRangeKernel'), arglist, lan.Compound([]))
        execBody.append(lan.Assignment(lval, rval))

        arglist = lan.ArgList([lan.Id(ErrName), lan.Constant('clEnqueueNDRangeKernel')])
        ErrCheck = lan.FuncDecl(ErrId, arglist, lan.Compound([]))
        execBody.append(ErrCheck)

        arglist = lan.ArgList([lan.Id('command_queue')])
        finish = lan.FuncDecl(lan.Id('clFinish'), arglist, lan.Compound([]))
        execBody.append(lan.Assignment(lan.Id(ErrName), finish))

        arglist = lan.ArgList([lan.Id(ErrName), lan.Constant('clFinish')])
        ErrCheck = lan.FuncDecl(ErrId, arglist, lan.Compound([]))
        execBody.append(ErrCheck)

        if not self.bps.NoReadBack:
            for n in self.bps.WriteOnly:
                lval = lan.Id(ErrName)
                Hstn = self.bps.HstId[n]
                try:
                    Hstn = self.bps.NameSwap[Hstn]
                except KeyError:
                    pass
                arglist = lan.ArgList([lan.Id('command_queue'), \
                                       lan.Id(self.bps.DevId[n]), \
                                       lan.Id('CL_TRUE'), \
                                       lan.Constant(0), \
                                       lan.Id(self.bps.Mem[n]), \
                                       lan.Id(Hstn), \
                                       lan.Constant(1),
                                       lan.Id('&' + eventName.name), lan.Id('NULL')])
                rval = lan.FuncDecl(lan.Id('clEnqueueReadBuffer'), arglist, lan.Compound([]))
                execBody.append(lan.Assignment(lval, rval))

                arglist = lan.ArgList([lan.Id(ErrName), lan.Constant('clEnqueueReadBuffer')])
                ErrCheck = lan.FuncDecl(ErrId, arglist, lan.Compound([]))
                execBody.append(ErrCheck)

            # add clFinish statement
            arglist = lan.ArgList([lan.Id('command_queue')])
            finish = lan.FuncDecl(lan.Id('clFinish'), arglist, lan.Compound([]))
            execBody.append(lan.Assignment(lan.Id(ErrName), finish))

            arglist = lan.ArgList([lan.Id(ErrName), lan.Constant('clFinish')])
            ErrCheck = lan.FuncDecl(ErrId, arglist, lan.Compound([]))
            execBody.append(ErrCheck)

            for n in self.bps.WriteTranspose:
                execBody.append(n)

        runOCL = ast_bb.EmptyFuncDecl('RunOCL' + self.bps.KernelName)
        fileAST.ext.append(runOCL)
        runOCLBody = runOCL.compound.statements

        argIds = self.bps.NonArrayIds.union(self.ks.ArrayIds)  #

        typeIdList = []
        ifThenList = []
        for n in argIds:
            type = self.ks.Type[n]
            argn = lan.Id('arg_' + n)
            typeIdList.append(lan.TypeId(type, argn))
            try:
                newn = self.bps.HstId[n]
            except KeyError:
                newn = n
            lval = lan.Id(newn)
            rval = argn
            ifThenList.append(lan.Assignment(lval, rval))
            try:
                for m in self.ks.ArrayIdToDimName[n]:
                    type = ['size_t']
                    argm = lan.Id('arg_' + m)
                    lval = lan.Id(m)
                    rval = argm
                    ifThenList.append(lan.Assignment(lval, rval))
                    typeIdList.append(lan.TypeId(type, argm))
            except KeyError:
                pass

        arglist = lan.ArgList(typeIdList)
        runOCL.arglist = arglist

        arglist = lan.ArgList([])
        ifThenList.append(lan.FuncDecl(lan.Id('StartUpGPU'), arglist, lan.Compound([])))
        ifThenList.append(lan.FuncDecl(lan.Id('AllocateBuffers'), arglist, lan.Compound([])))
        useFile = 'true'
        if self.kgen_strt.KernelStringStream:
            useFile = 'false'

        ifThenList.append(lan.Id('cout << "$Defines " << KernelDefines << endl;'))
        arglist = lan.ArgList([lan.Constant(self.bps.DevFuncId),
                               lan.Constant(self.bps.DevFuncId + '.cl'),
                               lan.Id('GetKernelCode()'),
                               lan.Id(useFile),
                               lan.Id('&' + self.bps.KernelName),
                               lan.Id('KernelDefines')])
        ifThenList.append(lan.FuncDecl(lan.Id('compileKernel'), arglist, lan.Compound([])))
        ifThenList.append(
            lan.FuncDecl(lan.Id('SetArguments' + self.bps.DevFuncId), lan.ArgList([]), lan.Compound([])))

        runOCLBody.append(lan.IfThen(lan.Id('isFirstTime'), lan.Compound(ifThenList)))
        arglist = lan.ArgList([])

        # Insert timing
        runOCLBody.append(lan.Id('timer.start();'))
        runOCLBody.append(lan.FuncDecl(lan.Id('Exec' + self.bps.DevFuncId), arglist, lan.Compound([])))
        runOCLBody.append(lan.Id('cout << "$Time " << timer.stop() << endl;'))

        return fileAST
