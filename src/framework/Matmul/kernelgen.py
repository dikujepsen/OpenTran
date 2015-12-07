import lan
import ast_buildingblock as ast_bb
import snippetgen
import copy


class KernelGen(object):
    def __init__(self, tf, rw, ks):
        self.PlaceInLocalArgs = list()
        self.PlaceInLocalCond = None
        self.PlaceInRegArgs = list()
        self.PlaceInRegCond = None
        self.KernelStringStream = list()
        self.IfThenElse = None
        self.tf = tf
        self.rw = rw
        self.ks = ks

    def GenerateKernels(self, ast, name, fileprefix):
        rw = self.rw
        tf = self.tf
        # Create base version and possible version with Local and
        # Register optimizations
        funcname = name + 'Base'

        if self.PlaceInRegArgs and self.PlaceInLocalArgs:
            raise Exception("""GenerateKernels: Currently unimplemented to perform
                                PlaceInReg and PlaceInLocal together from the analysis""")

        ss = snippetgen.SnippetGen()

        ss.set_datastructure(self.ks,
                             ast)

        ss.InSourceKernel(copy.deepcopy(ast), lan.Id('true'), filename=fileprefix + name + '/' + funcname + '.cl',
                          kernelstringname=funcname)
        for (arg, insideloop) in self.PlaceInRegArgs:
            funcname = name + 'PlaceInReg'
            tf.placeInReg3(arg, list(insideloop))
            ss.InSourceKernel(copy.deepcopy(ast), lan.Id('true'), filename=fileprefix + name + '/' + funcname + '.cl',
                              kernelstringname=funcname)

        for arg in self.PlaceInLocalArgs:
            funcname = name + 'PlaceInLocal'
            tf.localMemory3(arg)
            ss.InSourceKernel(copy.deepcopy(ast), self.PlaceInLocalCond,
                              filename=fileprefix + name + '/' + funcname + '.cl', kernelstringname=funcname)

        self.KernelStringStream = ss.KernelStringStream

        MyCond = None
        if self.PlaceInLocalCond:
            MyCond = self.PlaceInLocalCond
        if self.PlaceInRegCond:
            MyCond = self.PlaceInRegCond

        if MyCond:
            name = self.KernelStringStream[0]['name']
            func = ast_bb.EmptyFuncDecl(name, type=[])
            returnfunc1 = lan.Assignment(lan.Id('return'), func, op='')
            name = self.KernelStringStream[1]['name']
            func = ast_bb.EmptyFuncDecl(name, type=[])
            returnfunc2 = lan.Assignment(lan.Id('return'), func, op='')
            ifthenelse = lan.IfThenElse(MyCond, \
                                        lan.Compound([returnfunc2]), lan.Compound([returnfunc1]))

            self.IfThenElse = ifthenelse
        else:
            name = self.KernelStringStream[0]['name']
            func = ast_bb.EmptyFuncDecl(name, type=[])
            returnfunc1 = lan.Assignment(lan.Id('return'), func, op='')
            self.IfThenElse = returnfunc1
