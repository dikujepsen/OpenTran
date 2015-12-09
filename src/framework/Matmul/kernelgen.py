import lan
import ast_buildingblock as ast_bb
import snippetgen
import copy
import place_in_reg as pireg
import place_in_local as piloc

class KernelGen(object):
    def __init__(self, tf, ks):
        self.PlaceInLocalArgs = list()
        self.PlaceInLocalCond = None
        self.PlaceInRegArgs = list()
        self.PlaceInRegCond = None
        self.KernelStringStream = list()
        self.IfThenElse = None
        self.tf = tf
        self.ks = ks

    def GenerateKernels(self, ast, name, fileprefix):
        tf = self.tf
        # Create base version and possible version with Local and
        # Register optimizations
        funcname = name + 'Base'

        if self.PlaceInRegArgs and self.PlaceInLocalArgs:
            raise Exception("""GenerateKernels: Currently unimplemented to perform
                                PlaceInReg and PlaceInLocal together from the analysis""")

        ss = snippetgen.SnippetGen()

        ss.set_datastructure(self.ks, ast)

        ss.InSourceKernel(copy.deepcopy(ast), lan.Id('true'), filename=fileprefix + name + '/' + funcname + '.cl',
                          kernelstringname=funcname)
        pir = pireg.PlaceInReg()
        for (arg, insideloop) in self.PlaceInRegArgs:
            funcname = name + 'PlaceInReg'
            pir.placeInReg3(self.ks, arg, list(insideloop))
            ss.InSourceKernel(copy.deepcopy(ast), lan.Id('true'), filename=fileprefix + name + '/' + funcname + '.cl',
                              kernelstringname=funcname)

        pil = piloc.PlaceInLocal()
        for arg in self.PlaceInLocalArgs:
            funcname = name + 'PlaceInLocal'

            pil.localMemory3(self.ks, arg)
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
