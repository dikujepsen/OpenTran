import lan
import ast_buildingblock as ast_bb
import snippetgen
import copy
import place_in_reg as pireg
import place_in_local as piloc

class KernelGen(object):
    def __init__(self, ks):
        self.ks = ks

        # Output
        self.KernelStringStream = list()
        self.IfThenElse = None


    def GenerateKernels(self, ast, name, fileprefix):
        # Create base version and possible version with Local and
        # Register optimizations
        funcname = name + 'Base'

        if self.ks.PlaceInRegArgs and self.ks.PlaceInLocalArgs:
            raise Exception("""GenerateKernels: Currently unimplemented to perform
                                PlaceInReg and PlaceInLocal together from the analysis""")

        ss = snippetgen.SnippetGen()

        ss.set_datastructure(self.ks, ast)

        ss.InSourceKernel(copy.deepcopy(ast), lan.Id('true'), filename=fileprefix + name + '/' + funcname + '.cl',
                          kernelstringname=funcname)
        pir = pireg.PlaceInReg()
        for (arg, insideloop) in self.ks.PlaceInRegArgs:
            funcname = name + 'PlaceInReg'
            pir.placeInReg3(self.ks, arg, list(insideloop))
            ss.InSourceKernel(copy.deepcopy(ast), lan.Id('true'), filename=fileprefix + name + '/' + funcname + '.cl',
                              kernelstringname=funcname)

        pil = piloc.PlaceInLocal()
        for arg in self.ks.PlaceInLocalArgs:
            funcname = name + 'PlaceInLocal'

            pil.localMemory3(self.ks, arg)
            ss.InSourceKernel(copy.deepcopy(ast), self.ks.PlaceInLocalCond,
                              filename=fileprefix + name + '/' + funcname + '.cl', kernelstringname=funcname)

        self.KernelStringStream = ss.KernelStringStream

        MyCond = None
        if self.ks.PlaceInLocalCond:
            MyCond = self.ks.PlaceInLocalCond
        if self.ks.PlaceInRegCond:
            MyCond = self.ks.PlaceInRegCond

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
