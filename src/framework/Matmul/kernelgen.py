import lan
import ast_buildingblock as ast_bb
import snippetgen
import copy
import place_in_reg as pireg
import place_in_local as piloc

def print_dict_sorted(mydict):
    keys = sorted(mydict)

    entries = ""
    for key in keys:
        value = mydict[key]
        entries += "'" + key + "': " + value.__repr__() + ","

    return "{" + entries[:-1] + "}"

class KernelGenStruct(object):
    def __int__(self):
        self.KernelStringStream = list()
        self.IfThenElse = None


class KernelGen(object):
    def __init__(self, ks):
        self.ks = ks

        # Output
        self.kgen_strt = KernelGenStruct()

    def generate_kernels(self, ast, name, fileprefix):
        # Create base version and possible version with Local and
        # Register optimizations
        funcname = name + 'Base'

        ss = snippetgen.SnippetGen()

        ss.set_datastructure(self.ks, ast)

        ss.in_source_kernel(copy.deepcopy(ast), lan.Id('true'), filename=fileprefix + name + '/' + funcname + '.cl',
                            kernelstringname=funcname)

        pir = pireg.PlaceInReg()
        funcname = name + 'PlaceInReg'
        pir.place_in_reg3(ast, self.ks.ParDim, self.ks)
        if pir.perform_transformation:
            ss.in_source_kernel(copy.deepcopy(ast), lan.Id('true'), filename=fileprefix + name + '/' + funcname + '.cl',
                                kernelstringname=funcname)

        if pir.PlaceInRegFinding and self.ks.PlaceInLocalArgs:
            raise Exception("""GenerateKernels: Currently unimplemented to perform
                                PlaceInReg and PlaceInLocal together from the analysis""")

        pil = piloc.PlaceInLocal()
        pil.set_datastructures(ast)
        for arg in self.ks.PlaceInLocalArgs:
            funcname = name + 'PlaceInLocal'

            pil.local_memory3(self.ks, arg)
            ss.in_source_kernel(copy.deepcopy(ast), self.ks.PlaceInLocalCond,
                                filename=fileprefix + name + '/' + funcname + '.cl', kernelstringname=funcname)

        self.kgen_strt.KernelStringStream = ss.KernelStringStream

        my_cond = None
        if self.ks.PlaceInLocalCond:
            my_cond = self.ks.PlaceInLocalCond
        if pir.PlaceInRegCond:
            my_cond = pir.PlaceInRegCond

        if my_cond:
            name = self.kgen_strt.KernelStringStream[0]['name']
            func = ast_bb.EmptyFuncDecl(name, type=[])
            returnfunc1 = lan.Assignment(lan.Id('return'), func, op='')
            name = self.kgen_strt.KernelStringStream[1]['name']
            func = ast_bb.EmptyFuncDecl(name, type=[])
            returnfunc2 = lan.Assignment(lan.Id('return'), func, op='')
            ifthenelse = lan.IfThenElse(my_cond,
                                        lan.Compound([returnfunc2]), lan.Compound([returnfunc1]))

            self.kgen_strt.IfThenElse = ifthenelse
        else:
            name = self.kgen_strt.KernelStringStream[0]['name']
            func = ast_bb.EmptyFuncDecl(name, type=[])
            returnfunc1 = lan.Assignment(lan.Id('return'), func, op='')
            self.kgen_strt.IfThenElse = returnfunc1
