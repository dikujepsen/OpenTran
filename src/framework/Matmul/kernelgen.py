import lan
import ast_buildingblock as ast_bb
import snippetgen
import copy
import place_in_reg as pireg
import place_in_local as piloc
import collect_device as cd

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
    def __init__(self, par_dim):
        self.ParDim = par_dim

        # Output
        self.kgen_strt = KernelGenStruct()

    def generate_kernels(self, ast, name, fileprefix):
        # Create base version and possible version with Local and
        # Register optimizations
        funcname = name + 'Base'

        ss = snippetgen.SnippetGen()

        ss.set_datastructure(self.ParDim, ast)

        ss.in_source_kernel(copy.deepcopy(ast), lan.Id('true'), filename=fileprefix + name + '/' + funcname + '.cl',
                            kernelstringname=funcname)

        pir = pireg.PlaceInReg()
        funcname = name + 'PlaceInReg'
        pir.place_in_reg3(ast, self.ParDim,)
        if pir.perform_transformation:
            ss.in_source_kernel(copy.deepcopy(ast), lan.Id('true'), filename=fileprefix + name + '/' + funcname + '.cl',
                                kernelstringname=funcname)

        pil = piloc.PlaceInLocal()
        pil.set_datastructures(ast)
        pil.place_in_local()
        if pir.PlaceInRegFinding and pil.PlaceInLocalArgs:
            raise Exception("""GenerateKernels: Currently unimplemented to perform
                                PlaceInReg and PlaceInLocal together from the analysis""")
        for arg in pil.PlaceInLocalArgs:
            funcname = name + 'PlaceInLocal'

            pil.local_memory3(arg)
            ss.in_source_kernel(copy.deepcopy(ast), pil.PlaceInLocalCond,
                                filename=fileprefix + name + '/' + funcname + '.cl', kernelstringname=funcname)

        self.kgen_strt.KernelStringStream = ss.KernelStringStream

        my_cond = None
        if pil.PlaceInLocalCond:
            my_cond = pil.PlaceInLocalCond
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
