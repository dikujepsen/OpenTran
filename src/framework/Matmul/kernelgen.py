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


class KernelGen(object):
    def __init__(self):
        pass
        # Output

    def generate_kernels(self, ast, name, fileprefix):
        # Create base version and possible version with Local and
        # Register optimizations
        funcname = name + 'Base'
        ast = copy.deepcopy(ast)
        ss = snippetgen.SnippetGen(ast)

        ss.in_source_kernel(copy.deepcopy(ast), filename=fileprefix + name + '/' + funcname + '.cl',
                            kernelstringname=funcname)

        pir = pireg.PlaceInReg(ast)
        funcname = name + 'PlaceInReg'
        pir.place_in_reg3()
        if pir.perform_transformation:
            ss.in_source_kernel(copy.deepcopy(ast), filename=fileprefix + name + '/' + funcname + '.cl',
                                kernelstringname=funcname)

        pil = piloc.PlaceInLocal(ast)
        pil.place_in_local()
        if pir.PlaceInRegFinding and pil.PlaceInLocalArgs:
            raise Exception("""GenerateKernels: Currently unimplemented to perform
                                PlaceInReg and PlaceInLocal together from the analysis""")
        for arg in pil.PlaceInLocalArgs:
            funcname = name + 'PlaceInLocal'

            pil.local_memory3(arg)
            ss.in_source_kernel(copy.deepcopy(ast), filename=fileprefix + name + '/' + funcname + '.cl',
                                kernelstringname=funcname, cond=pil.PlaceInLocalCond)
