import lan
import ast_buildingblock as ast_bb
import snippetgen
import copy
import place_in_reg as pireg
import place_in_local as piloc
import collect_id as ci

def print_dict_sorted(mydict):
    keys = sorted(mydict)

    entries = ""
    for key in keys:
        value = mydict[key]
        entries += "'" + key + "': " + value.__repr__() + ","

    return "{" + entries[:-1] + "}"


class KernelGen(object):
    def __init__(self, ast, fileprefix):
        self.ast = ast
        self.fileprefix = fileprefix

    def generate_kernels(self):
        # Create base version and possible version with Local and
        # Register optimizations
        name = ci.get_program_name(self.ast)
        funcname = name + 'Base'
        ast = copy.deepcopy(self.ast)
        self._create_base_kernel(funcname, ast)

        pir = pireg.PlaceInReg(ast)
        pir.place_in_reg3()
        if pir.perform_transformation:
            funcname = name + 'PlaceInReg'
            self._create_optimized_kernel(funcname, ast, pir.PlaceInRegCond)

        pil = piloc.PlaceInLocal(ast)
        pil.place_in_local()
        if pir.PlaceInRegFinding and pil.PlaceInLocalArgs:
            raise Exception("""GenerateKernels: Currently unimplemented to perform
                                PlaceInReg and PlaceInLocal together from the analysis""")
        for arg in pil.PlaceInLocalArgs:
            funcname = name + 'PlaceInLocal'
            pil.local_memory3(arg)
            self._create_optimized_kernel(funcname, ast, pil.PlaceInLocalCond)

    def __get_file_name(self, funcname):
        name = ci.get_program_name(self.ast)
        return self.fileprefix + name + '/' + funcname + '.cl'

    def _create_base_kernel(self, funcname, testast):
        ss = snippetgen.SnippetGen(testast)
        ss.in_source_kernel(copy.deepcopy(testast), filename=self.__get_file_name(funcname),
                            kernelstringname=funcname)

    def _create_optimized_kernel(self, funcname, testast, cond):
        ss = snippetgen.SnippetGen(testast)
        ss.in_source_kernel(copy.deepcopy(testast), filename=self.__get_file_name(funcname),
                            kernelstringname=funcname)


class CreateKernels(KernelGen):
    def __init__(self, ast, file_ast):
        super(CreateKernels, self).__init__(ast, "dontcate")
        self.file_ast = file_ast

        # Output
        self.IfThenElse = None

    def create_get_kernel_code(self):

        self.generate_kernels()

        get_kernel_code = ast_bb.EmptyFuncDecl('GetKernelCode', type=['std::string'])
        get_kernel_stats = [self.IfThenElse]
        get_kernel_code.compound.statements = get_kernel_stats
        self.file_ast.ext.append(get_kernel_code)

    def _create_base_kernel(self, funcname, testast):
        sg = snippetgen.SnippetGen(self.ast)
        newast = sg.generate_kernel_ss(copy.deepcopy(testast), funcname)
        self.file_ast.ext.append(newast)
        self.IfThenElse = self.__create_base_kernel_func()

    def __create_base_kernel_func(self):
        program_name = ci.get_program_name(self.ast)
        name = program_name + 'Base'
        func = ast_bb.EmptyFuncDecl(name, type=[])
        return lan.Return(func)

    def __create_optimized_kernel_func(self, funcname, cond):
        returnfunc1 = self.__create_base_kernel_func()
        name = funcname
        func = ast_bb.EmptyFuncDecl(name, type=[])
        returnfunc2 = lan.Return(func)
        ifthenelse = lan.IfThenElse(cond,
                                    lan.Compound([returnfunc2]), lan.Compound([returnfunc1]))
        return ifthenelse

    def _create_optimized_kernel(self, funcname, testast, cond):

        sg = snippetgen.SnippetGen(testast)
        newast = sg.generate_kernel_ss(copy.deepcopy(testast), funcname)
        self.file_ast.ext.append(newast)
        self.IfThenElse = self.__create_optimized_kernel_func(funcname, cond)
