import lan
import collect_transformation_info as cti
import collect_array as ca
import collect_gen as cg


class DefineArguments(object):
    def __init__(self):
        self.type = dict()
        self.name_swap = dict()
        self.ParDim = None  # int

        self.kernel_args = dict()
        self.define_compound = None
        self.ast = None

    def set_datastructures(self, ast):
        self.ast = ast
        self.define_compound = lan.GroupCompound([lan.Comment('// Defines for the kernel')])

        fpl = cti.FindPerfectForLoop()
        fpl.ParDim = self.ParDim
        fpl.collect(ast)

        fai = cti.FindArrayIdsKernel()
        fai.ParDim = self.ParDim
        fai.collect(ast)

        self.type = fai.type
        self.kernel_args = cg.get_kernel_args(ast, fpl.par_dim)


    def define_arguments(self):
        """ Find all kernel arguments that can be defined
            at compilation time. Then defines them.
            :param name_swap - Dict of possible name swaps
        """

        defines = list()
        for n in self.kernel_args:
            if len(self.type[n]) < 2:
                defines.append(n)
                self.ast.ext.append(lan.KernelArgDefine(lan.Id(n)))

        self.__setdefine(defines)

    def __setdefine(self, var_list):
        # TODO: Check that the vars in varlist are actually an argument

        accname = 'str'
        sstream = lan.TypeId(['std::stringstream'], lan.Id(accname))
        stats = self.define_compound.statements
        stats.append(sstream)

        # add the defines to the string stream
        array_dim_swap = ca.get_array_dim_swap(self.ast)
        for var in var_list:
            try:
                hstvar = array_dim_swap[var]
            except KeyError:
                hstvar = var
            add = lan.Id(accname + ' << ' + '\"' + '-D' + var + '=\"' + ' << ' + hstvar + ' << \" \";')
            stats.append(add)

        # Set the string to the global variable
        lval = lan.Id('KernelDefines')
        stats.append(lan.Assignment(lval, lan.Id(accname + '.str()')))
