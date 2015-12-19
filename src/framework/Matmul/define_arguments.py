import lan
import transf_visitor as tvisitor
import visitor
import collect_transformation_info as cti


class DefineArguments(object):
    def __init__(self):
        self.type = dict()
        self.name_swap = dict()
        self.ParDim = None  # int

        self.kernel_args = dict()
        self.define_compound = None

    def set_datastructures(self, ast):

        self.define_compound = lan.GroupCompound([lan.Comment('// Defines for the kernel')])

        fpl = cti.FindPerfectForLoop()
        fpl.ParDim = self.ParDim
        fpl.collect(ast)

        fai = cti.FindArrayIds()
        fai.ParDim = self.ParDim
        fai.collect(ast)

        self.type = fai.type
        self.kernel_args = fai.kernel_args

    def define_arguments(self, name_swap):
        """ Find all kernel arguments that can be defined
            at compilation time. Then defines them.
            :param name_swap - Dict of possible name swaps
        """

        defines = list()
        for n in self.kernel_args:
            if len(self.type[n]) < 2:
                defines.append(n)

        self.__setdefine(defines, name_swap)

    def __setdefine(self, var_list, name_swap):
        # TODO: Check that the vars in varlist are actually an argument

        accname = 'str'
        sstream = lan.TypeId(['std::stringstream'], lan.Id(accname))
        stats = self.define_compound.statements
        stats.append(sstream)

        # add the defines to the string stream
        for var in var_list:
            try:
                hstvar = name_swap[var]
            except KeyError:
                hstvar = var
            add = lan.Id(accname + ' << ' + '\"' + '-D' + var + '=\"' + ' << ' + hstvar + ' << \" \";')
            stats.append(add)

        # Set the string to the global variable
        lval = lan.Id('KernelDefines')
        stats.append(lan.Assignment(lval, lan.Id(accname + '.str()')))

        # Need to remove the corresponding kernel arguments
        for var in var_list:
            self.kernel_args.pop(var)
