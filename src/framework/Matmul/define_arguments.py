import lan
import collect_transformation_info as cti
import collect_array as ca
import collect_gen as cg
import collect_id as ci
import collect_loop as cl


class DefineArguments(object):
    def __init__(self):
        self.ParDim = None  # int

        self.kernel_args = dict()
        self.ast = None

    def set_datastructures(self, ast):
        self.ParDim = cl.get_par_dim(ast)
        self.ast = ast

        fpl = cti.FindPerfectForLoop()
        fpl.ParDim = self.ParDim
        fpl.collect(ast)
        self.kernel_args = cg.get_kernel_args(ast, fpl.par_dim)

    def define_arguments(self):
        """ Find all kernel arguments that can be defined
            at compilation time. Then defines them.
        """

        types = ci.get_types(self.ast)
        defines = list()
        for n in self.kernel_args:
            if len(types[n]) < 2:
                defines.append(n)
                self.ast.ext.append(lan.KernelArgDefine(lan.Id(n)))


def setdefine(ast):
    # TODO: Check that the vars in varlist are actually an argument

    define_compound = lan.GroupCompound([lan.Comment('// Defines for the kernel')])
    accname = 'str'
    sstream = lan.TypeId(['std::stringstream'], lan.Id(accname))
    stats = define_compound.statements
    stats.append(sstream)

    # add the defines to the string stream
    array_dim_swap = ca.get_array_dim_swap(ast)
    kernel_arg_defines = ci.get_kernel_arg_defines(ast)
    for var in sorted(kernel_arg_defines):
        try:
            hstvar = array_dim_swap[var]
        except KeyError:
            hstvar = var
        add = lan.Id(accname + ' << ' + '\"' + '-D' + var + '=\"' + ' << ' + hstvar + ' << \" \";')
        stats.append(add)

    # Set the string to the global variable
    lval = lan.Id('KernelDefines')
    stats.append(lan.Assignment(lval, lan.Id(accname + '.str()')))
    return define_compound
