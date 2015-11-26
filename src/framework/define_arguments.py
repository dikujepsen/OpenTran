import lan


class DefineArguments(object):
    def __init__(self):
        self.kernel_args = dict()
        self.type = dict()
        self.define_compound = None

    def define_arguments(self):
        """ Find all kernel arguments that can be defined
            at compilation time. Then defines them.
        """

        defines = list()
        for n in self.KernelArgs:
            if len(self.Type[n]) < 2:
                defines.append(n)

        self.SetDefine(defines)

    def SetDefine(self, var_list):
        # TODO: Check that the vars in varlist are actually an argument

        accname = 'str'
        sstream = lan.TypeId(['std::stringstream'], lan.Id(accname))
        stats = self.define_compound.statements
        stats.append(sstream)

        # add the defines to the string stream
        for var in var_list:
            try:
                hstvar = rw.NameSwap[var]
            except KeyError:
                hstvar = var
            add = lan.Id(accname + ' << ' + '\"' + '-D' + var + '=\"' + ' << ' + hstvar + ' << \" \";')
            stats.append(add)

        # Set the string to the global variable
        lval = lan.Id('KernelDefines')
        stats.append(lan.Assignment(lval, lan.Id(accname + '.str()')))

        # Need to remove the kernel corresponding kernel arguments
        for var in var_list:
            rw.KernelArgs.pop(var)
