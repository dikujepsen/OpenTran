import lan
import transf_visitor as tvisitor
import visitor


class DefineArguments(object):
    def __init__(self):
        self.kernel_args = dict()
        self.type = dict()
        self.define_compound = None
        self.name_swap = dict()
        self.ParDim = None  # int


    def set_datastructures(self, ast):

        self.define_compound = lan.GroupCompound([lan.Comment('// Defines for the kernel')])

        perfect_for_loop = tvisitor.PerfectForLoop()
        perfect_for_loop.visit(ast)

        self.ParDim = perfect_for_loop.depth

        grid_ids = list()
        init_ids = tvisitor.InitIds()
        init_ids.visit(perfect_for_loop.ast.init)
        grid_ids.extend(init_ids.index)
        kernel = perfect_for_loop.ast.compound
        if self.ParDim == 2:
            init_ids = tvisitor.InitIds()
            init_ids.visit(kernel.statements[0].init)
            kernel = kernel.statements[0].compound
            grid_ids.extend(init_ids.index)

        self.GridIndices = grid_ids
        self.Kernel = kernel

        loops = visitor.ForLoops()
        loops.visit(ast)
        for_loop_ast = loops.ast
        loop_indices = visitor.LoopIndices()
        loop_indices.visit(for_loop_ast)
        self.loop_index = loop_indices.index
        self.UpperLimit = loop_indices.end

        self.RemovedIds = set(self.UpperLimit[i] for i in self.GridIndices)

        ids_still_in_kernel = tvisitor.Ids()
        ids_still_in_kernel.visit(self.Kernel)
        self.RemovedIds = self.RemovedIds - ids_still_in_kernel.ids

        type_ids = visitor.TypeIds()
        type_ids.visit(for_loop_ast)
        arrays = visitor.Arrays(self.loop_index)
        arrays.visit(ast)
        for n in arrays.numIndices:
            if arrays.numIndices[n] == 2:
                arrays.numSubscripts[n] = 2
            elif arrays.numIndices[n] > 2:
                arrays.numSubscripts[n] = 1

        self.num_array_dims = arrays.numSubscripts

        ids = visitor.Ids2()
        ids.visit(ast)

        # print ids.ids, "123"
        # print arrays.ids
        # print type_ids.ids
        other_ids = ids.ids - arrays.ids - type_ids.ids
        self.ArrayIds = arrays.ids - type_ids.ids
        self.NonArrayIds = other_ids

        type_ids2 = visitor.TypeIds()
        type_ids2.visit(ast)
        for n in type_ids.ids:
            type_ids2.dictIds.pop(n)

        self.type = type_ids2.dictIds

        find_dim = tvisitor.FindDim(self.num_array_dims)
        find_dim.visit(ast)
        self.ArrayIdToDimName = find_dim.dimNames


        arg_ids = self.NonArrayIds.union(self.ArrayIds) - self.RemovedIds

        # print arg_ids

        for n in arg_ids:
            tmplist = [n]
            try:
                if self.num_array_dims[n] == 2:
                    tmplist.append(self.ArrayIdToDimName[n][0])
            except KeyError:
                pass
            for m in tmplist:
                self.kernel_args[m] = self.type[m]

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
