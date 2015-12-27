import lan
import collect_id as ci
import collect_array as ca
import collect_array_rewrite as car


class Rewriter(object):
    def __init__(self):
        # Output
        self.Includes = list()

    def rewrite_array_ref(self, ast):
        naref = car.NormArrayRef(ast)
        naref.visit(ast)

    def rewrite_to_baseform(self, ast, functionname='FunctionName', change_ast=True):
        """ Rewrites a few things in the AST, ast, to increase the
            abstraction level.
            :param ast: abstract syntax tree
            :param functionname: nameo of kernel function
            :param change_ast: whether or not we should make the changes to the ast
        """

        typeid = lan.TypeId(['void'], lan.Id(functionname), ast.coord)

        lan_kernel_args = LanKernelArgs()
        lan_kernel_args.generate(ast)
        array_args = lan_kernel_args.array_args

        arglist = lan.ArgList([] + array_args)
        while isinstance(ast.ext[0], lan.Include):
            include = ast.ext.pop(0)
            self.Includes.append(include)

        while not isinstance(ast.ext[0], lan.ForLoop):
            ast.ext.pop(0)
        compound = lan.Compound(ast.ext)
        if change_ast:
            ast.ext = list()
            ast.ext.append(lan.FuncDecl(typeid, arglist, compound))


class LanKernelArgs(object):
    def __init__(self):
        self.array_args = list()

    def generate(self, ast):
        num_array_dim = ca.NumArrayDim(ast)
        num_array_dim.visit(ast)

        num_array_dims = num_array_dim.numSubscripts

        mytype_ids = ci.GlobalTypeIds()
        mytype_ids.visit(ast)
        # print print_dict_sorted(mytype_ids.dictIds)
        types = mytype_ids.types

        g_arrays_ids = ca.GlobalArrayIds()
        g_arrays_ids.visit(ast)
        array_ids = g_arrays_ids.ids
        # print arrays_ids.all_a_ids

        g_non_array_ids = ci.GlobalNonArrayIds()
        g_non_array_ids.visit(ast)
        non_array_ids = g_non_array_ids.ids
        array_args = list()
        for arrayid in array_ids:
            array_args.append(lan.TypeId(types[arrayid], lan.Id(arrayid, ast.coord), ast.coord))

            for iarg in xrange(num_array_dims[arrayid]):
                array_args.append(
                    lan.TypeId(['size_t'], lan.Id('hst_ptr' + arrayid + '_dim' + str(iarg + 1), ast.coord),
                               ast.coord))

        for arrayid in non_array_ids:
            array_args.append(lan.TypeId(types[arrayid], lan.Id(arrayid, ast.coord), ast.coord))

        self.array_args = array_args
