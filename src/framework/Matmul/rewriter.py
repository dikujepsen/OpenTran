
import lan


class Rewriter(object):

    def __init__(self, astrepr):
        self.astrepr = astrepr

    def rewrite_to_baseform(self, ast, functionname ='FunctionName', changeAST = True):
        """ Rewrites a few things in the AST, ast, to increase the
            abstraction level.
            :param ast: abstract syntax tree
            :param functionname: nameo of kernel function
            :param changeAST: whether or not we should make the changes to the ast
        """

        typeid = lan.TypeId(['void'], lan.Id(functionname), ast.coord)
        array_args = list()
        for arrayid in self.astrepr.ArrayIds:
            array_args.append(lan.TypeId(self.astrepr.Type[arrayid], lan.Id(arrayid, ast.coord), ast.coord))
            for iarg in xrange(self.astrepr.NumDims[arrayid]):
                array_args.append(lan.TypeId(['size_t'], lan.Id('hst_ptr'+arrayid+'_dim'+str(iarg+1), ast.coord),
                                             ast.coord))

        for arrayid in self.astrepr.NonArrayIds:
            array_args.append(lan.TypeId(self.astrepr.Type[arrayid], lan.Id(arrayid, ast.coord), ast.coord))

        arglist = lan.ArgList([] + array_args)
        while isinstance(ast.ext[0], lan.Include):
            include = ast.ext.pop(0)
            self.astrepr.Includes.append(include)

        while not isinstance(ast.ext[0], lan.ForLoop):
            ast.ext.pop(0)
        compound = lan.Compound(ast.ext)
        if changeAST:
            ast.ext = list()
            ast.ext.append(lan.FuncDecl(typeid, arglist, compound))
