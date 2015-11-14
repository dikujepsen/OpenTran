



import framework.lan as lan





class Rewriter(object):


    def __init__(self, astrepr):
        self.astrepr = astrepr


    def rewrite(self, ast, functionname = 'FunctionName', changeAST = True):
        """ Rewrites a few things in the AST to increase the
    	abstraction level.
        """

        typeid = lan.TypeId(['void'], lan.Id(functionname),ast.coord)
        arraysArg = list()
        for arrayid in self.astrepr.ArrayIds:
            arraysArg.append(lan.TypeId(self.astrepr.Type[arrayid], lan.Id(arrayid,ast.coord),ast.coord))
            for iarg in xrange(self.astrepr.NumDims[arrayid]):
                arraysArg.append(lan.TypeId(['size_t'], lan.Id('hst_ptr'+arrayid+'_dim'+str(iarg+1),ast.coord),ast.coord))

        for arrayid in self.astrepr.NonArrayIds:
             arraysArg.append(lan.TypeId(self.astrepr.Type[arrayid], lan.Id(arrayid,ast.coord),ast.coord))

        arglist = lan.ArgList([] + arraysArg)
        while isinstance(ast.ext[0], lan.Include):
            include = ast.ext.pop(0)
            self.astrepr.Includes.append(include)

        while not isinstance(ast.ext[0], lan.ForLoop):
            ast.ext.pop(0)
        compound = lan.Compound(ast.ext)
        if changeAST:
            ast.ext = list()
            ast.ext.append(lan.FuncDecl(typeid, arglist, compound))