import transf_visitor as tvisitor
import lan
import visitor
import copy

class Transpose(object):
    def __init__(self):
        self.SubscriptNoId = dict()
        self.IdxToDim = dict()
        self.ParDim = None  # int
        self.HstId = dict()
        self.ArrayIds = set()
        self.num_array_dims = dict()
        self.GlobalVars = dict()
        self.Type = dict()
        self.NameSwap = dict()
        self.ArrayIdToDimName = dict()
        self.Mem = dict()
        self.Transposition = None
        self.ReadWrite = dict()
        self.WriteOnly = list()
        self.WriteTranspose = list()
        self.Subscript = dict()


    def set_datastructures(self, ast):
        perfect_for_loop = tvisitor.PerfectForLoop()
        perfect_for_loop.visit(ast)

        if self.ParDim is None:
            self.ParDim = perfect_for_loop.depth

        loops = visitor.ForLoops()
        loops.visit(ast)
        for_loop_ast = loops.ast
        loop_indices = visitor.LoopIndices()
        loop_indices.visit(for_loop_ast)
        self.loop_index = loop_indices.index

        arrays = visitor.Arrays(self.loop_index)
        arrays.visit(ast)

        for n in arrays.numIndices:
            if arrays.numIndices[n] == 2:
                arrays.numSubscripts[n] = 2
            elif arrays.numIndices[n] > 2:
                arrays.numSubscripts[n] = 1

        self.num_array_dims = arrays.numSubscripts
        self.Subscript = arrays.Subscript

        self.SubscriptNoId = copy.deepcopy(self.Subscript)
        for n in self.SubscriptNoId.values():
            for m in n:
                for i, k in enumerate(m):
                    try:
                        m[i] = k.name
                    except AttributeError:
                        try:
                            m[i] = k.value
                        except AttributeError:
                            m[i] = 'unknown'

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

        for i, n in enumerate(reversed(self.GridIndices)):
            self.IdxToDim[i] = n


        type_ids = visitor.TypeIds()
        type_ids.visit(for_loop_ast)

        ids = visitor.Ids2()
        ids.visit(ast)

        # print ids.ids, "123"
        # print arrays.ids
        # print type_ids.ids
        other_ids = ids.ids - arrays.ids - type_ids.ids
        self.ArrayIds = arrays.ids - type_ids.ids
        self.NonArrayIds = other_ids

        for n in self.ArrayIds:
            self.HstId[n] = 'hst_ptr' + n
            self.Mem[n] = 'hst_ptr' + n + '_mem_size'

        type_ids2 = visitor.TypeIds()
        type_ids2.visit(ast)
        for n in type_ids.ids:
            type_ids2.dictIds.pop(n)
        self.Type = type_ids2.dictIds

        find_dim = tvisitor.FindDim(self.num_array_dims)
        find_dim.visit(ast)
        self.ArrayIdToDimName = find_dim.dimNames
        self.Transposition = lan.GroupCompound([lan.Comment('// Transposition')])

        find_read_write = tvisitor.FindReadWrite(self.ArrayIds)
        find_read_write.visit(ast)
        self.ReadWrite = find_read_write.ReadWrite

        for n in self.ReadWrite:
            pset = self.ReadWrite[n]
            if len(pset) == 1:
                if 'write' in pset:
                    self.WriteOnly.append(n)

    def transpose(self):
        """ Find all arrays that *should* be transposed
            and transpose them
        """
        # self.SubscriptNoId

        # self.IdxToDim idx to dim number
        notranspose_arrays = set()
        transpose_arrays = set()
        maybetranspose_arrays = set()
        # print self.SubscriptNoId, "qwe123"
        # print self.IdxToDim, "qwe123"
        for (n, sub) in self.SubscriptNoId.items():
            # Every ArrayRef, Every list of subscripts
            for s in sub:
                if len(s) == 2:
                    if s[0] == self.IdxToDim[0]:
                        transpose_arrays.add(n)
                    elif s[1] == self.IdxToDim[0]:
                        notranspose_arrays.add(n)
                    elif self.ParDim == 2 and s[0] == self.IdxToDim[1]:
                        maybetranspose_arrays.add(n)
        # print transpose_arrays, "qwe123"
        for n in transpose_arrays:
            self.__transpose(n)

    def __transpose(self, arr_name):

        # if rw.DefinesAreMade:
        #     print "Transposed must be called before SetDefine, returning..."
        #     return

        if self.num_array_dims[arr_name] != 2:
            print "Array ", arr_name, "of dimension ", \
                self.num_array_dims[arr_name], "cannot be transposed"
            return

        hstName = self.HstId[arr_name]
        hstTransName = hstName + '_trans'
        self.GlobalVars[hstTransName] = ''
        self.HstId[hstTransName] = hstTransName
        self.Type[hstTransName] = self.Type[arr_name]
        # Swap the hst ptr
        self.NameSwap[hstName] = hstTransName
        # Swap the dimension argument
        dimName = self.ArrayIdToDimName[arr_name]
        self.NameSwap[dimName[0]] = dimName[1]

        lval = lan.Id(hstTransName)
        natType = self.Type[arr_name][0]
        rval = lan.Id('new ' + natType + '[' \
                      + self.Mem[arr_name] + ']')
        self.Transposition.statements.append(lan.Assignment(lval, rval))
        if arr_name not in self.WriteOnly:
            arglist = lan.ArgList([lan.Id(hstName), \
                                   lan.Id(hstTransName), \
                                   lan.Id(dimName[0]), \
                                   lan.Id(dimName[1])])
            trans = lan.FuncDecl(lan.Id('transpose<' + natType + '>'), arglist, lan.Compound([]))
            self.Transposition.statements.append(trans)

        if arr_name in self.ReadWrite:
            if 'write' in self.ReadWrite[arr_name]:
                arglist = lan.ArgList([lan.Id(hstTransName), \
                                       lan.Id(hstName), \
                                       lan.Id(dimName[1]), \
                                       lan.Id(dimName[0])])
                trans = lan.FuncDecl(lan.Id('transpose<' + natType + '>'), arglist, lan.Compound([]))
                self.WriteTranspose.append(trans)


        for sub in self.Subscript[arr_name]:
            (sub[0], sub[1]) = \
                (sub[1], sub[0])
        # print self.Subscript[arr_name], "sub123"