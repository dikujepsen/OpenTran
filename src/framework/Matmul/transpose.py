import lan
import collect_transformation_info as cti
import collect_gen as cg


class Transpose(object):
    def __init__(self):
        self.SubscriptNoId = dict()
        self.IdxToDim = dict()
        self.ParDim = None  # int
        self.ArrayIds = set()
        self.num_array_dims = dict()
        self.ArrayIdToDimName = dict()
        self.Mem = dict()
        self.Subscript = dict()
        self.ReadWrite = dict()
        self.WriteOnly = list()

        self.Type = dict()
        self.NameSwap = dict()
        self.WriteTranspose = list()
        self.Transposition = None
        self.HstId = dict()
        self.GlobalVars = dict()

    def set_datastructures(self, ast):

        self.Transposition = lan.GroupCompound([lan.Comment('// Transposition')])

        fpl = cti.FindGridIndices()
        fpl.ParDim = self.ParDim
        fpl.collect(ast)

        fs = cti.FindSubscripts()
        fs.collect(ast)

        fai = cti.FindReadWrite()
        fai.ParDim = self.ParDim
        fai.collect(ast)

        host_array_data = cg.GenHostArrayData()
        host_array_data.collect(ast)

        self.ParDim = fpl.ParDim

        self.num_array_dims = fai.num_array_dims
        self.Subscript = fs.Subscript
        self.SubscriptNoId = fs.SubscriptNoId

        self.IdxToDim = fpl.IdxToDim

        self.ArrayIds = fai.ArrayIds

        self.HstId = host_array_data.HstId
        self.Mem = host_array_data.Mem
        self.Type = fai.type

        self.ArrayIdToDimName = fai.ArrayIdToDimName

        self.ReadWrite = fai.ReadWrite

        self.WriteOnly = fai.WriteOnly

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

        hst_name = self.HstId[arr_name]
        hst_trans_name = hst_name + '_trans'
        self.GlobalVars[hst_trans_name] = ''
        self.HstId[hst_trans_name] = hst_trans_name
        self.Type[hst_trans_name] = self.Type[arr_name]
        # Swap the hst ptr
        self.NameSwap[hst_name] = hst_trans_name
        # Swap the dimension argument
        dim_name = self.ArrayIdToDimName[arr_name]
        self.NameSwap[dim_name[0]] = dim_name[1]

        lval = lan.Id(hst_trans_name)
        nat_type = self.Type[arr_name][0]
        rval = lan.Id('new ' + nat_type + '[' + self.Mem[arr_name] + ']')
        self.Transposition.statements.append(lan.Assignment(lval, rval))
        if arr_name not in self.WriteOnly:
            arglist = lan.ArgList([lan.Id(hst_name),
                                   lan.Id(hst_trans_name),
                                   lan.Id(dim_name[0]),
                                   lan.Id(dim_name[1])])
            trans = lan.FuncDecl(lan.Id('transpose<' + nat_type + '>'), arglist, lan.Compound([]))
            self.Transposition.statements.append(trans)

        if arr_name in self.ReadWrite:
            if 'write' in self.ReadWrite[arr_name]:
                arglist = lan.ArgList([lan.Id(hst_trans_name),
                                       lan.Id(hst_name),
                                       lan.Id(dim_name[1]),
                                       lan.Id(dim_name[0])])
                trans = lan.FuncDecl(lan.Id('transpose<' + nat_type + '>'), arglist, lan.Compound([]))
                self.WriteTranspose.append(trans)

        for sub in self.Subscript[arr_name]:
            if isinstance(sub[0], lan.Id) and isinstance(sub[1], lan.Id):
                (sub[0].name, sub[1].name) = \
                    (sub[1].name, sub[0].name)
            # print self.Subscript[arr_name], "sub123"
