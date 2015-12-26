import lan
import collect_transformation_info as cti


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

        self.ParDim = fpl.ParDim

        self.num_array_dims = fai.num_array_dims
        self.Subscript = fs.Subscript
        self.SubscriptNoId = fs.SubscriptNoId

        self.IdxToDim = fpl.IdxToDim

        self.ArrayIds = fai.ArrayIds

        self.HstId = fai.HstId
        self.Mem = fai.Mem
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
