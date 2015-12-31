import lan
import collect_transformation_info as cti
import collect_gen as cg
import collect_array as ca


class Transpose(object):
    def __init__(self):
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
        self.HstId = dict()
        self.GlobalVars = dict()
        self.ast = None

    def set_datastructures(self, ast, par_dim):
        if par_dim is not None:
            self.ParDim = par_dim

        self.ast = ast

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

        self.ParDim = fpl.par_dim

        self.num_array_dims = fai.num_array_dims
        self.Subscript = fs.Subscript

        self.ArrayIds = fai.ArrayIds

        self.HstId = host_array_data.HstId
        self.Mem = host_array_data.Mem
        self.Type = fai.type

        self.ArrayIdToDimName = fai.ArrayIdToDimName

        self.ReadWrite = fai.ReadWrite

        self.WriteOnly = fai.WriteOnly

    def transpose(self, ast):
        """ Find all arrays that *should* be transposed
            and transpose them
            :param ast:
        """
        self.ast = ast
        transpose_arrays = self.find_transposable_arrays()
        # print transpose_arrays
        for n in transpose_arrays:
            hst_name = self.HstId[n]
            hst_trans_name = hst_name + '_trans'
            self.ast.ext.append(lan.Transpose(self.Type[n], lan.Id(hst_trans_name), lan.Id(n)))
            self.__transpose(n)

    def find_transposable_arrays(self):
        transpose_arrays = set()

        subscript_no_id = ca.get_subscript_no_id(self.ast)
        for (n, sub) in subscript_no_id.items():
            # Every ArrayRef, Every list of subscripts
            for s in sub:
                if self._subscript_should_be_swapped(s):
                    transpose_arrays.add(n)
        return transpose_arrays

    def _subscript_should_be_swapped(self, sub):
        """
        The array ref has two subscripts and the othermost subscript contains the inner most grid index,
        i.e. the thread id that changes most often. This situation leads to uncoalesced memory access.
        :param sub:
        :return:
        """
        idx_to_dim = cg.gen_idx_to_dim(self.ast, self.ParDim)
        return len(sub) == 2 and sub[0] == idx_to_dim[0]

    def __transpose(self, arr_name):

        if self.num_array_dims[arr_name] != 2:
            print "Array ", arr_name, "of dimension ", \
                self.num_array_dims[arr_name], "cannot be transposed"
            return

        hst_name = self.HstId[arr_name]
        hst_trans_name = hst_name + '_trans'

        # Swap the hst ptr
        self.NameSwap[hst_name] = hst_trans_name
        # Swap the dimension argument
        dim_name = self.ArrayIdToDimName[arr_name]
        self.NameSwap[dim_name[0]] = dim_name[1]

        for sub in self.Subscript[arr_name]:
            (sub[0], sub[1]) = \
                (sub[1], sub[0])

    def create_transposition_func(self, arr_name):
        my_transposition = []
        hst_name = self.HstId[arr_name]
        hst_trans_name = hst_name + '_trans'
        dim_name = self.ArrayIdToDimName[arr_name]

        lval = lan.Id(hst_trans_name)
        nat_type = self.Type[arr_name][0]
        rval = lan.Id('new ' + nat_type + '[' + self.Mem[arr_name] + ']')
        my_transposition.append(lan.Assignment(lval, rval))
        if arr_name not in self.WriteOnly:
            arglist = lan.ArgList([lan.Id(hst_name),
                                   lan.Id(hst_trans_name),
                                   lan.Id(dim_name[0]),
                                   lan.Id(dim_name[1])])
            trans = lan.FuncDecl(lan.Id('transpose<' + nat_type + '>'), arglist, lan.Compound([]))
            my_transposition.append(trans)

        return my_transposition
