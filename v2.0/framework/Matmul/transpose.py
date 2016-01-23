import lan
import collect_gen as cg
import collect_array as ca
import collect_id as ci


class Transpose(object):
    def __init__(self, ast):
        self.ast = ast

    def transpose(self):
        """ Find all arrays that *should* be transposed
            and transpose them
            :param ast:
        """
        transpose_arrays = self.find_transposable_arrays()
        hst_id = cg.get_host_ids(self.ast)
        types = ci.get_types(self.ast)
        for n in transpose_arrays:
            hst_name = hst_id[n]
            hst_trans_name = hst_name + '_trans'
            self.ast.ext.append(lan.Transpose(types[n], lan.Id(hst_trans_name), lan.Id(n), lan.Id(hst_name)))
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
        idx_to_dim = cg.gen_idx_to_dim(self.ast)
        return len(sub) == 2 and sub[0] == idx_to_dim[0]

    def __transpose(self, arr_name):

        num_array_dim = ca.get_num_array_dims(self.ast)
        if num_array_dim[arr_name] != 2:
            print "Array ", arr_name, "of dimension ", \
                num_array_dim[arr_name], "cannot be transposed"
            return

        subscript = ca.get_subscript(self.ast)
        for sub in subscript[arr_name]:
            (sub[0], sub[1]) = \
                (sub[1], sub[0])

    def create_transposition_func(self, arr_name):
        my_transposition = []
        hst_id = cg.get_host_ids(self.ast)
        array_id_to_dim_name = cg.get_array_id_to_dim_name(self.ast)
        types = ci.get_types(self.ast)
        mem_names = cg.get_mem_names(self.ast)
        write_only = ca.get_write_only(self.ast)

        hst_name = hst_id[arr_name]
        hst_trans_name = hst_name + '_trans'
        dim_name = array_id_to_dim_name[arr_name]

        lval = lan.Id(hst_trans_name)
        nat_type = types[arr_name][0]
        rval = lan.Id('new ' + nat_type + '[' + mem_names[arr_name] + ']')
        my_transposition.append(lan.Assignment(lval, rval))
        if arr_name not in write_only:
            arglist = lan.ArgList([lan.Id(hst_name),
                                   lan.Id(hst_trans_name),
                                   lan.Id(dim_name[0]),
                                   lan.Id(dim_name[1])])
            trans = lan.FuncDecl(lan.Id('transpose<' + nat_type + '>'), arglist, lan.Compound([]))
            my_transposition.append(trans)

        return my_transposition
