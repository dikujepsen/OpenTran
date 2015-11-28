

class Transpose(object):

    def __init__(self, rw):
        self.rw = rw

    def Transpose123(self):
        """ Find all arrays that *should* be transposed
            and transpose them
        """
        rw = self.rw
        # self.SubscriptNoId

        # self.IdxToDim idx to dim number
        notranspose = set()
        transpose = set()
        maybetranspose = set()
        for (n, sub) in rw.SubscriptNoId.items():
            # Every ArrayRef, Every list of subscripts
            if len(sub) == 2:
                if sub[0] == rw.IdxToDim[0]:
                    transpose.add(n)
                elif sub[1] == rw.IdxToDim[0]:
                    notranspose.add(n)
                elif rw.ParDim == 2 and sub[0] == rw.IdxToDim[1]:
                    maybetranspose.add(n)
        for n in transpose:
            self.tf.transpose(n)
