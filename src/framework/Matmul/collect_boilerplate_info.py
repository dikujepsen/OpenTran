import collect_transformation_info as cti


class GenReverseIdx(object):
    def __init__(self):
        self.ReverseIdx = dict()
        self.ReverseIdx[0] = 1
        self.ReverseIdx[1] = 0


class FindLoopArrays(cti.FindLoops):
    def __init__(self):
        super(FindLoopArrays, self).__init__()

    def collect(self, ast):
        super(FindLoopArrays, self).collect(ast)

    @property
    def loop_arrays(self):
        return self.arrays.LoopArrays
