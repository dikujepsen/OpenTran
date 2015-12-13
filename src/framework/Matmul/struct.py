


class BoilerPlateStruct(object):
    def __init__(self):
        # From Transpose
        self.Type = dict()
        self.NameSwap = dict()
        self.WriteTranspose = list()
        self.Transposition = None
        self.HstId = dict()
        self.GlobalVars = dict()

        # From Define
        self.kernel_args = dict()
        self.define_compound = None

        # From PlaceInReg

        # From PlaceInLocal
        self.Local = dict()

        # From Stencil
        self.ArrayIdToDimName = dict()
        self.Kernel = None
        self.LoopArrays = dict()
