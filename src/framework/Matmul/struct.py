class ChangedByTransformation(object):
    def __init__(self):
        # Kun sat af transformation
        self.PlaceInRegArgs = list()
        self.PlaceInRegCond = None
        self.PlaceInLocalArgs = list()
        self.PlaceInLocalCond = None
        self.Type = dict()
        self.KernelArgs = dict()

        # Stencil
        self.Kernel = None
        self.LocalSwap = dict()
        self.num_array_dims = dict()
        self.ArrayIdToDimName = dict()
        self.LoopArrays = dict()
        self.Add = dict()


class KernelStruct(ChangedByTransformation):
    def __init__(self):
        super(KernelStruct, self).__init__()
        # Heller ikke aendret af nogen transformation, ved ikke hvorfor de ikke staar nederst
        self.ArrayIds = set()
        self.Includes = list()
        self.SubSwap = dict()
        self.ParDim = None

        # Ikke aendret af nogen transformation
        self.Loops = dict()
        self.UpperLimit = dict()
        self.SubscriptNoId = dict()
        self.GridIndices = list()
        self.Local = dict()
        self.ReverseIdx = dict()

    def set_datastructure(self, tranf_rp):
        self.ArrayIds = tranf_rp.astrepr.ArrayIds
        self.Includes = tranf_rp.astrepr.Includes
        self.SubSwap = tranf_rp.SubSwap
        self.ParDim = tranf_rp.ParDim


        self.Loops = tranf_rp.Loops
        self.UpperLimit = tranf_rp.astrepr.UpperLimit

        self.SubscriptNoId = tranf_rp.SubscriptNoId
        self.GridIndices = tranf_rp.GridIndices
        self.Local = tranf_rp.Local
        self.ReverseIdx = tranf_rp.ReverseIdx

        # Stencil
        self.ArrayIdToDimName = tranf_rp.ArrayIdToDimName  #
        self.LocalSwap = tranf_rp.LocalSwap  #
        self.LoopArrays = tranf_rp.astrepr.LoopArrays  #
        self.Kernel = tranf_rp.Kernel  #
        self.num_array_dims = tranf_rp.astrepr.num_array_dims  #
        self.Add = tranf_rp.Add  #


class BoilerPlateStruct(object):
    def __init__(self):
        # # From Transpose
        # self.Type = dict()
        #
        #
        #
        # # From Define
        # self.kernel_args = dict()
        #
        # # From PlaceInReg
        #
        # # From PlaceInLocal
        # self.Local = dict()
        #
        # # From Stencil
        # self.ArrayIdToDimName = dict()
        # self.Kernel = None
        # self.LoopArrays = dict()

        # Eneste der skal med til boilerplate
        self.Transposition = None
        self.NameSwap = dict()
        self.HstId = dict()
        self.GlobalVars = dict()
        self.WriteTranspose = list()
        self.define_compound = None
