class KernelStruct(object):
    def __init__(self):
        self.ArrayIdToDimName = dict()
        self.Type = dict()
        self.ArrayIds = set()
        self.KernelArgs = dict()
        self.LocalSwap = dict()
        self.LoopArrays = dict()
        self.Kernel = None
        self.Includes = list()
        self.num_array_dims = dict()
        self.SubSwap = dict()
        self.ParDim = None

        # Ikke aendret af nogen transformation
        self.Loops = dict()
        self.UpperLimit = dict()

        self.SubscriptNoId = dict()
        self.GridIndices = list()
        self.Local = dict()
        self.Add = dict()
        self.ReverseIdx = dict()

        # Kun sat af transformation
        self.PlaceInRegArgs = list()
        self.PlaceInRegCond = None
        self.PlaceInLocalArgs = list()
        self.PlaceInLocalCond = None

    def set_datastructure(self, tranf_rp):
        self.ArrayIds = tranf_rp.astrepr.ArrayIds
        self.Includes = tranf_rp.astrepr.Includes
        self.SubSwap = tranf_rp.SubSwap
        self.ParDim = tranf_rp.ParDim

        self.ArrayIdToDimName = tranf_rp.ArrayIdToDimName  #
        self.Type = tranf_rp.astrepr.Type  #
        self.KernelArgs = tranf_rp.KernelArgs  #
        self.LocalSwap = tranf_rp.LocalSwap  #
        self.LoopArrays = tranf_rp.astrepr.LoopArrays  #
        self.Kernel = tranf_rp.Kernel  #
        self.num_array_dims = tranf_rp.astrepr.num_array_dims  #

        self.Loops = tranf_rp.Loops
        self.UpperLimit = tranf_rp.astrepr.UpperLimit

        self.SubscriptNoId = tranf_rp.SubscriptNoId
        self.GridIndices = tranf_rp.GridIndices
        self.Local = tranf_rp.Local
        self.Add = tranf_rp.Add
        self.ReverseIdx = tranf_rp.ReverseIdx


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
