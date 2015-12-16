import collect_transformation_info as cti
import collect_boilerplate_info as cbi

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

        # bruges kun til at interchange subscript i transpose
        self.Subscript = dict()

    def set_datastructure(self, rw, ast):
        fpl = cti.FindGridIndices()
        fpl.ParDim = self.ParDim
        fpl.collect(ast)
        self.ParDim = fpl.par_dim

        fai = cti.FindArrayIds()
        fai.ParDim = self.ParDim
        fai.collect(ast)

        fs = cti.FindSubscripts()
        fs.collect(ast)

        fl = cti.FindLocal()
        fl.ParDim = self.ParDim
        fl.collect(ast)

        gr = cbi.GenReverseIdx()

        fla = cbi.FindLoopArrays()
        fla.ParDim = self.ParDim
        fla.collect(ast)

        self.ArrayIds = fai.ArrayIds
        self.Includes = rw.Includes
        self.ParDim = fpl.par_dim

        self.Loops = fai.Loops
        self.UpperLimit = fai.upper_limit

        self.Subscript = fs.Subscript
        self.SubscriptNoId = fs.SubscriptNoId
        self.GridIndices = fpl.GridIndices
        self.Local = fl.Local
        self.ReverseIdx = gr.ReverseIdx

        # Stencil
        self.ArrayIdToDimName = fai.ArrayIdToDimName  #
        self.LoopArrays = fla.loop_arrays  #
        self.Kernel = fpl.Kernel  #
        self.num_array_dims = fai.num_array_dims  #


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
        self.NoReadBack = None

    def SetNoReadBack(self):
        self.NoReadBack = True

