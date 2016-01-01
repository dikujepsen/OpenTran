import collect_transformation_info as cti
import collect_boilerplate_info as cbi
import collect_gen as cg
import collect_id as ci


class KernelChangedByTransformation(object):
    def __init__(self):
        self.Loops = dict()  # place_in_local

class KernelStruct(KernelChangedByTransformation):
    def __init__(self):
        super(KernelStruct, self).__init__()
        # Heller ikke aendret af nogen transformation, ved ikke hvorfor de ikke staar nederst
        self.Includes = list()
        self.ParDim = None

        # bruges kun til at interchange subscript i transpose
        self.Subscript = dict()
        self.LoopArrays = dict()
        self.LoopArraysParent = dict()

    def set_datastructure(self, rw, ast):
        fpl = cti.FindGridIndices()
        fpl.ParDim = self.ParDim
        fpl.collect(ast)
        self.ParDim = fpl.par_dim

        fai = cti.FindArrayIdsKernel()
        fai.ParDim = self.ParDim
        fai.collect(ast)

        fs = cti.FindSubscripts()
        fs.collect(ast)

        fl = cti.FindLocal()
        fl.ParDim = self.ParDim
        fl.collect(ast)

        fla = cbi.FindLoopArrays()
        fla.ParDim = self.ParDim
        fla.collect(ast)

        self.Includes = rw.Includes
        self.ParDim = fpl.par_dim

        self.Loops = fai.Loops
        self.Subscript = fs.Subscript

        # Stencil
        self.LoopArrays = fla.loop_arrays  #
        self.LoopArraysParent = fla.loop_arrays_parent


class BoilerPlateChangedByTransformation(object):
    def __init__(self):
        # Eneste der skal med til boilerplate
        # Ting som er aendret af en transformation
        self.NoReadBack = None

    def set_no_read_back(self):
        self.NoReadBack = True


class BoilerPlateStruct(object):
    def __init__(self):
        self.NonArrayIds = set()
        self.KernelName = None
        self.DevId = dict()
        self.ConstantMem = set()
        self.DevArgList = list()
        self.Mem = dict()
        self.DevFuncId = None
        self.DevFuncTypeId = None
        self.RemovedIds = list()
        self.LowerLimit = list()

        self.ConstantMemory = None
        self.WriteOnly = list()
        self.ReadOnly = list()
        self.Worksize = dict()

    def set_datastructure(self, ast, par_dim=None):
        fai = cti.FindReadWrite()
        fai.ParDim = par_dim
        fai.collect(ast)
        # print fai.NonArrayIds
        self.NonArrayIds = fai.NonArrayIds
        self.LowerLimit = fai.lower_limit
        self.WriteOnly = fai.WriteOnly
        self.ReadOnly = fai.ReadOnly

        fkn = cbi.FindKernelName()
        fkn.ParDim = fai.par_dim
        fkn.collect(ast)
        self.KernelName = fkn.KernelName
        self.DevId = fkn.DevId
        self.DevFuncId = fkn.DevFuncId
        self.DevFuncTypeId = fkn.DevFuncTypeId
        self.DevArgList = fkn.DevArgList
        self.Mem = fkn.Mem
        self.RemovedIds = fkn.RemovedIds
        self.Worksize = fkn.Worksize
