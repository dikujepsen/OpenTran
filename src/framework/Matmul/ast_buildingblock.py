
import lan

def EmptyFuncDecl(name, type = ['void']):
    """ Returns a FuncDecl with no arguments or body """
    allocateBufferTypeId = lan.TypeId(type, lan.Id(name))
    allocateBufferArgList = lan.ArgList([])
    allocateBufferCompound = lan.Compound([])
    allocateBuffer = lan.FuncDecl(allocateBufferTypeId,\
                              allocateBufferArgList,\
                              allocateBufferCompound)


    return allocateBuffer

def FuncCall(name, arglist):
    """ Returns a FuncDecl with no arguments or body """
    allocateBufferTypeId = lan.Id(name)
    allocateBufferArgList = lan.ArgList(arglist)
    allocateBufferCompound = lan.Compound([])
    allocateBuffer = lan.FuncDecl(allocateBufferTypeId,\
                              allocateBufferArgList,\
                              allocateBufferCompound)


    return allocateBuffer


def ConstantAssignment(name, constant = 0, type = ['unsigned']):
    lval = lan.TypeId(type, lan.Id(name))
    rval = lan.Constant(constant)
    return lan.Assignment(lval, rval)
