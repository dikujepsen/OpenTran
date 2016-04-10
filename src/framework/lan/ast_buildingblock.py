import lan


def EmptyFuncDecl(name, type=['void']):
    """ Returns a FuncDecl with no arguments or body """
    allocateBufferTypeId = lan.TypeId(type, lan.Id(name))
    allocateBufferArgList = lan.ArgList([])
    allocateBufferCompound = lan.Compound([])
    allocateBuffer = lan.FuncDecl(allocateBufferTypeId,
                                  allocateBufferArgList,
                                  allocateBufferCompound)

    return allocateBuffer


def FuncCall(name, arglist=[]):
    """ Returns a FuncDecl with no arguments or body """
    funcId = lan.Id(name)
    funcArgList = lan.ArgList(arglist)
    func_call = lan.FuncCall(funcId,
                             funcArgList)

    return func_call


def ClassMemberFuncCall(classname, name, arglist=[]):
    classname_id = lan.Id(classname)
    func_id = lan.Id(name)
    func_arg_list = lan.ArgList(arglist)
    func_call = lan.ClassMemberFuncCall(classname_id, func_id,
                                        func_arg_list)

    return func_call


def ConstantAssignment(name, constant=0, type=['unsigned']):
    lval = lan.TypeId(type, lan.Id(name))
    rval = lan.Constant(constant)
    return lan.Assignment(lval, rval)
