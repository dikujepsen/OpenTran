import copy
import os
from visitor import *
from stringstream import *


class Transformation():
    """ Apply transformations to the original AST. Includes:
    1. Local Memory
    2. Stencil Local Memory
    3. Placing data in registers
    4. Transposing arrays
    5. Unrolling loops
    6. Adding defines
    7. Setting the number of dimensions to parallelize
    8. Setting the local work-group size
    9. Setting if we should read data back from the GPU
    """
    def __init__(self, rw):
        self.rw = rw

    def SetParDim(self, number):
        rw = self.rw
        rw.ParDim = number
        

    def SetLSIZE(self, lsizelist):
        rw = self.rw
        if len(lsizelist) == 1 and rw.ParDim == 1 \
               or len(lsizelist) == 2 and rw.ParDim == 2:
            rw.Local['size'] = lsizelist
        else:
            print """SetLSIZE: the local work-group size must be 1D or 2D and it must have the same number of dimensions as we are parallelizing """

    def SetNoReadBack(self):
        rw = self.rw
        rw.NoReadBack = True

        
    def Unroll(self, looplist):
        rw = self.rw
        # find loops and check that the loops given in the argument
        # exist
        loopIndices = LoopIndices()
        loopIndices.visit(rw.Kernel)
        kernelLoops = loopIndices.Loops
        for l in looplist:
            if l not in kernelLoops:
                print "Valid loops are %r. Your input contained %r. Aborting..." % (kernelLoops.keys(), l)
                return
        
        rw.UnrollLoops.extend(looplist)
        
    def Unroll2(self, looplist):
        rw = self.rw
        
        for n in looplist:
            outerloop = rw.Loops[n]
            outeridx = n
            compound = outerloop.compound
            outerloop.compound = Compound([])
            innerloop = copy.deepcopy(outerloop)
            innerloop.compound = compound
            outerstats = outerloop.compound.statements
            # Add the TypeId declarations to the outer loop
            # First find the typeids
            typeIds = TypeIds2()
            typeIds.visit(innerloop.compound)
            for m in typeIds.ids:
                outerstats.append(m)
            outerstats.append(innerloop)
            upperbound = str(looplist[n])
            if upperbound == '0':
                upperbound = outerloop.cond.rval.name
            # change increment of outer loop
            outerloop.inc = Increment(Id(outeridx), '+='+upperbound)
            inneridx = outeridx*2
            # For adding to this index in other subscripts
            rw.Add[outeridx] = inneridx

            # new inner loop
            innerloop.cond = BinOp(Id(inneridx), '<' , Id(upperbound))
            innerloop.inc = Increment(Id(inneridx), '++')
            innerloop.init = ConstantAssignment(inneridx)
            rw.Loops[inneridx] = innerloop


            
            # In the inner loop: Add the extra index to the index of the
            # outer loop
            exchangeId = ExchangeIdWithBinOp({n : BinOp(Id(outeridx),'+', Id(inneridx))})
            exchangeId.visit(innerloop.compound)

            # unroll the inner index in stringstream
            self.Unroll([inneridx])
            if looplist[n] == 0:
                self.Unroll([outeridx])

    def SetDefine(self, varList):
        # TODO: Check that the vars in varlist are actually an argument
        rw = self.rw

        rw.DefinesAreMade = True
        accname = 'str'
        sstream = TypeId(['std::stringstream'], Id(accname))
        stats = rw.Define.statements
        stats.append(sstream)

        # add the defines to the string stream 
        for var in varList:
            try:
                hstvar = rw.NameSwap[var]
            except KeyError:
                hstvar = var
            add = Id(accname + ' << ' + '\"' + '-D' + var+ '=\"' + \
                     ' << ' + hstvar + ' << \" \";')
            stats.append(add)

        # Set the string to the global variable
        lval = Id('KernelDefines')
        stats.append(Assignment(lval, Id(accname + '.str()')))

        # Need to remove the kernel corresponding kernel arguments
        for var in varList:
            rw.KernelArgs.pop(var)

            
    def placeInReg3(self, arrDict):
        """ Check if the arrayref is inside a loop and use a static
            array for the allocation of the registers
        """
        rw = self.rw
        stats = rw.Kernel.statements
        initstats = []
        loadings = []
        writes = []

        insideloop = ''
        # check what loop we are inside
        for n in arrDict:
            # assume all refs are inside the same loop
            idx = arrDict[n]
            sublist = rw.SubscriptNoId[n]
            for sub in sublist[idx[0]]:
                if sub not in rw.GridIndices:
                    insideloop = sub
                    break
            break

        if insideloop == '':
            print "placeInReg3 only works when the ArrayRef is inside a loop"
            return

        # Add allocation of registers to the initiation stage
        for n in arrDict:
            lval = TypeId([rw.Type[n][0]], \
                          Id(n+'_reg['+ str(rw.UpperLimit[insideloop])\
                             + ']'))
            initstats.append(lval)


        # add the loop to the initiation stage
        loop = copy.deepcopy(rw.Loops[insideloop])
        loopstats = []
        # Exchange loop index
        loop.compound.statements = loopstats       
        ## exchangeId = ExchangeId({insideloop : 'outer_'+insideloop})
        ## exchangeId.visit(loop)
        initstats.append(loop)
       
        # Create the loadings
        for i,n in enumerate(arrDict):
            for m in arrDict[n]:
                idx = m
                sub = copy.deepcopy(rw.LoopArrays[n][idx])
                regid = ArrayRef(Id(n + '_reg'), [Id(insideloop)])
                writes.append(regid)
                assign = Assignment(regid, sub)
                loopstats.append(assign)
        
        stats.insert(0, GroupCompound(initstats))
        # Replace the global Arefs with the register Arefs
        count = 0
        for i,n in enumerate(arrDict):
            for m in arrDict[n]:
                idx = m
                aref_new = copy.deepcopy(writes[count])
                aref_old = rw.LoopArrays[n][idx]
                # Copying the internal data of the two arefs
                aref_old.name.name = aref_new.name.name
                aref_old.subscript = aref_new.subscript
                count += 1


    def localMemory3(self, arrDict, loopDict = None, blockDict = None):
        rw = self.rw
        initstats = []
        initComp = GroupCompound(initstats)
        rw.Kernel.statements.insert(0, initComp)
        
        if loopDict is None:
            loopDict = dict()
            # So we create it
            for n in arrDict:
                for i in arrDict[n]:
                    loopDict[(n,i)] = []
                
            for n in arrDict:
                for i in arrDict[n]:
                    subscript = rw.SubscriptNoId[n][i]
                    acc = []
                    for m in subscript:
                        try:
                            something = int(m)
                        except:
                            if m not in rw.GridIndices:
                                acc.append(m)
                    loopDict[(n,i)] = acc

        # Check that all ArrayRefs are blocked using only one loop
        # otherwise we do not know what to do
        for n in arrDict:
            for i in arrDict[n]:
                if len(loopDict[(n,i)]) > 1:
                    print "Array %r is being blocked by %r. Returning..." \
                          % (n, loopDict[(n,i)])
                    return

        # Find which loops must be extended
        loopext = set()
        for n in arrDict:
            for i in arrDict[n]:
                loopext.add(loopDict[(n,i)][0])

        # do the extending
        for n in loopext:
            outerloop = rw.Loops[n]
            outeridx = n
            compound = outerloop.compound
            outerloop.compound = Compound([])
            innerloop = copy.deepcopy(outerloop)
            innerloop.compound = compound
            outerstats = outerloop.compound.statements
            outerstats.insert(0,innerloop)
            loadstats = []
            loadComp = GroupCompound(loadstats)
            outerstats.insert(0, loadComp)
            # change increment of outer loop
            outerloop.inc = Increment(Id(outeridx), '+='+rw.Local['size'][0])
            inneridx = outeridx*2
            # For adding to this index in other subscripts
            rw.Add[outeridx] = inneridx

            # new inner loop
            innerloop.cond = BinOp(Id(inneridx), '<' , Constant(rw.Local['size'][0]))
            innerloop.inc = Increment(Id(inneridx), '++')
            innerloop.init = ConstantAssignment(inneridx)
            rw.Loops[inneridx] = innerloop
      
        for n in arrDict:
            # Add array allocations
            ## dim = rw.NumDims[n]
            localName = n + '_local'
            arrayinit = '['
            arrayinit += rw.Local['size'][0]
            if rw.NumDims[n] == 2 and rw.ParDim == 2:
                arrayinit += '*' + rw.Local['size'][1]
            arrayinit += ']'
            
            localId = Id(localName + arrayinit)
            localTypeId = TypeId(['__local'] + [rw.Type[n][0]], localId)
            initstats.append(localTypeId)

        loadings = []
        for n in arrDict:
            loc_name = n+'_local'
            for i in arrDict[n]:
                glob_subs = copy.deepcopy(rw.LoopArrays[n][i])
                # Change loop idx to local idx
                loopname = loopDict[(n,i)][0]
                loc_subs = copy.deepcopy(glob_subs).subscript
                for k, m in enumerate(loc_subs):
                        if isinstance(m, Id) and \
                               m.name not in rw.GridIndices:
                            tid = str(rw.ReverseIdx[k])
                            tidstr = 'get_local_id('+tid+')'
                            exchangeId = ExchangeId({loopname : tidstr})
                            exchangeId.visit(m)
                            exchangeId2 = ExchangeId({loopname : loopname + ' + ' + tidstr})
                            exchangeId2.visit(glob_subs.subscript[k])
                loc_ref = ArrayRef(Id(loc_name), loc_subs)
                ## exchangeId = ExchangeId({loopname : loopname + ' + get_local_id(0)'})
                ## exchangeId.visit(glob_subs)

                ## if rw.NumDim[n] == 1:
                ##     exchangeId = ExchangeId({n : 'get_local_id(0)'})
                ## exchangeId.visit(loc_ref)
                loadings.append(Assignment(loc_ref, glob_subs))
                if rw.ParDim == 2:
                    exchangeId = ExchangeId({rw.GridIndices[1] : 'get_local_id(0)', rw.GridIndices[0] : 'get_local_id(1)'})
                else: 
                    exchangeId = ExchangeId({rw.GridIndices[0] : 'get_local_id(0)'})
                exchangeId.visit(loc_ref)

                inner_loc = rw.LoopArrays[n][i]
                inner_loc.name.name = loc_name
                exchangeId2 = ExchangeId({loopname : loopname*2})
                exchangeId2.visit(inner_loc)
                exchangeId.visit(inner_loc)
                ## inner_ref = ArrayRef(Id(loc_name), inner_loc)
                ## initstats.append(Assignment(inner_ref, Id('1')))
                

            rw.ArrayIdToDimName[loc_name] = rw.Local['size']
            rw.NumDims[loc_name] = rw.NumDims[n]
        # Must also create the barrier
        arglist = ArgList([Id('CLK_LOCAL_MEM_FENCE')])
        func = EmptyFuncDecl('barrier', type = [])
        func.arglist = arglist
        loadings.append(func)

        outerstats.insert(0, GroupCompound(loadings))
        outerstats.append(func)
        

    def localMemory(self, arrNames, west = 0, north = 0, east = 0, south = 0, middle = 1):
        rw = self.rw

        isInsideLoop = False
        try: 
            # Find out if arrName is inside a loop
            forLoops = ForLoops()
            forLoops.visit(rw.Kernel)
            forLoopAst = forLoops.ast
            arrays = Arrays([])
            arrays.visit(forLoopAst)
            for arrName in arrNames:
                if arrName in arrays.ids:
                    isInsideLoop = True
        except AttributeError:
            pass
            ## print "NOT INSIDE LOOP"

        if isInsideLoop:
            # find loop index
            loopIndices = LoopIndices()
            loopIndices.visit(forLoopAst)
            outeridx = loopIndices.index[0]
            forLoopAst.inc = Increment(Id(outeridx), '+='+ rw.Local['size'][0])
            
            inneridx = outeridx*2
            rw.Add[outeridx] = inneridx
            cond = BinOp(Id(inneridx), '<' , Constant(rw.Local['size'][0]))
            innerinc = Increment(Id(inneridx), '++')
            innercomp = copy.copy(forLoopAst.compound)
            innerloop = ForLoop(ConstantAssignment(inneridx), cond, \
                                innerinc, innercomp)
            forLoopAst.compound = Compound([innerloop])

        direction = [west, north, east, south, middle]
        dirname = [(0,-1), (1,0), (0,1), (-1,0), (0,0)]
        loadings = [elem for i, elem in enumerate(dirname)
                    if direction[i] == 1]
        if not loadings:
            loadings = [(0,0)]


        ## finding the correct local memory size
        arrName = arrNames[0]
        localDims = [int(rw.Local['size'][0])\
                     for i in xrange(rw.NumDims[arrName])]
        if rw.ParDim == 1 and len(localDims) == 2:
            localDims[0] = 1;
        arrIdx = rw.IndexInSubscript[arrName]
        localOffset = [int(rw.LowerLimit[i])\
                     for i in arrIdx]

        for (x,y) in loadings:
            localDims[0] += abs(x)
            if rw.NumDims[arrName] == 2:
                localDims[1] += abs(y)


        stats = []
        for arrName in arrNames:
            localName = arrName + '_local'
            arrayinit = '['
            for i, d in enumerate(localDims):
                arrayinit += str(d)
                if i == 0 and len(localDims) == 2:
                    arrayinit += '*'
            arrayinit += ']'

            localId = Id(localName + arrayinit)
            localTypeId = TypeId(['__local'] + [rw.Type[arrName][0]], localId)
            rw.NumDims[localName] = rw.NumDims[arrName]
            rw.LocalSwap[arrName] = localName
            rw.ArrayIdToDimName[localName] = [rw.Local['size'][0], rw.Local['size'][0]]
            stats.append(localTypeId)

        InitComp = GroupCompound(stats)
        stats2 = []
        LoadComp = GroupCompound(stats2)

        ## Insert local id with offset
        for i, offset in enumerate(localOffset):
            idd = rw.ReverseIdx[i] if len(localOffset) == 2 else i
            if offset != 0:
                
                rval = BinOp(Id('get_local_id('+str(idd)+')'), '+', \
                             Constant(offset))
            else:
                rval = Id('get_local_id('+str(idd)+')')
            lval = TypeId(['unsigned'], Id('l'+rw.GridIndices[i]))
            stats.append(Assignment(lval,rval))

        exchangeIndices = ExchangeIndices(rw.IndexToLocalVar,  rw.LocalSwap.values())
        
        ## Creating the loading of values into the local array.
        for arrName in arrNames:
            for k, l in enumerate(loadings):
                arrayId = Id(arrName)
                # get first ArrayRef
                aref = rw.LoopArrays[arrName][k]
                subscript = aref.subscript
                lsub = copy.deepcopy(subscript)
                lval = ArrayRef(Id(rw.LocalSwap[arrName]), lsub)
                rsub = copy.deepcopy(subscript)
                rval = ArrayRef(arrayId, rsub, extra = {'localMemory' : True})
                load = Assignment(lval,rval)
                exchangeId = ExchangeId(rw.IndexToLocalVar)
                orisub = subscript
                for m in orisub:
                    exchangeId.visit(m)
                if isInsideLoop:
                    for i, n in enumerate(orisub):
                        addToId = Ids()
                        addToId.visit(n)
                        # REMEMBER: orisub[i] might not simply be an Id
                        # might need to do something more complicated here
                        if outeridx in addToId.ids:
                            orisub[i] = Id(inneridx)

                    for i, n in enumerate(rsub): # GlobalLoad
                        idd = rw.ReverseIdx[i] if rw.NumDims[arrName] == 2 else i
                        locIdx = 'get_local_id('+str(idd)+')'
                        addToId = Ids()
                        addToId.visit(n)
                        if outeridx in addToId.ids:
                            rsub[i] = BinOp(rsub[i],'+',\
                            Id(locIdx))
                    for i, n in enumerate(lsub): # Local Write
                        idd = rw.ReverseIdx[i] if rw.NumDims[arrName] == 2 else i
                        locIdx = 'get_local_id('+str(idd)+')'
                        exchangeId = ExchangeId({''+outeridx : locIdx})
                        exchangeId.visit(n)                    

                stats2.append(load)

        # Must also create the barrier
        arglist = ArgList([Id('CLK_LOCAL_MEM_FENCE')])
        func = EmptyFuncDecl('barrier', type = [])
        func.arglist = arglist
        stats2.append(func)

        
        
        exchangeIndices.visit(InitComp)
        exchangeIndices.visit(LoadComp)
        if isInsideLoop:
            forLoopAst.compound.statements.insert(0,LoadComp)
            forLoopAst.compound.statements.append(func)
        else:
            rw.Kernel.statements.insert(0, LoadComp)
        rw.Kernel.statements.insert(0, InitComp)
        
        

    def transpose(self, arrName):
        rw = self.rw
        if rw.DefinesAreMade:
            print "Transposed must be called before SetDefine, returning..."
            return

        if rw.NumDims[arrName] != 2:
            print "Array ", arrName , "of dimension " , \
                  rw.NumDims[arrName], "cannot be transposed"
            return
        hstName = rw.HstId[arrName]
        hstTransName = hstName + '_trans'
        rw.GlobalVars[hstTransName] = ''
        rw.HstId[hstTransName] = hstTransName
        rw.Type[hstTransName] = rw.Type[arrName]
        # Swap the hst ptr
        rw.NameSwap[hstName] = hstTransName
        # Swap the dimension argument
        dimName = rw.ArrayIdToDimName[arrName]
        rw.NameSwap[dimName[0]] = dimName[1]

        if arrName not in rw.WriteOnly:
            lval = Id(hstTransName)
            natType = rw.Type[arrName][0]
            rval = Id('new ' + natType + '['\
                      + rw.Mem[arrName] + ']')
            rw.Transposition.statements.append(Assignment(lval,rval))
            arglist = ArgList([Id(hstName),\
                       Id(hstTransName),\
                       Id(dimName[0]),\
                       Id(dimName[1])])
            trans = FuncDecl(Id('transpose<'+natType+'>'), arglist, Compound([]))
            rw.Transposition.statements.append(trans)

        # Forget this and just swap subscripts immediately
        ## rw.SubSwap[arrName] = True

        for sub in rw.Subscript[arrName]:
            (sub[0], sub[1]) = \
                     (sub[1], sub[0])

