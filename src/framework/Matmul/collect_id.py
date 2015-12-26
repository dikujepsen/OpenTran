import lan
import copy


class Ids(lan.NodeVisitor):
    """ Finds all unique IDs, excluding function IDs"""

    def __init__(self):
        self.ids = set()

    def visit_FuncDecl(self, node):
        if node.compound.statements:
            self.visit(node.arglist)

    def visit_Id(self, node):
        self.ids.add(node.name)


class GlobalNonArrayIds(lan.NodeVisitor):
    def __init__(self):
        self.HasNotVisitedFirst = True
        self.gnai = _NonArrayIdsInLoop()

    def visit_ForLoop(self, node):
        if self.HasNotVisitedFirst:
            self.gnai.visit(node)
            self.HasNotVisitedFirst = False

    @property
    def ids(self):
        return self.gnai.ids


class _NonArrayIdsInLoop(lan.NodeVisitor):
    def __init__(self):
        self.all_na_ids = set()
        self.local_na_ids = set()
        self.local_a_tids = set()

    def visit_ArrayRef(self, node):
        for subnode in node.subscript:
            self.visit(subnode)

    def visit_ArrayTypeId(self, node):
        name = node.name.name
        self.local_a_tids.add(name)
        for subnode in node.subscript:
            self.visit(subnode)

    def visit_TypeId(self, node):
        name = node.name.name
        self.local_na_ids.add(name)

    def visit_FuncDecl(self, node):
        self.visit(node.arglist)
        self.visit(node.compound)

    def visit_Id(self, node):
        name = node.name
        self.all_na_ids.add(name)

    @property
    def ids(self):
        return self.all_na_ids - self.local_na_ids - self.local_a_tids


class GlobalTypeIds(lan.NodeVisitor):
    """ Finds type Ids """

    def __init__(self):
        self.ids = set()
        self.types = dict()

    def visit_TypeId(self, node):
        name = node.name.name
        self.ids.add(name)
        self.types[name] = node.type

    def visit_ArrayTypeId(self, node):
        name = node.name.name
        self.ids.add(name)
        self.types[name] = copy.deepcopy(node.type)
        if len(node.type) != 2:
            # "ArrayTypeId: Need to check, type of array is ", node.type
            # self.dictIds[name].append('*')
            pass

    def visit_ForLoop(self, node):
        pass
