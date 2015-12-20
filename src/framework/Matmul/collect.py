import lan


class GlobalArrayIds(lan.NodeVisitor):
    def __init__(self):
        self.all_a_ids = set()
        self.local_a_ids = set()

    def visit_ArrayRef(self, node):
        name = node.name.name
        self.all_a_ids.add(name)
        for c_name, c in node.children():
            self.visit(c)

    def visit_ArrayTypeId(self, node):
        name = node.name.name
        self.local_a_ids.add(name)
        for c_name, c in node.children():
            self.visit(c)

    @property
    def ids(self):
        return self.all_a_ids - self.local_a_ids


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
        # if name == "hA":
        #     print self.current_parent
        self.local_na_ids.add(name)

    def visit_FuncDecl(self, node):
        self.visit(node.arglist)
        self.visit(node.compound)

    def visit_Id(self, node):
        name = node.name
        # if name == "db":
        #     print self.current_parent
        self.all_na_ids.add(name)

    @property
    def ids(self):
        return self.all_na_ids - self.local_na_ids - self.local_a_tids