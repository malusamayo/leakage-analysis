import ast
import astunparse


class FactManager(object):

    def __init__(self) -> None:
        self.invo_num = 0 
        self.var_num = 0
        self.heap_num = 0
        self.datalog_facts = {
            "AssignVar": [],
            "AssignStrConstant": [],
            "AssignBoolConstant": [],
            "AssignIntConstant": [],
            "AssignFloatConstant": [],
            "LoadField": [],
            "StoreField": [],
            "LoadIndex": [],
            "StoreIndex": [],
            "AssignInvoke": [],
            "Invoke": [],
            "CallGraphEdge": [],
            "ActualParam": [],
            "ActualKeyParam": [], 
            "FormalParam": [],
            "FormalKeyParam": [],
            "MethodUpdate": [],
            "Alloc": []
        }


    def add_fact(self, fact_name, fact_tuple):
        print(fact_name, fact_tuple)
        self.datalog_facts[fact_name].append(fact_tuple)

    def get_new_invo(self):
        old_invo = self.invo_num
        self.invo_num += 1
        return "$invo" + str(old_invo)

    def get_new_var(self):
        old_var = self.var_num
        self.var_num += 1
        return "_var" + str(old_var)

    def get_new_heap(self):
        old_heap = self.heap_num
        self.heap_num += 1
        return "$heap" + str(old_heap)

code = '''
a = b = 1
x = A()
y = B()
x.f = y.g
# y = t.k
a, b = b, a
df['Fare'] = df['Fare'].fillna(np.mean(df['Fare']), inplace = True)
'''

type_map = {
    "np": ["module", "numpy"],
    "_var5": ["var", "pandas.Series"]
}

name_map = {}

FManager = FactManager()

class FactGenerator(ast.NodeVisitor):
    def visit_Import(self,node):
        return ast.NodeTransformer.generic_visit(self, node)

    def visit_ImportFrom(self,node):
        return ast.NodeTransformer.generic_visit(self, node)

    def handle_assign_value(self, target, value):
        assert(type(target) == ast.Name)
        target_name = target.id
        if type(value) == ast.Name:
            FManager.add_fact("AssignVar", (target_name, value.id))
        elif type(value) == ast.Call:
            cur_invo = self.visit_Call(value)
            FManager.add_fact("AssignInvoke", (target_name, cur_invo))
            FManager.add_fact("Alloc", (target_name, FManager.get_new_heap()))
        elif type(value) == ast.Constant:
            if type(value.value) == int:
                FManager.add_fact("AssignIntConstant", (target_name, value.value))
            elif type(value.value) == bool:
                FManager.add_fact("AssignBoolConstant", (target_name, value.value))
            elif type(value.value) == float:
                FManager.add_fact("AssignFloatConstant", (target_name, value.value))
            elif type(value.value) == str:
                FManager.add_fact("AssignStrConstant", (target_name, value.value))
        elif type(value) == ast.Subscript:
            assert type(value.value) == ast.Name
            assert type(value.slice) == ast.Index # Slice not handled yet
            assert type(value.slice.value) == ast.Name
            FManager.add_fact("LoadIndex", (target_name, value.value.id, value.slice.value.id))
        elif type(value) == ast.Attribute:
            assert type(value.value) == ast.Name
            FManager.add_fact("LoadField", (target_name, value.value.id, value.attr))
        else:
            print("Unkown source type! " + str(type(value)))
            assert 0

    def visit_Assign(self,node):
        # print('Node type: Assign and fields: ', node.targets)
        for target in node.targets:
            if type(target) == ast.Name:
                self.handle_assign_value(target, node.value)
            elif type(target) == ast.Attribute:
                assert type(target.value) == ast.Name
                assert type(node.value) == ast.Name
                FManager.add_fact("StoreField", (target.value.id, target.attr, node.value.id))
            elif type(target) == ast.Subscript:
                assert type(target.value) == ast.Name
                assert type(target.slice) == ast.Index # Slice not handled yet
                assert type(target.slice.value) == ast.Name
                assert type(node.value) == ast.Name
                FManager.add_fact("StoreIndex", (target.value.id, target.slice.value, node.value.id))
            else:
                print("Unkown target type! " + str(type(target)))
                assert 0

        return node
    
    # def visit_Subscript(self, node, assigned = False):
    #     new_var = ""
    #     if type(node.value) == ast.Name:
    #         new_var = self.visit_Name(node.value, assigned=assigned)
    #     else:
    #         new_var = FManager.get_new_var()
    #         self.visit_Assign(ast.Assign([ast.Name(new_var)], node.value))
    #     idx_new_var = FManager.get_new_var()
    #     self.visit_Assign(ast.Assign([ast.Name(idx_new_var)], node.slice))
    #     return new_var, idx_new_var

    # def visit_BinOp(self, node):
    #     print('Node type: BinOp and fields: ', node._fields)
    #     return ast.NodeTransformer.generic_visit(self, node)

    def visit_Call(self, node):
        cur_invo = FManager.get_new_invo()
        if type(node.func) == ast.Attribute:
            self.visit_Attribute(node.func, cur_invo=cur_invo)
        elif type(node.func) == ast.Name:
            FManager.add_fact("CallGraphEdge", (cur_invo, node.func.id))
        else:
            print("Impossible!")
        self.visit_arguments(node.args, cur_invo=cur_invo)
        self.visit_keywords(node.keywords, cur_invo=cur_invo)
        # print(node.func, node.args, node.keywords)
        return cur_invo


    def visit_Attribute(self, node, assigned = False, cur_invo = None):
        assert type(node.value) == ast.Name
        if cur_invo:
            value_type = type_map[node.value.id]
            method_sig = ".".join([value_type[1], node.attr])
            if value_type[0] == "var":
                FManager.add_fact("ActualParam", (0, cur_invo, node.value.id))
            FManager.add_fact("CallGraphEdge", (cur_invo, method_sig))

    def visit_arguments(self, args, cur_invo):
        for i, arg in enumerate(args):
            assert type(arg) == ast.Name
            FManager.add_fact("ActualParam", (i + 1, cur_invo, arg.id))
        return args

    def visit_keywords(self, keywords, cur_invo):
        for keyword in keywords:
            assert type(keyword.value) == ast.Name
            FManager.add_fact("ActualKeyParam", (keyword.arg, cur_invo, keyword.value.id))
        return keywords


class CodeTransformer(ast.NodeTransformer):
    def generic_visit(self, node):
        rets = ast.NodeTransformer.generic_visit(self, node)
        if type(rets) != tuple:
            return rets, ""
        else:
            return rets

    def handle_assign_value(self, target, value):
        assert(type(target) == ast.Name)
        nodes = []

        if type(value) == ast.Name:
            nodes1, target_name = self.visit_Name(target, assigned = True)
            nodes = nodes + nodes1 + [ast.Assign([ast.Name(target_name)], value)]
        elif type(value) == ast.Call:
            nodes, call = self.visit_Call(value)
            nodes1, target_name = self.visit_Name(target, assigned = True)
            nodes = nodes + nodes1 + [ast.Assign([ast.Name(target_name)], call)]
        elif type(value) == ast.Constant:
            nodes1, target_name = self.visit_Name(target, assigned = True)
            nodes = nodes + nodes1 + [ast.Assign([ast.Name(target_name)], value)]
        elif type(value) == ast.Subscript:
            nodes1, target_name = self.visit_Name(target, assigned = True)
            nodes2, subscript = self.visit_Subscript(value)
            nodes = nodes + nodes1 + nodes2 + [ast.Assign([ast.Name(target_name)], subscript)]
        elif type(value) == ast.Attribute:
            nodes1, target_name = self.visit_Name(target, assigned = True)
            nodes2, attr = self.visit_Attribute(value)
            nodes = nodes + nodes1 + nodes2 + [ast.Assign([ast.Name(target_name)], attr)]
        elif type(value) == ast.Index:
            nodes += self.handle_assign_value(target, value.value)
        else:
            print("Unkown source type! " + str(type(value)))
            assert 0
        return nodes

    def visit_Assign(self,node):
        # print('Node type: Assign and fields: ', node.targets)
        nodes = []
        for target in node.targets:
            if type(target) == ast.Name:
                nodes += self.handle_assign_value(target, node.value)
            elif type(target) == ast.Attribute:
                nodes1, attr = self.visit_Attribute(target) #, assigned=True)
                from_new_var = FManager.get_new_var()
                nodes2 = self.visit_Assign(ast.Assign([ast.Name(from_new_var)], node.value))
                nodes = nodes + nodes1 + nodes2 + [ast.Assign([attr], ast.Name(from_new_var))]
            elif type(target) == ast.Subscript:
                nodes1, subscript = self.visit_Subscript(target)
                from_new_var = FManager.get_new_var()
                nodes2 = self.visit_Assign(ast.Assign([ast.Name(from_new_var)], node.value))
                nodes = nodes + nodes1 + nodes2 + [ast.Assign([subscript], ast.Name(from_new_var))]
            elif type(target) == ast.Tuple:
                if type(node.value) == ast.Tuple and len(target.elts) == len(node.value.elts):
                    new_vars = []
                    for v in node.value.elts:
                        to_new_var = FManager.get_new_var()
                        new_vars.append(to_new_var)
                        nodes += self.visit_Assign(ast.Assign([ast.Name(to_new_var)], v))
                    for v_name, t in zip(new_vars, target.elts):
                        nodes += self.visit_Assign(ast.Assign([t], ast.Name(v_name)))
                else:
                    assert 0
            else:
                print("Unkown target type! " + str(type(target)))
                assert 0

        return nodes
    
    def visit_Subscript(self, node, assigned = False):
        nodes = []
        new_var = ""
        if type(node.value) == ast.Name:
            nodes1, new_var = self.visit_Name(node.value, assigned=assigned)
            nodes += nodes1
        else:
            new_var = FManager.get_new_var()
            nodes += self.visit_Assign(ast.Assign([ast.Name(new_var)], node.value))
        if type(node.slice) == ast.Slice:
            nodes2, slice = self.visit_Slice(node.slice)
        elif type(node.slice) == ast.Index:
            nodes2, slice = self.visit_Index(node.slice)
        else:
            assert 0
        return nodes + nodes2, ast.Subscript(ast.Name(new_var), slice, node.ctx)

    def visit_Index(self, node):
        nodes = []
        new_var = FManager.get_new_var()
        nodes += self.visit_Assign(ast.Assign([ast.Name(new_var)], node.value))
        return nodes, ast.Index(ast.Name(new_var))

    def visit_Slice(self, node):
        nodes = []
        nodes1, l = self.generic_visit(node.lower)
        nodes2, h = self.generic_visit(node.higher)
        nodes3, stp = self.generic_visit(node.step)
        return nodes + nodes1 + nodes2 + nodes3, ast.Slice(l, h, stp)

    def visit_BinOp(self, node):
        print('Node type: BinOp and fields: ', node._fields)
        return ast.NodeTransformer.generic_visit(self, node)

    def visit_Call(self, node):
        nodes = []
        if type(node.func) == ast.Attribute:
            nodes1, base = self.visit_Attribute(node.func)
        elif type(node.func) == ast.Name:
            nodes1, name = self.visit_Name(node.func)
            base = ast.Name(name)
        nodes2, args = self.visit_arguments(node.args)
        nodes3, kws = self.visit_keywords(node.keywords)
        # print(node.func, node.args, node.keywords)
        return nodes + nodes1 + nodes2 + nodes3, ast.Call(base, args, kws)


    def visit_Attribute(self, node, assigned = False):
        nodes = []
        if type(node.value) != ast.Name:
            to_new_var = FManager.get_new_var()
            nodes1 = self.visit_Assign(ast.Assign([ast.Name(to_new_var)], node.value))
            nodes += nodes1
        else:
            nodes1, to_new_var = self.visit_Name(node.value, assigned=assigned)
            nodes += nodes1
        return nodes, ast.Attribute(ast.Name(to_new_var), node.attr)

    def visit_arguments(self, args):
        nodes = []
        arg_names = []
        for i, arg in enumerate(args):
            new_var = FManager.get_new_var()
            nodes += self.visit_Assign(ast.Assign([ast.Name(new_var)],arg))
            arg_names.append(ast.Name(new_var))
        return nodes, arg_names

    def visit_keywords(self, keywords):
        nodes = []
        kws = []
        for keyword in keywords:
            new_var = FManager.get_new_var()
            nodes += self.visit_Assign(ast.Assign([ast.Name(new_var)], keyword.value))
            kws.append(ast.keyword(arg = keyword.arg, value = ast.Name(new_var)))
        return nodes, kws

    def visit_Num(self, node):
        return node

    def visit_Name(self, node, assigned = False):
        nodes = []
        if node.id not in name_map:
            name_map[node.id] = node.id
            return nodes, name_map[node.id]
        if assigned:
            old_id = name_map[node.id]
            try:
                num = int(old_id.split("$")[-1])
                name_map[node.id] = node.id + '$' + str(num + 1)
            except ValueError:
                name_map[node.id] = node.id + '$0'
            # nodes += self.visit_Assign(ast.Assign([ast.Name(name_map[node.id])], ast.Name(old_id)))
        return nodes, name_map[node.id]

    def visit_Str(self, node):
        return node



p = ast.parse(code)

v = CodeTransformer()
new_tree, _ = v.visit(p)
new_code = astunparse.unparse(new_tree)
print(new_code)

f = FactGenerator()
f.visit(p)
# print(FManager.datalog_facts)