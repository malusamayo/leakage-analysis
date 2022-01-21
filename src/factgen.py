import os, sys
import ast
import astunparse
import json

class FactManager(object):

    def __init__(self) -> None:
        self.invo_num = 0 
        self.var_num = 0
        self.func_num = 0
        self.heap_num = 0
        self.datalog_facts = {
            "AssignVar": [],
            "AssignStrConstant": [],
            "AssignBoolConstant": [],
            "AssignBool": [],
            "AssignIntConstant": [],
            "AssignFloatConstant": [],
            "AssignBinOp": [],
            "AssignUnaryOp": [],
            "LoadField": [],
            "StoreField": [],
            "LoadIndex": [],
            "StoreIndex": [],
            "LoadSlice": [],
            "StoreSlice": [],
            "Invoke": [],
            "CallGraphEdge": [],
            "ActualParam": [],
            "ActualKeyParam": [], 
            "FormalParam": [],
            # "FormalKeyParam": [],
            "ActualReturn": [],
            "FormalReturn": [],
            # "MethodUpdate": [],
            "Alloc": []
        }


    def add_fact(self, fact_name, fact_tuple):
        print(fact_name, fact_tuple)
        fact_tuple = (str(t) for t in fact_tuple)
        self.datalog_facts[fact_name].append(fact_tuple)

    def get_new_invo(self):
        old_invo = self.invo_num
        self.invo_num += 1
        return "$invo" + str(old_invo)

    def get_new_var(self):
        old_var = self.var_num
        self.var_num += 1
        return "_var" + str(old_var)
    
    def get_new_func(self):
        old_func = self.func_num
        self.func_num += 1
        return "_func" + str(old_func)

    def get_new_heap(self):
        old_heap = self.heap_num
        self.heap_num += 1
        return "$heap" + str(old_heap)

    def get_new_list(self):
        old_var = self.var_num
        self.var_num += 1
        return "$list" + str(old_var)

    def get_new_tuple(self):
        old_var = self.var_num
        self.var_num += 1
        return "$tuple" + str(old_var)

    def get_new_set(self):
        old_var = self.var_num
        self.var_num += 1
        return "$set" + str(old_var)

    def get_new_dict(self):
        old_var = self.var_num
        self.var_num += 1
        return "$dict" + str(old_var)


class FactGenerator(ast.NodeVisitor):
    def __init__(self, json_path) -> None:
        super().__init__()
        self.FManager = FactManager()
        self.load_type_map(json_path)
        self.meth_map = {
            ast.Set: self.FManager.get_new_set,
            ast.Tuple: self.FManager.get_new_tuple,
            ast.List: self.FManager.get_new_list,
            ast.Dict: self.FManager.get_new_dict,
            ast.SetComp: self.FManager.get_new_set,
            ast.GeneratorExp: self.FManager.get_new_var,
            ast.ListComp: self.FManager.get_new_list,
            ast.DictComp: self.FManager.get_new_dict,
        }
        self.scope_stack = []
    
    def load_type_map(self, json_path):
        with open(json_path) as f:
            self.type_map = json.load(f) 

    def visit_Import(self,node):
        return ast.NodeTransformer.generic_visit(self, node)

    def visit_ImportFrom(self,node):
        return ast.NodeTransformer.generic_visit(self, node)

    def visit_FunctionDef(self, node, inClass=False):
        self.scope_stack.append(node.name)
        for i, arg in enumerate(node.args.args):
            self.FManager.add_fact("FormalParam", (i+1, node.name, arg.arg))
        ret = ast.NodeTransformer.generic_visit(self, node)
        self.scope_stack.pop()
        return ret
    
    def visit_For(self, node):
        assert(type(node.iter) == ast.Name)

        def visit_Iter(target, iter_id):
            for i, x in enumerate(target.elts):
                if type(x) == ast.Name:
                    self.FManager.add_fact("LoadIndex", (x.id, iter_id, "index_placeholder"))
                elif type(x) in [ast.Tuple, ast.List]:
                    visit_Iter(x, iter_id)
                else:
                    assert 0

        if type(node.target) == ast.Name:
            self.FManager.add_fact("LoadIndex", (node.target.id, node.iter.id, "index_placeholder"))
        elif type(node.target) in [ast.Tuple, ast.List]:
            visit_Iter(node.target, node.iter.id)
        else:
            assert 0
        return ast.NodeTransformer.generic_visit(self, node)
    
    def visit_Return(self, node, inClass=False):
        if type(node.value) == ast.Name:
            self.FManager.add_fact("FormalReturn", (0, self.scope_stack[-1], node.value.id))
        elif type(node.value) == ast.Tuple:
            for i, x in enumerate(node.value.elts):
                assert type(x) == ast.Name
                self.FManager.add_fact("FormalReturn", (i, self.scope_stack[-1], x.id))
        return ast.NodeTransformer.generic_visit(self, node)
    
    def visit_Yield(self, node, inClass=False):
        if type(node.value) == ast.Name:
            self.FManager.add_fact("FormalReturn", (0, self.scope_stack[-1], node.value.id))
        elif type(node.value) == ast.Tuple:
            for i, x in enumerate(node.value.elts):
                assert type(x) == ast.Name
                self.FManager.add_fact("FormalReturn", (i, self.scope_stack[-1], x.id))
        return ast.NodeTransformer.generic_visit(self, node)

    def handle_assign_value(self, target, value):
        assert(type(target) == ast.Name)
        target_name = target.id
        if type(value) == ast.Name:
            self.FManager.add_fact("AssignVar", (target_name, value.id))
        elif type(value) == ast.Call:
            cur_invo = self.visit_Call(value)
            self.FManager.add_fact("ActualReturn", (0, cur_invo, target_name))
            self.FManager.add_fact("Alloc", (target_name, self.FManager.get_new_heap()))
        elif type(value) == ast.Constant:
            if type(value.value) == int:
                self.FManager.add_fact("AssignIntConstant", (target_name, value.value))
            elif type(value.value) == bool:
                self.FManager.add_fact("AssignBoolConstant", (target_name, value.value))
            elif type(value.value) == float:
                self.FManager.add_fact("AssignFloatConstant", (target_name, value.value))
            elif type(value.value) == str:
                self.FManager.add_fact("AssignStrConstant", (target_name, value.value))
        # other literals
        elif type(value) in [ast.List, ast.Tuple, ast.Set]:
            if len(value.elts) <= 50 and ast.Name in [type(x) for x in value.elts]:
                for i, x in enumerate(value.elts):
                    if type(x) == ast.Name:
                        self.FManager.add_fact("StoreIndex", (target_name, i, x.id))
                    else:
                        assert type(x) ==  ast.Constant
            new_iter = self.meth_map[type(value)]()
            self.FManager.add_fact("Alloc", (new_iter, self.FManager.get_new_heap()))
            self.FManager.add_fact("AssignVar", (target_name, new_iter))
        elif type(value) == ast.Dict:
            if len(value.values) <= 50 and ast.Name in [type(x) for x in value.values]:
                for k, v in zip(value.keys, value.values):
                    if type(v) == ast.Name:
                        k_literal = k.id if type(k) == ast.Name else k.value
                        self.FManager.add_fact("StoreIndex", (target_name, k_literal, v.id))
                    else:
                        assert type(v) ==  ast.Constant
            new_iter = self.meth_map[type(value)]()
            self.FManager.add_fact("Alloc", (new_iter, self.FManager.get_new_heap()))
            self.FManager.add_fact("AssignVar", (target_name, new_iter))
        # comprehensions [TODO]
        elif type(value) in [ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp]:
            new_iter = self.meth_map[type(value)]()
            self.FManager.add_fact("Alloc", (new_iter, self.FManager.get_new_heap()))
            self.FManager.add_fact("AssignVar", (target_name, new_iter))
        elif type(value) == ast.Lambda:
            new_iter = self.FManager.get_new_heap()
            self.FManager.add_fact("Alloc", (new_iter, self.FManager.get_new_heap()))
            self.FManager.add_fact("AssignVar", (target_name, new_iter))
        elif type(value) == ast.Subscript:
            assert type(value.value) == ast.Name
            if type(value.slice) == ast.Index:
                assert type(value.slice.value) == ast.Name
                self.FManager.add_fact("LoadIndex", (target_name, value.value.id, value.slice.value.id))
            elif type(value.slice) == ast.Slice:
                slice_ids = [x.id if x else "none" for x in [value.slice.lower, value.slice.upper, value.slice.step]]
                self.FManager.add_fact("LoadSlice", (target_name, value.value.id, *slice_ids))
                self.FManager.add_fact("Alloc", (target_name, self.FManager.get_new_heap())) # should be generated on the fly
            elif type(value.slice) == ast.ExtSlice:
                self.FManager.add_fact("LoadIndex", (target_name, value.value.id, "slice_placeholder"))
                self.FManager.add_fact("Alloc", (target_name, self.FManager.get_new_heap())) # should be generated on the fly
        elif type(value) == ast.Attribute:
            assert type(value.value) == ast.Name
            self.FManager.add_fact("LoadField", (target_name, value.value.id, value.attr))
        elif type(value) == ast.BinOp:
            assert type(value.left) == ast.Name
            assert type(value.right) == ast.Name
            self.FManager.add_fact("AssignBinOp", (target_name, value.left.id, value.op.__class__.__name__, value.right.id))
        elif type(value) == ast.UnaryOp:
            assert type(value.operand) in [ast.Name, ast.Constant]
            if type(value.operand) == ast.Name:
                self.FManager.add_fact("AssignUnaryOp", (target_name, value.op.__class__.__name__, value.operand.id))
            elif type(value.operand) == ast.Constant:
                self.FManager.add_fact("AssignUnaryOp", (target_name, value.op.__class__.__name__, value.operand.value))
        elif type(value) == ast.Compare:
            assert type(value.left) == ast.Name
            self.FManager.add_fact("AssignVar", (target_name, value.left.id))
            for com in value.comparators:
                assert type(com) == ast.Name
                self.FManager.add_fact("AssignVar", (target_name, com.id)) # maybe vectors!! [TODO]
        elif type(value) == ast.BoolOp:
            for v in value.values:
                assert type(v) == ast.Name
                self.FManager.add_fact("AssignVar", (target_name, v.id))
        elif type(value) == ast.Starred:
            assert type(value.value) == ast.Name
            self.FManager.add_fact("LoadField", (target_name, value.value.id, "")) # better modeling? [TODO]
        elif type(value) == ast.IfExp:
            assert type(value.test) == ast.Name
            assert type(value.body) == ast.Name
            assert type(value.orelse) == ast.Name
            self.FManager.add_fact("AssignVar", (target_name, value.test.id))
            self.FManager.add_fact("AssignVar", (target_name, value.body.id))
            self.FManager.add_fact("AssignVar", (target_name, value.orelse.id))
        else:
            print("Unkown source type! " + str(type(value)))
            assert 0

    def visit_Assign(self, node):
        for target in node.targets:
            if type(target) == ast.Name:
                self.handle_assign_value(target, node.value)
            elif type(target) == ast.Attribute:
                assert type(target.value) == ast.Name
                assert type(node.value) == ast.Name
                self.FManager.add_fact("StoreField", (target.value.id, target.attr, node.value.id))
            elif type(target) == ast.Subscript:
                assert type(target.value) == ast.Name
                assert type(node.value) == ast.Name
                if type(target.slice) == ast.ExtSlice:
                    self.FManager.add_fact("StoreIndex", (target.value.id, "slice_placeholder", node.value.id))
                else:
                    assert type(target.slice) == ast.Index # Slice not handled yet
                    assert type(target.slice.value) == ast.Name
                    self.FManager.add_fact("StoreIndex", (target.value.id, target.slice.value.id, node.value.id))
            elif type(target) == ast.Tuple:
                assert type(node.value) == ast.Call
                cur_invo = self.visit_Call(node.value)
                for i, t in enumerate(target.elts):
                    assert type(t) == ast.Name
                    self.FManager.add_fact("ActualReturn", (i, cur_invo, t.id))
                    self.FManager.add_fact("Alloc", (t.id, self.FManager.get_new_heap()))
            else:
                print("Unkown target type! " + str(type(target)))
                assert 0

        return node
    

    def visit_Call(self, node):
        cur_invo = self.FManager.get_new_invo()
        if type(node.func) == ast.Attribute:
            hasInnerCall = self.visit_Attribute(node.func, cur_invo=cur_invo)
            if hasInnerCall:
                new_invo = self.FManager.get_new_invo()
                assert type(node.args[0]) == ast.Name
                self.FManager.add_fact("CallGraphEdge", (new_invo, node.args[0].id))
                self.FManager.add_fact("ActualParam", (1, new_invo, node.func.value.id))
                self.FManager.add_fact("ActualReturn", (0, new_invo, node.func.value.id))
        elif type(node.func) == ast.Name:
            self.FManager.add_fact("CallGraphEdge", (cur_invo, node.func.id))
        else:
            print("Impossible!")
        self.visit_arguments(node.args, cur_invo=cur_invo)
        self.visit_keywords(node.keywords, cur_invo=cur_invo)
        return cur_invo


    def visit_Attribute(self, node, assigned = False, cur_invo = None):
        assert type(node.value) == ast.Name
        if cur_invo:
            value_type = self.type_map[node.value.id]
            method_sig = ".".join([value_type[1], node.attr])
            if value_type[0] == "var":
                self.FManager.add_fact("ActualParam", (0, cur_invo, node.value.id))
            self.FManager.add_fact("CallGraphEdge", (cur_invo, method_sig))
            if method_sig in ["pandas.Series.map", "pandas.Series.apply", "pandas.DataFrame.apply", "FrameOrSeries.apply",  "pandas.DataFrame.applymap"]:
                return True

    def visit_arguments(self, args, cur_invo=None):
        if type(args) == ast.arguments:
            return args
        for i, arg in enumerate(args):
            assert type(arg) == ast.Name
            self.FManager.add_fact("ActualParam", (i + 1, cur_invo, arg.id))
        return args

    def visit_keywords(self, keywords, cur_invo):
        for keyword in keywords:
            assert type(keyword.value) == ast.Name
            self.FManager.add_fact("ActualKeyParam", (keyword.arg, cur_invo, keyword.value.id))
        return keywords


