import os, sys
import ast
import astunparse
import json
from collections import defaultdict
from .scope import ScopeManager

class FactManager(object):

    def __init__(self) -> None:
        self.invo_num = 0 
        self.var_num = 0
        self.func_num = 0
        self.heap_num = 0
        self.datalog_facts = {
            "AssignVar": [],
            "AssignGlobal": [],
            "AssignStrConstant": [],
            "AssignBoolConstant": [],
            "AssignBool": [],
            "AssignIntConstant": [],
            "AssignFloatConstant": [],
            "AssignBinOp": [],
            "AssignUnaryOp": [],
            "LoadField": [],
            "StoreField": [],
            "StoreFieldSSA": [],
            "LoadIndex": [],
            "StoreIndex": [],
            "StoreIndexSSA": [],
            "LoadSlice": [],
            "StoreSlice": [],
            "StoreSliceSSA": [],
            "Invoke": [],
            "CallGraphEdge": [],
            "ActualParam": [],
            "ActualKeyParam": [], 
            "FormalParam": [],
            "ActualReturn": [],
            "FormalReturn": [],
            # "MethodUpdate": [],
            "VarType": [],
            "SubType": [],
            "VarInMethod": [],
            "Alloc": [],
            "LocalMethod": [],
            "LocalClass": [],
            "InvokeInLoop": [],
            "NextInvoke": [],
            "InvokeLineno": []
        }


    def add_fact(self, fact_name, fact_tuple):
        # print(fact_name, fact_tuple)
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
        self.scopeManager = ScopeManager()
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
        self.import_map = {}
        self.meth2invokes = defaultdict(list)
        self.meth_in_loop = set()
        self.in_loop = False
        self.loop_vars = []
        self.in_class = False
        self.injected_methods = ["__phi__", "set_field_wrapper", "set_index_wrapper", "global_wrapper"]
    
    def load_type_map(self, json_path):
        with open(json_path) as f:
            self.type_map = json.load(f) 
        # Builtin Types
        self.type_map.update({'set':['module', 'set'],
                            'list':['module', 'list'],
                            'dict':['module', 'dict'],
                            'str':['module', 'str']})
        def filter_unbound(x):
            return ' | '.join([t for t in x.split(' | ') if t != "Unbound"])
        
        for varname, v in self.type_map.items():
            if v[0] == "var":
                self.FManager.add_fact("VarType", (varname, filter_unbound(v[1])))
        

    def import_map_get(self, key):
        if key in self.import_map:
            return self.import_map[key]
        return key

    def get_cur_sig(self):
        return self.scopeManager.get_cur_sig()
    
    def mark_localvars(self, varname):
        if self.scopeManager.in_globals(varname):
            self.FManager.add_fact("AssignGlobal", (varname, varname))
            return
        self.FManager.add_fact("VarInMethod", (varname, self.get_cur_sig()))

    def mark_loopcalls(self):
        for meth_name, loop_var in self.meth_in_loop:
            for invo in self.meth2invokes[meth_name]:
                self.FManager.add_fact("InvokeInLoop", (invo, loop_var))

    def build_invoke_graphs(self):
        for _, invos in self.meth2invokes.items():
            for (from_invo, to_invo) in zip(invos, invos[1:] + ["invo_end"]):
                self.FManager.add_fact("NextInvoke", (from_invo, to_invo))

    def add_loop_facts(self, cur_invo, meth_name):
        for loop_var in self.loop_vars:
            self.FManager.add_fact("InvokeInLoop", (cur_invo, loop_var))
            self.meth_in_loop.add((meth_name, loop_var))

    def visit_Body(self, body):
        if isinstance(body, list):
            new_values = []
            for value in body:
                if isinstance(value, ast.AST):
                    value = self.visit(value)
                    if value is None:
                        continue
                    elif not isinstance(value, ast.AST):
                        new_values.extend(value)
                        continue
                new_values.append(value)
            body[:] = new_values
        return body

    def visit_Module(self, node) :
        ret = ast.NodeTransformer.generic_visit(self, node)
        self.mark_loopcalls()
        self.build_invoke_graphs()
        return ret

    def visit_Import(self,node):
        return ast.NodeTransformer.generic_visit(self, node)

    def visit_ImportFrom(self,node):
        for name in node.names:
            assert type(name) == ast.alias
            self.import_map[name.name] = '.'.join([node.module, name.name])
        return ast.NodeTransformer.generic_visit(self, node)

    def visit_Global(self, node):
        self.scopeManager.update_globals(node.names)
        return node

    def visit_Nonlocal(self, node):
        self.scopeManager.update_globals(node.names)
        return node

    def visit_ClassDef(self, node):
        self.scopeManager.enterNamedBlock(node.name)
        self.in_class = True
        self.FManager.add_fact("LocalClass", (node.name,))
        for base in node.bases:
            base_type = self.type_map[base.id][1]
            if base_type.startswith("Type["):
                base_type = base_type[5:-1]
            self.FManager.add_fact("SubType", (node.name, base_type))
        self.visit_Body(node.body)
        self.in_class = False
        self.scopeManager.leaveNamedBlock()
        return node

    def visit_FunctionDef(self, node):
        self.scopeManager.enterNamedBlock(node.name)
        self.FManager.add_fact("LocalMethod", (self.get_cur_sig(),))
        
        meth = self.get_cur_sig()
        for i, arg in enumerate(node.args.args):
            self.mark_localvars(arg.arg)
            if self.in_class:
                self.FManager.add_fact("FormalParam", (i, meth, arg.arg))
            else:
                self.FManager.add_fact("FormalParam", (i+1, meth, arg.arg))
        self.visit_Body(node.body)
        if self.in_class and node.name == "__init__" and len(node.args.args) > 0:
            self.FManager.add_fact("Alloc", (node.args.args[0].arg, self.FManager.get_new_heap(), self.get_cur_sig()))
            self.FManager.add_fact("FormalReturn", (0, meth, node.args.args[0].arg))
        self.scopeManager.leaveNamedBlock()
        return node
    
    def visit_For(self, node):
        assert(type(node.iter) == ast.Name)
        assert(type(node.target) == ast.Name)

        self.mark_localvars(node.target.id)
        self.FManager.add_fact("LoadIndex", (node.target.id, node.iter.id, "index_placeholder"))
        self.in_loop = True
        self.loop_vars.append(node.iter.id)
        ret = ast.NodeTransformer.generic_visit(self, node)
        self.loop_vars.pop()
        self.in_loop = False
        return ret
    
    def visit_ExceptHandler(self, node):
        node.body = self.visit_Body(node.body)
        return node
    
    # async ast nodes
    def visit_AsyncFunctionDef(self, node):
        return self.visit_FunctionDef(node)

    def visit_AsyncFor(self, node):
        return self.visit_For(node)

    def visit_Return(self, node):
        if type(node.value) == ast.Name:
            self.FManager.add_fact("FormalReturn", (0, self.get_cur_sig(), node.value.id))
        elif type(node.value) == ast.Tuple:
            for i, x in enumerate(node.value.elts):
                assert type(x) == ast.Name
                self.FManager.add_fact("FormalReturn", (i, self.get_cur_sig(), x.id))
        return ast.NodeTransformer.generic_visit(self, node)
    
    def visit_Yield(self, node):
        if type(node.value) == ast.Name:
            self.FManager.add_fact("FormalReturn", (0, self.get_cur_sig(), node.value.id))
        elif type(node.value) == ast.Tuple:
            for i, x in enumerate(node.value.elts):
                assert type(x) == ast.Name
                self.FManager.add_fact("FormalReturn", (i, self.get_cur_sig(), x.id))
        return ast.NodeTransformer.generic_visit(self, node)

    def handle_assign_value(self, target, value):
        assert(type(target) == ast.Name)
        target_name = target.id
        self.mark_localvars(target_name)
        if type(value) == ast.Name:
            self.FManager.add_fact("AssignVar", (target_name, value.id))
        elif type(value) == ast.Call:
            # handle injected method
            if type(value.func) == ast.Name and value.func.id in self.injected_methods:
                if value.func.id == "set_field_wrapper":
                    self.FManager.add_fact("StoreFieldSSA", (target_name, value.args[0].id, value.args[1].value, value.args[2].id))
                elif value.func.id == "set_index_wrapper":
                    idx = value.args[1]
                    if type(idx) == ast.Name:
                        self.FManager.add_fact("StoreIndexSSA", (target_name, value.args[0].id, idx.id, value.args[2].id))
                    elif type(idx) == ast.Index:
                        assert type(idx.value) == ast.Name
                        self.FManager.add_fact("StoreIndexSSA", (target_name, value.args[0].id, idx.value.id, value.args[2].id))
                    elif type(idx) == ast.Call:
                        assert type(idx.func) == ast.Name
                        idx_ids = [x.id if type(x) == ast.Name else "none" for x in idx.args]
                        self.FManager.add_fact("StoreSliceSSA", (target_name, value.args[0].id, *idx_ids, value.args[2].id))
                    elif type(idx) == ast.Tuple:
                        self.FManager.add_fact("StoreIndexSSA", (target_name, value.args[0].id, "slice_placeholder", value.args[2].id))
                    else:
                        assert False, "Unknown slice!"
                elif value.func.id == "global_wrapper":
                    self.FManager.add_fact("AssignGlobal", (target_name, value.args[0].id))
                elif value.func.id == "__phi__":
                    self.FManager.add_fact("AssignVar", (target_name, value.args[0].id))
                    self.FManager.add_fact("AssignVar", (target_name, value.args[1].id))
                return
            cur_invo = self.visit_Call(value)
            self.FManager.add_fact("ActualReturn", (0, cur_invo, target_name))
            self.FManager.add_fact("Alloc", (target_name, self.FManager.get_new_heap(), self.get_cur_sig()))
        elif type(value) == ast.Constant:
            if type(value.value) == int:
                self.FManager.add_fact("AssignIntConstant", (target_name, value.value))
            elif type(value.value) == bool:
                self.FManager.add_fact("AssignBoolConstant", (target_name, value.value))
            elif type(value.value) == float:
                self.FManager.add_fact("AssignFloatConstant", (target_name, value.value))
            elif type(value.value) == str:
                self.FManager.add_fact("AssignStrConstant", (target_name, value.value.encode("unicode_escape").decode("utf-8")))
            self.FManager.add_fact("Alloc", (target_name, self.FManager.get_new_heap(), self.get_cur_sig()))
        # other literals
        elif type(value) in [ast.List, ast.Tuple, ast.Set]:
            if len(value.elts) <= 50 and ast.Name in [type(x) for x in value.elts]:
                for i, x in enumerate(value.elts):
                    if type(x) == ast.Name:
                        self.FManager.add_fact("StoreIndex", (target_name, i, x.id))
                    else:
                        assert type(x) ==  ast.Constant
            new_iter = self.meth_map[type(value)]()
            self.FManager.add_fact("Alloc", (new_iter, self.FManager.get_new_heap(), self.get_cur_sig()))
            self.FManager.add_fact("AssignVar", (target_name, new_iter))
        elif type(value) == ast.Dict:
            if len(value.values) <= 50 and ast.Name in [type(x) for x in value.values]:
                for k, v in zip(value.keys, value.values):
                    if type(v) == ast.Name:
                        if k == None:
                            self.FManager.add_fact("AssignVar", (target_name, v.id))
                        else:
                            k_literal = k.id if type(k) == ast.Name else k.value
                            self.FManager.add_fact("StoreIndex", (target_name, k_literal, v.id))
                    else:
                        assert type(v) ==  ast.Constant
            new_iter = self.meth_map[type(value)]()
            self.FManager.add_fact("Alloc", (new_iter, self.FManager.get_new_heap(), self.get_cur_sig()))
            self.FManager.add_fact("AssignVar", (target_name, new_iter))
        # comprehensions [TODO]
        elif type(value) in [ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp]:
            new_iter = self.meth_map[type(value)]()
            self.FManager.add_fact("Alloc", (new_iter, self.FManager.get_new_heap(), self.get_cur_sig()))
            self.FManager.add_fact("AssignVar", (target_name, new_iter))
        elif type(value) == ast.Lambda:
            new_iter = self.FManager.get_new_heap()
            self.FManager.add_fact("Alloc", (new_iter, self.FManager.get_new_heap(), self.get_cur_sig()))
            self.FManager.add_fact("AssignVar", (target_name, new_iter))
        elif type(value) == ast.Subscript:
            assert type(value.value) == ast.Name
            if type(value.slice) == ast.Index:
                assert type(value.slice.value) == ast.Name
                self.FManager.add_fact("LoadIndex", (target_name, value.value.id, value.slice.value.id))
            elif type(value.slice) == ast.Slice:
                slice_ids = [x.id if x else "none" for x in [value.slice.lower, value.slice.upper, value.slice.step]]
                self.FManager.add_fact("LoadSlice", (target_name, value.value.id, *slice_ids))
                self.FManager.add_fact("Alloc", (target_name, self.FManager.get_new_heap(), self.get_cur_sig())) # should be generated on the fly
            elif type(value.slice) == ast.ExtSlice:
                self.FManager.add_fact("LoadIndex", (target_name, value.value.id, "slice_placeholder"))
                self.FManager.add_fact("Alloc", (target_name, self.FManager.get_new_heap(), self.get_cur_sig())) # should be generated on the fly
        elif type(value) == ast.Attribute:
            assert type(value.value) == ast.Name
            self.FManager.add_fact("LoadField", (target_name, value.value.id, value.attr))
        elif type(value) == ast.BinOp:
            assert type(value.left) == ast.Name
            assert type(value.right) == ast.Name
            self.FManager.add_fact("AssignBinOp", (target_name, value.left.id, value.op.__class__.__name__, value.right.id))
            self.FManager.add_fact("Alloc", (target_name, self.FManager.get_new_heap(), self.get_cur_sig()))
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
        elif type(value) == ast.JoinedStr:
            self.FManager.add_fact("AssignStrConstant", (target_name, "str_placeholder"))
        else:
            print("Unkown source type! " + str(type(value)))
            assert 0

    def visit_Assign(self, node):
        for target in node.targets:
            if type(target) == ast.Name:
                self.handle_assign_value(target, node.value)
            elif type(target) == ast.Starred:
                self.handle_assign_value(target.value, node.value)
            elif type(target) == ast.Attribute:
                assert False, "Case deprecated!"
            elif type(target) == ast.Subscript:
                assert False, "Case deprecated!"
            elif type(target) == ast.Tuple:
                assert type(node.value) == ast.Call
                cur_invo = self.visit_Call(node.value)
                for i, t in enumerate(target.elts):
                    assert type(t) == ast.Name
                    self.mark_localvars(t.id)
                    self.FManager.add_fact("ActualReturn", (i, cur_invo, t.id))
                    self.FManager.add_fact("Alloc", (t.id, self.FManager.get_new_heap(), self.get_cur_sig()))
            else:
                assert False, "Unkown target type! " + str(type(target))

        return node
    
    def visit_Call(self, node):
        cur_invo = self.FManager.get_new_invo()
        self.FManager.add_fact("InvokeLineno", (cur_invo, node.lineno))
        self.meth2invokes[self.get_cur_sig()].append(cur_invo)
        if type(node.func) == ast.Attribute:
            hasInnerCall = self.visit_Attribute(node.func, cur_invo=cur_invo)
            # simulating invocations insde higher-order functions
            if hasInnerCall:
                new_invo = self.FManager.get_new_invo()
                self.FManager.add_fact("InvokeLineno", (new_invo, node.lineno))
                func_name = ""
                for kw in node.keywords:
                    if kw.arg == "func":
                        assert type(kw.value) == ast.Name
                        func_name = kw.value.id
                if func_name == "":
                    assert type(node.args[0]) == ast.Name
                    func_name = node.args[0].id
                self.meth2invokes[self.get_cur_sig()].append(new_invo)
                self.FManager.add_fact("Invoke", (new_invo, func_name, self.get_cur_sig()))
                if self.in_loop:
                    self.add_loop_facts(new_invo, node.args[0].id)
                self.FManager.add_fact("ActualParam", (1, new_invo, node.func.value.id))
                self.FManager.add_fact("ActualReturn", (0, new_invo, node.func.value.id))
        elif type(node.func) == ast.Name:
            self.FManager.add_fact("Invoke", (cur_invo, node.func.id, self.get_cur_sig()))
            if self.in_loop:
                self.add_loop_facts(cur_invo, node.func.id)
        else:
            print(type(node.func), ": Impossible!")
        self.visit_arguments(node.args, cur_invo=cur_invo)
        self.visit_keywords(node.keywords, cur_invo=cur_invo)
        return cur_invo


    def visit_Attribute(self, node, assigned = False, cur_invo = None):
        assert type(node.value) == ast.Name
        if cur_invo:
            value_type = self.type_map[node.value.id]
            method_sig = ".".join([value_type[1].replace('Self@', ''), node.attr])
            if value_type[0] == "var":
                self.FManager.add_fact("ActualParam", (0, cur_invo, node.value.id))
            self.FManager.add_fact("Invoke", (cur_invo, method_sig, self.get_cur_sig()))
            if self.in_loop:
                self.add_loop_facts(cur_invo, method_sig)
            if method_sig in ["pandas.Series.map", "pandas.Series.apply", "pandas.DataFrame.apply", "FrameOrSeries.apply",  "pandas.DataFrame.applymap"]:
                return True

    def visit_arguments(self, args, cur_invo=None):
        if type(args) == ast.arguments:
            return args
        for i, arg in enumerate(args):
            if type(arg) == ast.Starred:
                arg = arg.value
            assert type(arg) == ast.Name
            self.FManager.add_fact("ActualParam", (i + 1, cur_invo, arg.id))
        return args

    def visit_keywords(self, keywords, cur_invo):
        for keyword in keywords:
            assert type(keyword.value) == ast.Name
            self.FManager.add_fact("ActualKeyParam", (keyword.arg, cur_invo, keyword.value.id))
        return keywords

    # keep exprs below unchanged (for now)
    def visit_ListComp(self, node):
        return [], node

    def visit_SetComp(self, node):
        return [], node

    def visit_DictComp(self, node):
        return [], node

    def visit_GeneratorExp(self, node):
        return [], node

    def visit_comprehension(self, node):
        return [], node

    def visit_FormattedValue(self, node):
        return [], node

    def visit_JoinedStr(self, node):
        return [], node
