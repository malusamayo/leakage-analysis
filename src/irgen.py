import os, sys
import ast
import copy
import astunparse
import json
from collections import defaultdict
from .scope import ScopeManager
from .factgen import FactManager

# definition of injected functions, for the convience of type checking
phi_def_code = '''
def __phi__(phi_0, phi_1):
    if phi_0:
        return phi_0 
    return phi_1
def set_field_wrapper(base, attr, value):
    setattr(base, attr, value)
    return base
def set_index_wrapper(base, attr, value):
    setattr(base, attr, value)
    return base
def global_wrapper(x):
    return x

'''

'''
Transform code to a simpler IR, which is easier to translate to datalog facts
The exact semantics may not be equivalent
'''
class CodeTransformer(ast.NodeTransformer):
    def __init__(self, ignored_vars) -> None:
        super().__init__()
        self.FManager = FactManager()
        self.scopeManager = ScopeManager(ignored_vars)
        self.unchanged_nodeclasses = [ast.Global, ast.Nonlocal, ast.Pass, ast.Break, ast.Continue, ast.Import, ast.ImportFrom, ast.alias]

    def visit(self, node):
        """Visit a node."""
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        rets = ast.NodeTransformer.generic_visit(self, node)
        if type(node) not in self.unchanged_nodeclasses + [ast.Expr]:
            print(type(node))
            assert False
        return rets

    def visit_Module(self, node):
        phi_tree = ast.parse(phi_def_code)
        node.body = phi_tree.body + self.visit_Body(node.body)
        return node

    def visit_Body(self, body):
        self.scopeManager.enterBlock()
        if isinstance(body, list):
            new_values = []
            for value in body:
                if isinstance(value, ast.AST):   
                    if hasattr(value, "lineno"):
                        saved_lineno = value.lineno
                    else:
                        saved_lineno = -1
                    value = self.visit(value)
                    if value is None:
                        continue
                    elif not isinstance(value, ast.AST):
                        if saved_lineno != -1:
                            for v in value:
                                v.lineno = saved_lineno
                        new_values.extend(value)
                        continue
                new_values.append(value)
            body[:] = new_values
        self.scopeManager.leaveBlock()
        return body

    def visit_alias(self, node):
        self.scopeManager.defined_names.add(node.name)
        if node.asname:
            self.scopeManager.defined_names.add(node.asname)
        return node

    def visit_ClassDef(self, node):
        self.scopeManager.enterNamedBlock(node.name)
        nodes = []
        for i, base in enumerate(node.bases):
            nodes1, node.bases[i] = self.visitNameOnly(base)
            nodes += nodes1
        node.body = self.visit_Body(node.body)
        self.scopeManager.leaveNamedBlock()
        return nodes + [node]

    def visit_FunctionDef(self, node):
        self.scopeManager.enterNamedBlock(node.name)
        self.scopeManager.build_arg_map(node.args)
        
        node.body = self.visit_Body(node.body)
        self.scopeManager.leaveNamedBlock()
        return node

    def visit_Lambda(self, node):
        func_name = self.FManager.get_new_func()
        func_def = ast.FunctionDef(func_name, node.args, [ast.Return(node.body, lineno = node.body.lineno)], [], lineno=node.lineno)
        return [self.visit(func_def)], ast.Name(func_name)

    def visit_Return(self, node):
        nodes1 = []
        if node.value:
            nodes1, node.value = self.visitOnly(node.value, [ast.Tuple])
        return nodes1 + [node]
    
    def visit_Yield(self, node):
        nodes1 = []
        if node.value:
            nodes1, node.value = self.visitOnly(node.value, [ast.Tuple])
        return nodes1, node

    def visit_YieldFrom(self, node):
        nodes1 = []
        if node.value:
            nodes1, node.value = self.visitOnly(node.value, [ast.Tuple])
        return nodes1, node

    def visit_For(self, node):
        nodes, node.iter = self.visitNameOnly(node.iter)

        if not type(node.target) == ast.Name:
            new_var = ast.Name(self.FManager.get_new_var())
            node.body = [ast.Assign([node.target], new_var, lineno=node.lineno)] + node.body
            node.target = new_var
        nodes1, node.target = self.visit_Name(node.target, assigned=True)
        nodes += nodes1
        
        ctx1 = self.scopeManager.get_tmp_new_ctx()
        node.body = self.visit_Body(node.body)

        ctx2 = self.scopeManager.get_tmp_new_ctx()
        node.orelse = self.visit_Body(node.orelse)

        inits, phi_calls = self.scopeManager.resolve_upates(ctx1, ctx2, self.scopeManager.ctx)
        return nodes + inits + [node] + phi_calls

    def visit_While(self, node):
        nodes, node.test = self.visitNameOnly(node.test)

        ctx1 = self.scopeManager.get_tmp_new_ctx()
        node.body = self.visit_Body(node.body)

        ctx2 = self.scopeManager.get_tmp_new_ctx()
        node.orelse = self.visit_Body(node.orelse)

        inits, phi_calls = self.scopeManager.resolve_upates(ctx1, ctx2, self.scopeManager.ctx)

        return nodes + inits + [node] + phi_calls

    def visit_If(self, node):
        nodes, node.test = self.visitNameOnly(node.test)
        
        ctx1 = self.scopeManager.get_tmp_new_ctx()
        node.body = self.visit_Body(node.body)
        
        ctx2 = self.scopeManager.get_tmp_new_ctx()
        node.orelse = self.visit_Body(node.orelse)

        inits, phi_calls = self.scopeManager.resolve_upates(ctx1, ctx2, self.scopeManager.ctx)

        return nodes + inits + [node] + phi_calls

    def visit_IfExp(self, node):
        nodes, new_test = self.visitNameOnly(node.test)
        nodes1, new_body = self.visitNameOnly(node.body)
        nodes2, new_orelse = self.visitNameOnly(node.orelse)
        return nodes + nodes1 + nodes2, ast.IfExp(new_test, new_body, new_orelse)

    def visit_Try(self, node):
        node.body = self.visit_Body(node.body)
        node.handlers = self.visit_Body(node.handlers)
        node.orelse = self.visit_Body(node.orelse)
        node.finalbody = self.visit_Body(node.finalbody)
        # phi function here [TODO]
        return node

    def visit_ExceptHandler(self, node):
        node.body = self.visit_Body(node.body)
        return node

    def visit_With(self, node):
        nodes = []
        for item in node.items:
            nodes1, item.context_expr = self.visit(item.context_expr)
            nodes += nodes1
            if item.optional_vars:
                nodes += self.visit(ast.Assign([item.optional_vars], item.context_expr, lineno=node.lineno))
        ctx1 = self.scopeManager.get_tmp_new_ctx()
        node.body = self.visit_Body(node.body)

        ctx2 = self.scopeManager.get_tmp_new_ctx()
        inits, phi_calls = self.scopeManager.resolve_upates(ctx1, ctx2, self.scopeManager.ctx)
        return nodes + inits + [node] + phi_calls

    def visit_Delete(self, node):
        nodes = []
        new_vars = []
        for t in node.targets:
            nodes1, new_v = self.visitNameOnly(t)
            new_vars.append(new_v)
            nodes += nodes1
        return nodes + [ast.Delete(new_vars)]
    
    def visit_Raise(self, node):
        nodes1, new_exec = [], None
        nodes2, new_cause = [], None
        if node.exc:
            nodes1, new_exec = self.visitNameOnly(node.exc)
        if node.cause:
            nodes2, new_cause = self.visitNameOnly(node.cause)
        return nodes1 + nodes2 + [ast.Raise(new_exec, new_cause)]

    # async ast nodes
    def visit_AsyncFunctionDef(self, node):
        return self.visit_FunctionDef(node)

    def visit_AsyncFor(self, node):
        return self.visit_For(node)
    
    def visit_AsyncWith(self, node):
        return self.visit_With(node)
    
    def visit_Await(self, node):
        return self.visit(node.value)
    
    # ignore pattern matching for now

    def visit_Assert(self, node):
        nodes1, new_test = self.visitNameOnly(node.test)
        nodes2, new_msg = [], None
        if node.msg:
            nodes2, new_msg = self.visitNameOnly(node.msg)
        return nodes1 + nodes2 + [ast.Assert(new_test, new_msg)]

    def visit_Expr(self, node):
        node_saved = copy.deepcopy(node)
        rets = self.generic_visit(node)
        if len(rets.value) == 2 and type(rets.value[0]) == list:
            if type(rets.value[1]) == ast.Call:
                nodes = self.handle_call_updates(rets.value[1], node_saved.value)
                if nodes:
                    return rets.value[0] + nodes
            return rets.value[0] + [ast.Expr(rets.value[1])]
        return rets

    def handle_call_updates(self, call, call_saved, assigned_var=None):
        if type(call.func) == ast.Attribute and call.func.attr in ["fit", "fit_generator"]:
            if type(call_saved.func) == ast.Attribute and type(call_saved.func.value) == ast.Name:
                src_name = call_saved.func.value
            else:
                src_name = call.func.value
            nodes, new_base = self.visit_Name(src_name, assigned=True)
            if assigned_var:
                return nodes + [ast.Assign([new_base], assigned_var)]
            else:
                return nodes + [ast.Assign([new_base], call)]
        return []

    def handle_assign_value(self, target, value):
        assert(type(target) == ast.Name)
        nodes = []
        if type(value) == ast.Index:
            nodes += self.handle_assign_value(target, value.value)
        else:
            nodes, new_node = self.visit(value)
            nodes1, target = self.visit_Name(target, assigned = True)
            nodes = nodes + nodes1 + [ast.Assign([target], new_node)]
            if type(value) == ast.Call:
                nodes += self.handle_call_updates(new_node, value, target)
        return nodes

    def handle_single_assign(self, target, value):
        nodes = []
        if type(target) == ast.Name:
            nodes += self.handle_assign_value(target, value)
        elif type(target) == ast.Starred:
            nodes1, new_target = self.visit(target)
            nodes += nodes1
            nodes += self.handle_assign_value(new_target.value, value)
            nodes[-1].targets[0] = new_target
        elif type(target) in [ast.Attribute, ast.Subscript]:
            nodes1, new_target = self.visit(target)
            nodes2, new_value = self.visitNameOnly(value)
            if type(new_target) == ast.Attribute:
                new_attr = ast.Constant(new_target.attr, "") 
            else:
                def handle_slice(slice):
                    if type(slice) == ast.Index:
                       return slice
                    elif type(slice) == ast.Slice:
                        slice = [x if x else ast.Constant(None, kind="") for x in [slice.lower, slice.upper, slice.step]]
                        new_attr = ast.Call(ast.Name("slice"), slice, [])
                        return new_attr
                if type(new_target.slice) == ast.Index:
                    new_attr = new_target.slice
                elif type(new_target.slice) == ast.Slice:
                    new_attr = handle_slice(new_target.slice)
                elif type(new_target.slice) == ast.ExtSlice:
                    new_attr = ast.Tuple([handle_slice(slice) for slice in new_target.slice.dims])
            nodes3, new_name = self.visit_Name(target.value if type(target.value) == ast.Name else new_target.value, assigned=True)
            func = ast.Name("set_field_wrapper") if type(new_target) == ast.Attribute else ast.Name("set_index_wrapper")
            new_assign = [ast.Assign([new_name], ast.Call(func, [new_target.value, new_attr, new_value], []))]
            # new_assign = [ast.Assign([new_target], new_value)]
            nodes = nodes + nodes1 + nodes2 + nodes3 + new_assign # [ast.Assign([new_target], new_value)]
        elif type(target) in [ast.Tuple, ast.List]:
            if type(value) in [ast.Tuple, ast.List] and len(target.elts) == len(value.elts):
                new_vars = []
                for v in value.elts:
                    nodes1, new_v = self.visitNameOnly(v)
                    new_vars.append(new_v)
                    nodes += nodes1
                for v, t in zip(new_vars, target.elts):
                    nodes += self.visit_Assign(ast.Assign([t], v))
            elif type(value) == ast.Call:
                new_vars = []
                for v in target.elts:
                    nodes0, new_var = self.visit_Name(ast.Name(self.FManager.get_new_var()), assigned=True)
                    new_vars.append(new_var)
                    nodes += nodes0
                nodes1, call = self.visit_Call(value)
                nodes += nodes1 + [ast.Assign([ast.Tuple(new_vars)], call)]
                for v, var in zip(target.elts, new_vars):
                    nodes += self.visit_Assign(ast.Assign([v], var))
            else:
                new_vars = [self.FManager.get_new_var() for v in target.elts]
                nodes1, new_node = self.visitNameOnly(value)
                nodes += nodes1
                for i, t in enumerate(target.elts):
                    nodes += self.visit_Assign(ast.Assign([t], ast.Subscript(new_node, ast.Index(ast.Constant(i, "")))))
        else:
            assert 0, "Unkown target type! " + str(type(target))
        return nodes

    def visit_AnnAssign(self, node):
        if node.value:
            nodes, node.annotation = self.visitNameOnly(node.annotation)
            nodes += self.handle_single_assign(node.target, node.value)
            if type(node.target) == ast.Name and type(nodes[-1]) == ast.Assign:
                nodes[-1] = ast.AnnAssign(nodes[-1].targets[0], node.annotation, nodes[-1].value, simple = node.simple)
            return nodes
        else:
            return [node]
    
    def visit_AugAssign(self, node):
        return self.visit_Assign(ast.Assign([node.target], ast.BinOp(node.target, node.op, node.value)))

    def visit_Assign(self,node):
        nodes = []
        # [TODO] value should only be evaluated once
        for target in node.targets:
            nodes += self.handle_single_assign(target, node.value)
        return nodes

    def visit_Subscript(self, node, assigned = False):
        nodes = []
        nodes1, new_v = self.visitNameOnly(node.value)
        if type(node.slice) == ast.Slice:
            nodes2, slice = self.visit_Slice(node.slice)
        elif type(node.slice) == ast.Index:
            nodes2, slice = self.visit_Index(node.slice)
        elif type(node.slice) == ast.ExtSlice:
            nodes2, slice = self.visit_ExtSlice(node.slice)
        else:
            assert 0
        return nodes + nodes1 + nodes2, ast.Subscript(new_v, slice)

    def visit_Index(self, node):
        nodes = []
        nodes1, new_var = self.visitNameOnly(node.value)
        return nodes + nodes1, ast.Index(new_var)

    def visit_Slice(self, node):
        nodes = []

        def visitMaybe(self, node):
            return self.visitNameOnly(node) if node else ([], node)
        
        nodes1, l = visitMaybe(self, node.lower)
        nodes2, h = visitMaybe(self, node.upper)
        nodes3, stp = visitMaybe(self, node.step)
        return nodes + nodes1 + nodes2 + nodes3, ast.Slice(l, h, stp)

    def visit_ExtSlice(self, node):
        nodes = []
        new_dims = []
        for dim in node.dims:
            nodes1, dim1 = self.visit(dim)
            nodes += nodes1
            new_dims.append(dim1)
        return nodes, ast.ExtSlice(new_dims)

    def visit_BinOp(self, node):
        nodes = []
        nodes1, l = self.visitNameOnly(node.left)
        nodes2, r = self.visitNameOnly(node.right)
        return nodes + nodes1 + nodes2, ast.BinOp(l, node.op, r)
    
    def visit_BoolOp(self, node):
        nodes = []
        new_vs = []
        for v in node.values:
            nodes1, new_v = self.visitNameOnly(v)
            nodes += nodes1
            new_vs.append(new_v)
        return nodes, ast.BoolOp(node.op, new_vs)

    def visit_UnaryOp(self, node):
        nodes, oper = self.visitOnly(node.operand, [ast.Constant])
        return nodes, ast.UnaryOp(node.op, oper)

    def visit_Compare(self, node):
        nodes = []
        new_coms = []
        nodes1, l = self.visitNameOnly(node.left)
        nodes += nodes1
        for com in node.comparators:
            nodes2, new_com = self.visitNameOnly(com)
            nodes += nodes2
            new_coms.append(new_com)
        return nodes, ast.Compare(l, node.ops, new_coms)

    def visit_Starred(self, node):
        nodes, new_v = self.visitNameOnly(node.value)
        return nodes, ast.Starred(new_v)

    def visit_Call(self, node):
        nodes = []
        nodes1, base = self.visit(node.func)
        nodes2, args = self.visit_arguments(node.args)
        nodes3, kws = self.visit_keywords(node.keywords, node.func)
        return nodes + nodes1 + nodes2 + nodes3, ast.Call(base, args, kws)


    def visit_Attribute(self, node, assigned = False):
        nodes = []
        nodes1, new_v = self.visitNameOnly(node.value)
        return nodes + nodes1, ast.Attribute(new_v, node.attr)

    def visit_arguments(self, args):
        nodes = []
        arg_names = []
        for i, arg in enumerate(args):
            nodes1, new_arg = self.visitOnly(arg, [ast.Starred])
            nodes += nodes1
            arg_names.append(new_arg)
        return nodes, arg_names

    def visit_keywords(self, keywords, func_node):
        nodes = []
        kws = []
        for keyword in keywords:
            nodes1, new_v = self.visitNameOnly(keyword.value)
            nodes += nodes1
            new_arg = self.scopeManager.get_mapped_arg(func_node, keyword.arg)
            kws.append(ast.keyword(arg = new_arg, value = new_v))
        return nodes, kws

    def visitNameOnly(self, node):
        nodes, newNode = self.visit(node)
        if type(newNode) != ast.Name:
            nodes1, new_var = self.visit_Name(ast.Name(self.FManager.get_new_var()), assigned=True)
            return nodes + nodes1 + [ast.Assign([new_var], newNode)], new_var
        return nodes, newNode

    def visitOnly(self, node, nodetypes):
        nodes, newNode = self.visit(node)
        if type(newNode) not in [ast.Name] + nodetypes:
            nodes1, new_var = self.visit_Name(ast.Name(self.FManager.get_new_var()), assigned=True)
            return nodes + nodes1 + [ast.Assign([new_var], newNode)], new_var
        return nodes, newNode

    def visit_Name(self, node, assigned = False):
        if not assigned and node.id not in self.scopeManager.defined_names:
            if not self.scopeManager.in_locals(node.id):
                nodes, new_name = self.visit_Name(ast.Name(self.FManager.get_new_var()), assigned=True)
                new_node = ast.Name(self.scopeManager.getName(node.id, assigned))
                nodes += [ast.Assign([new_name], ast.Call(ast.Name("global_wrapper"), [new_node], []))]
                return nodes, new_name
        return [], ast.Name(self.scopeManager.getName(node.id, assigned))

    def visit_Constant(self, node):
        return [], node
    
    def visit_List(self, node):
        nodes = []
        new_list = []
        for v in node.elts:
            newNodes, new_v = self.visitOnly(v, [ast.Constant])
            nodes += newNodes
            new_list.append(new_v)
        return nodes, ast.List(new_list)

    def visit_Tuple(self, node):
        nodes = []
        new_tuple = []
        for v in node.elts:
            newNodes, new_v = self.visitNameOnly(v)
            nodes += newNodes
            new_tuple.append(new_v)
        return nodes, ast.Tuple(new_tuple)

    def visit_Set(self, node):
        nodes = []
        new_set = []
        for v in node.elts:
            newNodes, new_v = self.visitOnly(v, [ast.Constant])
            nodes += newNodes
            new_set.append(new_v)
        return nodes, ast.Set(new_set)
    
    def visit_Dict(self, node):
        nodes = []
        new_keys = []
        new_values = []
        for v in node.keys:
            if v != None:
                newNodes, new_v = self.visitOnly(v, [ast.Constant])
                nodes += newNodes
                new_keys.append(new_v)
            else:
                new_keys.append(v)
        for v in node.values:
            newNodes, new_v = self.visitOnly(v, [ast.Constant])
            nodes += newNodes
            new_values.append(new_v)
        return nodes, ast.Dict(new_keys, new_values)

    def visit_FormattedValue(self, node):
        nodes, new_v = self.visitOnly(node.value, [ast.Constant])
        nodes1, new_f = [], node.format_spec
        if node.format_spec:
            nodes1, new_f = self.visit(node.format_spec)
        return nodes + nodes1, ast.FormattedValue(new_v, node.conversion, new_f)

    def visit_JoinedStr(self, node):
        nodes = []
        new_values = []
        for v in node.values:
            newNodes, new_v = self.visit(v)
            nodes += newNodes
            new_values.append(new_v)
        return nodes, ast.JoinedStr(new_values)
    
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

    def visit_NamedExpr(self, node):
        return [], node