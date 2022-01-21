import os, sys
import ast
import astunparse
import json
from .factgen import FactManager


class CodeTransformer(ast.NodeTransformer):
    def __init__(self) -> None:
        super().__init__()
        self.FManager = FactManager()
        self.name_map = {}

    def visit(self, node):
        """Visit a node."""
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        rets = ast.NodeTransformer.generic_visit(self, node)
        if type(node) not in [ast.Module, ast.FunctionDef, ast.Import, ast.ImportFrom, ast.alias, ast.Expr]:
            print(type(node))
        # if type(rets) != tuple:
        #     return rets #, ast.Pass()
        # else:
        return rets

    def visit_ClassDef(self, node):
        node.body = self.visit_Body(node.body)
        return node

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

    def visit_For(self, node):
        nodes, new_iter = self.visitNameOnly(node.iter)
        node.iter = new_iter
        node.body = self.visit_Body(node.body)
        node.orelse = self.visit_Body(node.orelse)
        return nodes, node

    def visit_While(self, node):
        nodes, new_test = self.visitNameOnly(node.test)
        node.iter = new_test
        node.body = self.visit_Body(node.body)
        node.orelse = self.visit_Body(node.orelse)
        return nodes, node

    def visit_If(self, node):
        nodes, new_test = self.visitNameOnly(node.test)
        node.test = new_test
        node.body = self.visit_Body(node.body)
        node.orelse = self.visit_Body(node.orelse)
        return nodes, node

    def visit_IfExp(self, node):
        nodes, new_test = self.visitNameOnly(node.test)
        nodes1, new_body = self.visitNameOnly(node.body)
        nodes2, new_orelse = self.visitNameOnly(node.orelse)
        return nodes + nodes1 + nodes2, ast.IfExp(new_test, new_body, new_orelse)

    def visit_Return(self, node):
        nodes1, newValue = self.visitNameAndTupleOnly(node.value)
        node.value = newValue
        return nodes1 + [node]
    
    def visit_Yield(self, node):
        nodes1, newValue = self.visitNameAndTupleOnly(node.value)
        node.value = newValue
        return nodes1 + [node]

    def visit_Delete(self, node):
        nodes = []
        new_vars = []
        for t in node.targets:
            nodes1, new_v = self.visitNameOnly(t)
            new_vars.append(new_v)
            nodes += nodes1
        return nodes, ast.Delete(new_vars)

    def visit_Expr(self, node):
        rets = self.generic_visit(node)
        if len(rets.value) == 2:
            assert type(rets.value[0]) == list
            return rets.value[0] + [ast.Expr(rets.value[1])]
        return rets

    def handle_assign_value(self, target, value):
        assert(type(target) == ast.Name)
        nodes = []
        if type(value) in [ast.Attribute, ast.Name, ast.Call, ast.Constant, ast.Subscript, ast.List, ast.Tuple, ast.Set, ast.Dict, ast.BinOp, ast.UnaryOp, ast.Compare, ast.ListComp]:
            nodes, new_node = self.visit(value)
            nodes1, target = self.visit_Name(target, assigned = True)
            nodes = nodes + nodes1 + [ast.Assign([target], new_node)]
        elif type(value) == ast.Index:
            nodes += self.handle_assign_value(target, value.value)
        else:
            assert 0, "Unkown source type! " + str(type(value))
        return nodes

    def handle_single_assign(self, target, value):
        nodes = []
        if type(target) == ast.Name:
            nodes += self.handle_assign_value(target, value)
        elif type(target) in [ast.Attribute, ast.Subscript]:
            nodes1, new_target = self.visit(target)
            nodes2, new_value = self.visitNameOnly(value)
            nodes = nodes + nodes1 + nodes2 + [ast.Assign([new_target], new_value)]
        elif type(target) == ast.Tuple:
            if type(value) == ast.Tuple and len(target.elts) == len(value.elts):
                new_vars = []
                for v in value.elts:
                    nodes1, new_v = self.visitNameOnly(v)
                    new_vars.append(new_v)
                    nodes += nodes1
                for v, t in zip(new_vars, target.elts):
                    nodes += self.visit_Assign(ast.Assign([t], v))
            elif type(value) == ast.Call:
                new_vars = [self.FManager.get_new_var() for v in target.elts]
                nodes1, call = self.visit_Call(value)
                nodes = nodes + nodes1 + [ast.Assign([ast.Tuple([ast.Name(v) for v in new_vars])], call)]
                for v, var in zip(target.elts, new_vars):
                    nodes += self.visit_Assign(ast.Assign([v], ast.Name(var)))
            elif type(value) == ast.Name:
                for i, t in enumerate(target.elts):
                    nodes += self.visit_Assign(ast.Assign([t], ast.Subscript(value, ast.Index(ast.Constant(i, "")))))
            else:
                assert 0
        else:
            assert 0, "Unkown target type! " + str(type(target))
        return nodes

    def visit_AnnAssign(self, node):
        nodes = self.handle_single_assign(node.target, node.value)
        if type(node.target) == ast.Name:
            assert node.target.id == nodes[-1].targets[0].id
            nodes[-1] = ast.AnnAssign(node.target, node.annotation, nodes[-1].value, simple = node.simple)
        return nodes
    
    def visit_AugAssign(self, node):
        return self.visit_Assign(ast.Assign([node.target], ast.BinOp(node.target, node.op, node.value)))

    def visit_Assign(self,node):
        nodes = []
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
        nodes, oper = self.visitNameAndConsOnly(node.operand)
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
        nodes3, kws = self.visit_keywords(node.keywords)
        return nodes + nodes1 + nodes2 + nodes3, ast.Call(base, args, kws)


    def visit_Attribute(self, node, assigned = False):
        nodes = []
        nodes1, new_v = self.visitNameOnly(node.value)
        return nodes + nodes1, ast.Attribute(new_v, node.attr)

    def visit_arguments(self, args):
        # arg list in function definition
        if type(args) == ast.arguments:
            return args
        nodes = []
        arg_names = []
        for i, arg in enumerate(args):
            nodes1, new_arg = self.visitNameOnly(arg)
            nodes += nodes1
            arg_names.append(new_arg)
        return nodes, arg_names

    def visit_keywords(self, keywords):
        nodes = []
        kws = []
        for keyword in keywords:
            nodes1, new_v = self.visitNameOnly(keyword.value)
            nodes += nodes1
            kws.append(ast.keyword(arg = keyword.arg, value = new_v))
        return nodes, kws

    def visitNameOnly(self, node):
        nodes, newNode = self.visit(node)
        if type(newNode) != ast.Name:
            new_var = ast.Name(self.FManager.get_new_var())
            return nodes + [ast.Assign([new_var], newNode)], new_var
        return nodes, newNode

    def visitNameAndConsOnly(self, node):
        nodes, newNode = self.visit(node)
        if type(newNode) not in [ast.Name, ast.Constant]:
            new_var = ast.Name(self.FManager.get_new_var())
            return nodes + [ast.Assign([new_var], newNode)], new_var
        return nodes, newNode
    
    def visitNameAndTupleOnly(self, node):
        nodes, newNode = self.visit(node)
        if type(newNode) not in [ast.Name, ast.Tuple]:
            new_var = ast.Name(self.FManager.get_new_var())
            return nodes + [ast.Assign([new_var], newNode)], new_var
        return nodes, newNode

    def visit_Name(self, node, assigned = False):
        nodes = []
        if node.id not in self.name_map:
            self.name_map[node.id] = node.id
            return nodes, ast.Name(self.name_map[node.id])
        if assigned:
            old_id = self.name_map[node.id]
            try:
                num = int(old_id.split("$")[-1])
                self.name_map[node.id] = node.id + '$' + str(num + 1)
            except ValueError:
                self.name_map[node.id] = node.id + '$0'
            # nodes += self.visit_Assign(ast.Assign([ast.Name(self.name_map[node.id])], ast.Name(old_id)))
        return nodes, ast.Name(self.name_map[node.id].replace('$', '_'))

    def visit_Constant(self, node):
        return [], node
    
    def visit_List(self, node):
        nodes = []
        new_list = []
        for v in node.elts:
            newNodes, new_v = self.visitNameAndConsOnly(v)
            nodes += newNodes
            new_list.append(new_v)
        return nodes, ast.List(new_list)

    def visit_Tuple(self, node):
        nodes = []
        new_tuple = []
        for v in node.elts:
            newNodes, new_v = self.visitNameAndConsOnly(v)
            nodes += newNodes
            new_tuple.append(new_v)
        return nodes, ast.Tuple(new_tuple)

    def visit_Set(self, node):
        nodes = []
        new_set = []
        for v in node.elts:
            newNodes, new_v = self.visitNameAndConsOnly(v)
            nodes += newNodes
            new_set.append(new_v)
        return nodes, ast.Set(new_set)
    
    def visit_Dict(self, node):
        nodes = []
        new_keys = []
        new_values = []
        for v in node.keys:
            newNodes, new_v = self.visitNameAndConsOnly(v)
            nodes += newNodes
            new_keys.append(new_v)
        for v in node.values:
            newNodes, new_v = self.visitNameAndConsOnly(v)
            nodes += newNodes
            new_values.append(new_v)
        return nodes, ast.Dict(new_keys, new_values)

    def visit_Lambda(self, node):
        nodes, ret = self.visitNameOnly(node.body)
        nodes.append(ast.Return(ret))
        func_name = self.FManager.get_new_func()
        return [ast.FunctionDef(func_name, node.args, nodes, [])], ast.Name(func_name)

    # keep exprs below unchanged (for now)
    def visit_ListComp(self, node):
        return [], node

    def visit_SetComp(self, node):
        return [], node

    def visit_DictComp(self, node):
        return [], node

    def visit_GeneratorExp(self, node):
        return [], node
