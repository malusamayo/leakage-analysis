import os, sys
import ast
import astunparse
import json
from factgen import FactManager


class CodeTransformer(ast.NodeTransformer):
    def __init__(self) -> None:
        super().__init__()
        self.FManager = FactManager()
        self.name_map = {}

    def generic_visit(self, node):
        rets = ast.NodeTransformer.generic_visit(self, node)
        # if type(rets) != tuple:
        #     return rets #, ast.Pass()
        # else:
        return rets

    def visit_Expr(self, node):
        rets = self.generic_visit(node)
        if len(rets.value) == 2:
            assert type(rets.value[0]) == list
            return rets.value[0] + [ast.Expr(rets.value[1])]
        return rets

    def handle_assign_value(self, target, value):
        assert(type(target) == ast.Name)
        nodes = []

        if type(value) == ast.Name:
            nodes1, target_name = self.visit_Name(target, assigned = True)
            _, value_name = self.visit_Name(value)
            nodes = nodes + nodes1 + [ast.Assign([ast.Name(target_name)], ast.Name(value_name))]
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

    def handle_single_assign(self, target, value):
        nodes = []
        if type(target) == ast.Name:
            nodes += self.handle_assign_value(target, value)
        elif type(target) == ast.Attribute:
            nodes1, attr = self.visit_Attribute(target) #, assigned=True)
            from_new_var = self.FManager.get_new_var()
            nodes2 = self.visit_Assign(ast.Assign([ast.Name(from_new_var)], value))
            nodes = nodes + nodes1 + nodes2 + [ast.Assign([attr], ast.Name(from_new_var))]
        elif type(target) == ast.Subscript:
            nodes1, subscript = self.visit_Subscript(target)
            from_new_var = self.FManager.get_new_var()
            nodes2 = self.visit_Assign(ast.Assign([ast.Name(from_new_var)], value))
            nodes = nodes + nodes1 + nodes2 + [ast.Assign([subscript], ast.Name(from_new_var))]
        elif type(target) == ast.Tuple:
            if type(value) == ast.Tuple and len(target.elts) == len(value.elts):
                new_vars = []
                for v in value.elts:
                    to_new_var = self.FManager.get_new_var()
                    new_vars.append(to_new_var)
                    nodes += self.visit_Assign(ast.Assign([ast.Name(to_new_var)], v))
                for v_name, t in zip(new_vars, target.elts):
                    nodes += self.visit_Assign(ast.Assign([t], ast.Name(v_name)))
            elif type(value) == ast.Call:
                new_vars = [self.FManager.get_new_var() for v in target.elts]
                nodes1, call = self.visit_Call(value)
                nodes = nodes + nodes1 + [ast.Assign([ast.Tuple([ast.Name(v ) for v in new_vars])], call)]
                for v, var in zip(target.elts, new_vars):
                    nodes += self.visit_Assign(ast.Assign([v], ast.Name(var)))
            elif type(value) == ast.Name:
                for i, t in enumerate(target.elts):
                    nodes += self.visit_Assign(ast.Assign([t], ast.Subscript(value, ast.Index(ast.Constant(i, "")))))
            else:
                assert 0
        else:
            print("Unkown target type! " + str(type(target)))
            assert 0
        return nodes

    def visit_AnnAssign(self, node):
        nodes = self.handle_single_assign(node.target, node.value)
        if type(node.target) == ast.Name:
            assert node.target.id == nodes[-1].targets[0].id
            nodes[-1] = ast.AnnAssign(node.target, node.annotation, nodes[-1].value, simple = node.simple)
        return nodes

    def visit_Assign(self,node):
        # print('Node type: Assign and fields: ', node.targets)
        nodes = []
        for target in node.targets:
            nodes += self.handle_single_assign(target, node.value)
        return nodes
    
    def visit_Subscript(self, node, assigned = False):
        nodes = []
        new_var = ""
        if type(node.value) == ast.Name:
            nodes1, new_var = self.visit_Name(node.value, assigned=assigned)
            nodes += nodes1
        else:
            new_var = self.FManager.get_new_var()
            nodes += self.visit_Assign(ast.Assign([ast.Name(new_var)], node.value))
        if type(node.slice) == ast.Slice:
            nodes2, slice = self.visit_Slice(node.slice)
        elif type(node.slice) == ast.Index:
            nodes2, slice = self.visit_Index(node.slice)
        else:
            assert 0
        return nodes + nodes2, ast.Subscript(ast.Name(new_var), slice)

    def visit_Index(self, node):
        nodes = []
        new_var = self.FManager.get_new_var()
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
            to_new_var = self.FManager.get_new_var()
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
            new_var = self.FManager.get_new_var()
            nodes += self.visit_Assign(ast.Assign([ast.Name(new_var)],arg))
            arg_names.append(ast.Name(new_var))
        return nodes, arg_names

    def visit_keywords(self, keywords):
        nodes = []
        kws = []
        for keyword in keywords:
            new_var = self.FManager.get_new_var()
            nodes += self.visit_Assign(ast.Assign([ast.Name(new_var)], keyword.value))
            kws.append(ast.keyword(arg = keyword.arg, value = ast.Name(new_var)))
        return nodes, kws

    def visit_Num(self, node):
        return node

    def visit_Name(self, node, assigned = False):
        nodes = []
        if node.id not in self.name_map:
            self.name_map[node.id] = node.id
            return nodes, self.name_map[node.id]
        if assigned:
            old_id = self.name_map[node.id]
            try:
                num = int(old_id.split("$")[-1])
                self.name_map[node.id] = node.id + '$' + str(num + 1)
            except ValueError:
                self.name_map[node.id] = node.id + '$0'
            # nodes += self.visit_Assign(ast.Assign([ast.Name(self.name_map[node.id])], ast.Name(old_id)))
        return nodes, self.name_map[node.id].replace('$', '_')

    def visit_Str(self, node):
        return node
