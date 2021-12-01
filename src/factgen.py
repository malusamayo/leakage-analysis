import os, sys
import ast
import astunparse
import json

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
            "Invoke": [],
            "CallGraphEdge": [],
            "ActualParam": [],
            "ActualKeyParam": [], 
            "FormalParam": [],
            "FormalKeyParam": [],
            "ActualReturn": [],
            "FormalReturn": [],
            "MethodUpdate": [],
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

    def get_new_heap(self):
        old_heap = self.heap_num
        self.heap_num += 1
        return "$heap" + str(old_heap)


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
            FManager.add_fact("ActualReturn", (cur_invo, target_name))
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
                FManager.add_fact("StoreIndex", (target.value.id, target.slice.value.id, node.value.id))
            else:
                print("Unkown target type! " + str(type(target)))
                assert 0

        return node
    

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

def load_type_map(json_path):
    with open(json_path) as f:
        global type_map
        type_map = json.load(f) 

global FManager, type_map
FManager = FactManager()