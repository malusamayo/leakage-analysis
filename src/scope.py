import os, sys
import ast
from collections import defaultdict

class ScopeManager(object):
    def __init__(self) -> None:
        self.ctx = ['module']
        self.named_ctx = []

        self.name_map = {}
        self.name_nextid_map = {}
        self.arg_map = {}
        self.updated_in_ctx = defaultdict(set)
        self.defined_in_ctx = defaultdict(set)

        self.ctx_num = 0

    def get_tmp_new_ctx(self):
        return self.ctx + ["ctx" + str(self.ctx_num)]

    def get_new_ctx_num(self):
        old_var = self.ctx_num
        self.ctx_num += 1
        return "ctx" + str(old_var)
    
    def fill_updated(self, vars, _ctx):
        ctx_key = '.'.join(_ctx)
        defs = self.defined_in_ctx[ctx_key]
        updated = vars.difference(defs)
        self.updated_in_ctx[ctx_key].update(updated)

    def hasName(self, id, _ctx=None):
        if _ctx == None:
            ctx = [x for x in self.ctx]
        else:
            ctx = [x for x in _ctx]
        while ctx != []:
            key = '.'.join(ctx + [id])
            if key in self.name_map:
                return True
            ctx.pop()
        return False

    def getName(self, id, assigned=False, _ctx=None):
        if _ctx == None:
            ctx = [x for x in self.ctx]
        else:
            ctx = [x for x in _ctx]
        ctx_key = '.'.join(ctx)
        complete_key = '.'.join(ctx + [id])
        while True:
            # first appearance
            if ctx == []:
                self.name_map[complete_key] = id
                if assigned:
                    if id in self.name_nextid_map:
                        self.name_map[complete_key] = id + '$' + str(self.name_nextid_map[id])
                        self.name_nextid_map[id] += 1
                    self.defined_in_ctx[ctx_key].add(id)
                if id not in self.name_nextid_map:
                    self.name_nextid_map[id] = 0
                return self.name_map[complete_key].replace('$', '_')
            key = '.'.join(ctx + [id])
            # appeared before
            if key in self.name_map:
                # update when assigned
                if assigned:
                    # new locals
                    if key != complete_key:
                        self.name_map[complete_key] = self.name_map[key]
                        key = complete_key
                        self.updated_in_ctx[ctx_key].add(id)
                    # old_id = self.name_map[key]
                    self.name_map[key] = id + '$' + str(self.name_nextid_map[id])
                    self.name_nextid_map[id] += 1
                return self.name_map[key].replace('$', '_')
            ctx.pop()

    def enterBlock(self):
        self.ctx.append(self.get_new_ctx_num())

    def leaveBlock(self):
        self.ctx.pop()

    def enterNamedBlock(self, name):
        self.named_ctx.append(name)

    def leaveNamedBlock(self):
        self.named_ctx.pop()

    def resolve_upates(self, ctx1, ctx2, outer_ctx):
        updates1 = self.updated_in_ctx['.'.join(ctx1)]
        updates2 = self.updated_in_ctx['.'.join(ctx2)]
        up1set = updates1.difference(updates2)
        up2set = updates2.difference(updates1)
        conflicts = updates1.intersection(updates2)

        inits = []
        phi_calls = []

        for var_name in up1set:
            assert self.hasName(var_name, _ctx=outer_ctx)
            var_name_1 = ast.Name(self.getName(var_name, _ctx=ctx1))
            var_name_2 = ast.Name(self.getName(var_name, _ctx=outer_ctx))
            var_name_3 = ast.Name(self.getName(var_name, assigned=True, _ctx=outer_ctx))
            assign = ast.Assign([var_name_3], ast.Call(ast.Name("__phi__"), [var_name_1, var_name_2], []))
            # init = ast.Assign([var_name_1], var_name_2)
            # inits.append(init)
            phi_calls.append(assign)

        for var_name in up2set:
            assert self.hasName(var_name, _ctx=outer_ctx)
            var_name_1 = ast.Name(self.getName(var_name, _ctx=ctx2))
            var_name_2 = ast.Name(self.getName(var_name, _ctx=outer_ctx))
            var_name_3 = ast.Name(self.getName(var_name, assigned=True, _ctx=outer_ctx))
            assign = ast.Assign([var_name_3], ast.Call(ast.Name("__phi__"), [var_name_1, var_name_2], []))
            # init = ast.Assign([var_name_1], var_name_2)
            # inits.append(init)
            phi_calls.append(assign)
        
        for var_name in conflicts:
            assert self.hasName(var_name, _ctx=outer_ctx)
            # var_name_0 = ast.Name(self.getName(var_name, _ctx=outer_ctx))
            var_name_1 = ast.Name(self.getName(var_name, _ctx=ctx1))
            var_name_2 = ast.Name(self.getName(var_name, _ctx=ctx2))
            var_name_3 = ast.Name(self.getName(var_name, assigned=True, _ctx=outer_ctx))
            assign = ast.Assign([var_name_3], ast.Call(ast.Name("__phi__"), [var_name_1, var_name_2], []))
            # init1 = ast.Assign([var_name_1], var_name_0)
            # init2 = ast.Assign([var_name_2], var_name_0)
            # inits += [init1, init2]
            phi_calls.append(assign)

        defs1 = self.defined_in_ctx['.'.join(ctx1)]
        defs2 = self.defined_in_ctx['.'.join(ctx2)]

        def1set = defs1.difference(defs2)
        def2set = defs2.difference(defs1)
        conflicts = defs1.intersection(defs2)

        for var_name in def1set:
            self.name_map['.'.join(outer_ctx + [var_name])] = self.name_map['.'.join(ctx1 + [var_name])]

        for var_name in def2set:
            self.name_map['.'.join(outer_ctx + [var_name])] = self.name_map['.'.join(ctx2 + [var_name])]
        
        for var_name in conflicts:
            self.name_map['.'.join(outer_ctx + [var_name])] = "placeholder"
            var_name_1 = ast.Name(self.getName(var_name, _ctx=ctx1))
            var_name_2 = ast.Name(self.getName(var_name, _ctx=ctx2))
            var_name_3 = ast.Name(self.getName(var_name, assigned=True, _ctx=outer_ctx))
            assign = ast.Assign([var_name_3], ast.Call(ast.Name("__phi__"), [var_name_1, var_name_2], []))
            phi_calls.append(assign)

        self.defined_in_ctx[".".join(outer_ctx)].update(defs1.union(defs2))
        self.fill_updated(updates1.union(updates2), outer_ctx)
        
        return inits, phi_calls
    
    def build_arg_map(self, args):
        local_map = {}
        # "self"/first arg should be handled differently [TODO]
        for arg in args.posonlyargs:
            new_name = self.getName(arg.arg, assigned=True, _ctx = self.get_tmp_new_ctx())
            local_map[arg.arg] = new_name
            arg.arg = new_name
        for arg in args.args:
            new_name = self.getName(arg.arg, assigned=True, _ctx = self.get_tmp_new_ctx())
            local_map[arg.arg] = new_name
            arg.arg = new_name
        for arg in args.kwonlyargs:
            new_name = self.getName(arg.arg, assigned=True, _ctx = self.get_tmp_new_ctx())
            local_map[arg.arg] = new_name
            arg.arg = new_name
        if args.vararg:
            arg = args.vararg
            new_name = self.getName(arg.arg, assigned=True, _ctx = self.get_tmp_new_ctx())
            local_map[arg.arg] = new_name
            arg.arg = new_name
        if args.kwarg:
            arg = args.vararg
            new_name = self.getName(arg.arg, assigned=True, _ctx = self.get_tmp_new_ctx())
            local_map[arg.arg] = new_name
            arg.arg = new_name
        self.arg_map['.'.join(self.named_ctx)] = local_map


    def get_mapped_arg(self, func_node, arg):
        # do not consider class member now [TODO]
        if type(func_node) == ast.Name and func_node.id in self.arg_map:
            new_arg = self.arg_map[func_node.id][arg]
            return new_arg
        return arg
