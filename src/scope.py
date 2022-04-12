import os, sys
import ast
from collections import defaultdict

class ScopeManager(object):
    def __init__(self, ignored_vars=set()) -> None:
        self.ctx = ['module']
        self.named_ctx = []

        self.name_map = {}
        self.name_nextid_map = {}
        self.arg_map = {}
        self.updated_in_ctx = defaultdict(set)
        self.defined_in_ctx = defaultdict(set)

        self.defined_names = {
            'abs', 'all', 'any', 'apply', 'basestring', 'bin', 'bool', 'buffer', 'bytearray', 'bytes', 
            'callable', 'chr', 'classmethod', 'cmp', 'coerce', 'compile', 'complex', 'copyright', 'credits', 
            'delattr', 'dict', 'dir', 'divmod', 'enumerate', 'eval', 'execfile', 'exit', 'file', 'filter', 
            'float', 'format', 'frozenset', 'getattr', 'globals', 'hasattr', 'hash', 'help', 'hex', 'id', 
            'input', 'int', 'intern', 'isinstance', 'issubclass', 'iter', 'len', 'license', 'list', 'locals', 
            'long', 'map', 'max', 'memoryview', 'min', 'next', 'object', 'oct', 'open', 'ord', 'pow', 'print', 
            'property', 'quit', 'range', 'raw_input', 'reduce', 'reload', 'repr', 'reversed', 'round', 'set', 
            'setattr', 'slice', 'sorted', 'staticmethod', 'str', 'sum', 'super', 'tuple', 'type', 'unichr', 'unicode', 'vars', 'xrange', 'zip',
            'get_ipython', 'display'}
        self.locals = defaultdict(set)
        self.globalOrNonloacls = defaultdict(set) # specified with global/nonlocal keywords
        self.ignored_vars = ignored_vars

        self.ctx_num = 0

    def get_tmp_new_ctx(self):
        return self.ctx + ["ctx" + str(self.ctx_num)]

    def get_new_ctx_num(self):
        old_var = self.ctx_num
        self.ctx_num += 1
        return "ctx" + str(old_var)
    
    def update_globals(self, names):
        for name in names:
            self.globalOrNonloacls[self.get_cur_sig()].add(name)

    def in_globals(self, name):
        return name in self.globalOrNonloacls[self.get_cur_sig()]

    def update_locals(self, name):
        self.locals[self.get_cur_sig()].add(name)
    
    def in_locals(self, name):
        return name in self.locals[self.get_cur_sig()]
    
    def get_cur_sig(self):
        # [TODO] do not consider inner method now
        if self.named_ctx and self.named_ctx[-1] == "__init__":
            return '.'.join(self.named_ctx[:-1])
        return '.'.join(self.named_ctx)

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
        if id in self.ignored_vars:
            return id
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
                    self.locals[self.get_cur_sig()].add(id)
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
                    self.locals[self.get_cur_sig()].add(id)
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
        self.defined_names.add(name)

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
            if self.hasName(var_name, _ctx=outer_ctx):
                var_name_1 = ast.Name(self.getName(var_name, _ctx=ctx1))
                var_name_2 = ast.Name(self.getName(var_name, _ctx=outer_ctx))
                var_name_3 = ast.Name(self.getName(var_name, assigned=True, _ctx=outer_ctx))
                assign = ast.Assign([var_name_3], ast.Call(ast.Name("__phi__"), [var_name_1, var_name_2], []))
                # init = ast.Assign([var_name_1], var_name_2)
                # inits.append(init)
                phi_calls.append(assign)

        for var_name in up2set:
            if self.hasName(var_name, _ctx=outer_ctx):
                var_name_1 = ast.Name(self.getName(var_name, _ctx=ctx2))
                var_name_2 = ast.Name(self.getName(var_name, _ctx=outer_ctx))
                var_name_3 = ast.Name(self.getName(var_name, assigned=True, _ctx=outer_ctx))
                assign = ast.Assign([var_name_3], ast.Call(ast.Name("__phi__"), [var_name_1, var_name_2], []))
                # init = ast.Assign([var_name_1], var_name_2)
                # inits.append(init)
                phi_calls.append(assign)
        
        for var_name in conflicts:
            if self.hasName(var_name, _ctx=outer_ctx):
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
            arg = args.kwarg
            new_name = self.getName(arg.arg, assigned=True, _ctx = self.get_tmp_new_ctx())
            local_map[arg.arg] = new_name
            local_map["$kwarg"] = True
            arg.arg = new_name
        self.arg_map[self.get_cur_sig()] = local_map


    def get_mapped_arg(self, func_node, arg):
        # do not consider class member now [TODO]
        if type(func_node) == ast.Name and func_node.id in self.arg_map:
            if arg in self.arg_map[func_node.id]:
                new_arg = self.arg_map[func_node.id][arg]
            elif "$kwarg" in self.arg_map[func_node.id]:
                new_arg = arg
            else:
                new_arg = arg
            return new_arg
        return arg
