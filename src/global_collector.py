import ast
from collections import defaultdict
from .scope import ScopeManager

class GlobalCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.scopeManager = ScopeManager()
        self.globals = set() # set of variables that should not be renamed
    
    def visit(self, node):
        """Visit a node."""
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        return ast.NodeVisitor.generic_visit(self, node)
    
    def visit_Module(self, node):
        self.generic_visit(node)
        return self.globals

    def visit_alias(self, node):
        self.scopeManager.update_locals(node.name)
        if node.asname:
            self.scopeManager.update_locals(node.asname)
        return node

    def visit_Global(self, node):
        self.scopeManager.update_globals(node.names)
        return node

    def visit_Nonlocal(self, node):
        self.scopeManager.update_globals(node.names)
        return node

    def visit_ClassDef(self, node):
        self.scopeManager.enterNamedBlock(node.name)
        ret = self.generic_visit(node)
        self.scopeManager.leaveNamedBlock()
        return ret

    def visit_FunctionDef(self, node):
        self.scopeManager.enterNamedBlock(node.name)
        for i, arg in enumerate(node.args.args):
            self.scopeManager.update_locals(arg.arg)
        ret = self.generic_visit(node)
        self.scopeManager.leaveNamedBlock()
        return ret

    def visit_AsyncFunctionDef(self, node):
        return self.visit_FunctionDef(node)
    
    def visit_For(self, node):
        self.handle_single_assign(node.target)
        ret = self.generic_visit(node)
        return ret
    
    def visit_withitem(self, node):
        if node.optional_vars:
            self.handle_single_assign(node.optional_vars)
        return node

    def handle_name_assigned(self, name):
        if self.scopeManager.in_globals(name):
            self.globals.add(name)
        else:
            self.scopeManager.update_locals(name)
    
    def handle_name_updated(self, name):
        if not self.scopeManager.in_locals(name):
            self.globals.add(name)

    def handle_single_assign(self, target):
        if type(target) == ast.Name:
            name = self.visit(target)
            self.handle_name_assigned(name)
        elif type(target) == ast.Starred:
            name = self.visit(target)
            if type(target.value) == ast.Name:
                self.handle_name_assigned(name)
            else:
                self.handle_name_updated(name)
        elif type(target) in [ast.Attribute, ast.Subscript]:
            name = self.visit(target)
            self.handle_name_updated(name)
        elif type(target) in [ast.Tuple, ast.List]:
            for v in target.elts:
                name = self.visit(v)
                if type(v) == ast.Name:
                    self.handle_name_assigned(name)
                else:
                    self.handle_name_updated(name)
        else:
            assert 0, "Unkown target type! " + str(type(target))
        return

    def visit_AnnAssign(self, node):
        self.handle_single_assign(node.target)
        return node
    
    def visit_AugAssign(self, node):
        return self.visit_Assign(ast.Assign([node.target], ast.BinOp(node.target, node.op, node.value)))

    def visit_Assign(self,node):
        for target in node.targets:
            self.handle_single_assign(target)
        return node
    
    def visit_Call(self, node):
        name = self.visit(node.func)
        if name and type(node.func) != ast.Name:
            self.handle_name_updated(name)

    def visit_Subscript(self, node):
        return self.visit(node.value)

    def visit_Attribute(self, node):
        return self.visit(node.value)

    def visit_Starred(self, node):
        return self.visit(node.value)

    def visit_Name(self, node):
        return node.id