import os, sys
import ast
import astunparse
import json
import factgen
from factgen import FManager
from irgen import CodeTransformer



def load_input(input_path):
    with open(input_path) as f:
        code = f.read()
    tree = ast.parse(code)
    return tree

def ir_transform(tree, ir_path):
    v = CodeTransformer()
    new_tree = v.visit(tree)
    new_code = astunparse.unparse(new_tree)
    print(new_code)
    with open(ir_path, "w") as f:
        f.write(new_code)
    return new_tree

def infer_types(ir_path, json_path):
    # Call type inference engine here
    os.system("node ~/Projects/pyright/packages/pyright/index.js " + ir_path)
    factgen.load_type_map(json_path)

def generate_facts(tree, fact_path):
    f = factgen.FactGenerator()
    f.visit(tree)

    if not os.path.exists(fact_path):
        os.makedirs(fact_path)

    for fact_name, fact_list in FManager.datalog_facts.items():
        with open(os.path.join(fact_path, fact_name + ".facts"), "w") as f:
            facts = ["\t".join(t) for t in fact_list]
            f.writelines("\n".join(facts))

def datalog_analysis(fact_path):
    os.system(f"souffle src/main.dl -F {fact_path} -D {fact_path}")

def main(input_path):
    ir_path = input_path +".ir.py"
    json_path = input_path + ".json"
    fact_path = input_path[:-3] + "-fact"

    tree = load_input(input_path)
    newtree = ir_transform(tree, ir_path)
    infer_types(ir_path, json_path)
    generate_facts(newtree, fact_path)
    datalog_analysis(fact_path)

if __name__ == "__main__":
    main(os.path.abspath(sys.argv[1]))
