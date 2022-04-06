import os, sys
import ast
import astunparse
import json
import shutil
import argparse
from .global_collector import GlobalCollector
from . import factgen
from .irgen import CodeTransformer
from .render import to_html

class Config(object):
    def __init__(self, inference_path: str, output_flag: bool) -> None:
        self.inference_path = inference_path
        self.output_flag = output_flag
config = Config("../pyright-m/packages/pyright/index.js", False)

def remove_files(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def load_input(input_path):
    with open(input_path) as f:
        code = f.read()
    try:
        tree = ast.parse(code)
    except:
        print("Failed to parse " + input_path)
        exit(37)
    return tree

def ir_transform(tree, ir_path):
    ignored_vars = GlobalCollector().visit(tree)
    v = CodeTransformer(ignored_vars)
    new_tree = v.visit(tree)
    new_code = astunparse.unparse(new_tree)
    # print(new_code)
    with open(ir_path, "w") as f:
        f.write(new_code)
    return new_tree

def infer_types(ir_path):
    # Call type inference engine here
    os.system(f"node {config.inference_path} {ir_path} --lib")

def generate_lineno_mapping(tree1, tree2):
    lineno_map = {}
    if len(tree1.body) != len(tree2.body):
        return lineno_map
    def add_to_mapping(body1, body2):
        for stmt1, stmt2 in zip(body1, body2):
            if hasattr(stmt1, 'lineno') and hasattr(stmt2, 'lineno'):
                lineno_map[str(stmt2.lineno)] = str(stmt1.lineno)
            if hasattr(stmt1, 'body') and hasattr(stmt2, 'body'):
                add_to_mapping(stmt1.body, stmt2.body)
            if hasattr(stmt1, 'orelse') and hasattr(stmt2, 'orelse'):
                add_to_mapping(stmt1.orelse, stmt2.orelse)
            if hasattr(stmt1, 'handlers') and hasattr(stmt2, 'handlers'):
                add_to_mapping(stmt1.orelse, stmt2.orelse)
            if hasattr(stmt1, 'finalbody') and hasattr(stmt2, 'finalbody'):
                add_to_mapping(stmt1.orelse, stmt2.orelse)
                
    add_to_mapping(tree1.body, tree2.body)
    return lineno_map


def generate_facts(tree, json_path, fact_path):
    f = factgen.FactGenerator(json_path)
    f.visit(tree)


    if not os.path.exists(fact_path):
        os.makedirs(fact_path)
    else:
        remove_files(fact_path)

    for fact_name, fact_list in f.FManager.datalog_facts.items():
        with open(os.path.join(fact_path, fact_name + ".facts"), "w") as f:
            facts = ["\t".join(t) for t in fact_list]
            f.writelines("\n".join(facts))

def datalog_analysis(fact_path):
    os.system(f"souffle ./src/main.dl -F {fact_path} -D {fact_path}")

def main(input_path):
    ir_path = input_path +".ir.py"
    json_path = input_path + ".json"
    fact_path = input_path[:-3] + "-fact"
    html_path = input_path[:-3] + ".html"

    tree = load_input(input_path)
    tree = ir_transform(tree, ir_path)
    infer_types(ir_path)
    newtree = load_input(ir_path)
    if config.output_flag:
        lineno_map = generate_lineno_mapping(tree, newtree)
    generate_facts(newtree, json_path, fact_path)
    datalog_analysis(fact_path)
    if config.output_flag:
        print("Converting notebooks to html...")
        to_html(input_path, fact_path, html_path, lineno_map)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run analysis for a single file')
    parser.add_argument('file', help='the python file to be analyzed')
    parser.add_argument('-o', '--output-flag', help='output html file', action="store_true")
    args = parser.parse_args()
    config = Config("../pyright-m/packages/pyright/index.js", args.output_flag)
    main(os.path.abspath(sys.argv[1]))
