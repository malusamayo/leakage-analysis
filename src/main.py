import os, sys
import ast
import astunparse
import json
import shutil
import argparse
import time
import traceback
from .global_collector import GlobalCollector
from . import factgen
from .irgen import CodeTransformer
from .render import to_html
from .config import configs

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


def time_decorator(func):
    def wrapper_function(*args, **kwargs):
        try:
            st = time.time()
            ret = func(*args,  **kwargs)
            ed = time.time()
            return ret, ed - st
        except Exception as e:
            print("Failed!")
            print(e)
            print(traceback.format_exc())
            return None, -1
    return wrapper_function

@time_decorator
def load_input(input_path):
    with open(input_path) as f:
        code = f.read()
        tree = ast.parse(code)
    return tree

@time_decorator
def ir_transform(tree, ir_path):
    ignored_vars = GlobalCollector().visit(tree)
    v = CodeTransformer(ignored_vars)
    new_tree = v.visit(tree)
    new_code = astunparse.unparse(new_tree)
    # print(new_code)
    with open(ir_path, "w") as f:
        f.write(new_code)
    return new_tree

@time_decorator
def infer_types(ir_path):
    # Call type inference engine here
    os.system(f"timeout 5m node {configs.inference_path} {ir_path} --lib")

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

@time_decorator
def generate_facts(tree, json_path, fact_path):
    f = factgen.FactGenerator(json_path)
    f.visit(tree)

    for fact_name, fact_list in f.FManager.datalog_facts.items():
        with open(os.path.join(fact_path, fact_name + ".facts"), "w") as f:
            facts = ["\t".join(t) for t in fact_list]
            f.writelines("\n".join(facts))

@time_decorator
def datalog_analysis(fact_path):
    ret = os.system(f"timeout 5m souffle ./src/main.dl -F {fact_path} -D {fact_path}")
    if ret != 0:
        raise TimeoutError

def main(input_path):
    ir_path = input_path +".ir.py"
    json_path = input_path + ".json"
    fact_path = input_path[:-3] + "-fact"
    html_path = input_path[:-3] + ".html"
    t = [None]*6

    tree, t[0] = load_input(input_path)
    if t[0] == -1:
        print("Failed to parse: " + input_path)
        return "Failed to parse"
    
    tree, t[1] = ir_transform(tree, ir_path)
    if t[1]== -1:
        print("Failed to generate IR: " + input_path)
        return "Failed to generate IR"
    
    _, t[2] = infer_types(ir_path)
    if not os.path.exists(json_path):
        print("Failed to infer types: " + input_path)
        return "Failed to infer types" 

    
    newtree, t[3] = load_input(ir_path)
    if t[3] == -1:
        print("Failed to parse transformed file: " + input_path)
        return "Failed to parse transformed file"

    # clean facts
    if not os.path.exists(fact_path):
        os.makedirs(fact_path)
    else:
        remove_files(fact_path)

    if configs.output_flag:
        lineno_map = generate_lineno_mapping(tree, newtree)
        with open(os.path.join(fact_path, "LinenoMapping.facts"), "w") as f:
            facts = [a + "\t" + b for a, b in lineno_map.items()]
            f.writelines("\n".join(facts))
    
    _, t[4] = generate_facts(newtree, json_path, fact_path)
    if t[4] == -1:
        print("Failed to generate facts: " + input_path)
        return "Failed to generate facts" 
    
    _, t[5] = datalog_analysis(fact_path)
    if t[5] == -1:
        print("Failed to analyze: " + input_path)
        return "Failed to analyze" 

    if configs.output_flag:
        print("Converting notebooks to html...")
        try:
            to_html(input_path, fact_path, html_path, lineno_map)
        except:
            print("Conversion failed!")
    
    print("Success!\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t".format(t[0]+t[1]+t[3]+t[4], t[2], t[5], sum(t)))
    return t

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run analysis for a single file')
    parser.add_argument('file', help='the python file to be analyzed')
    parser.add_argument('-o', '--output-flag', help='output html file', action="store_true")
    args = parser.parse_args()
    configs.output_flag = args.output_flag
    main(os.path.abspath(sys.argv[1]))
