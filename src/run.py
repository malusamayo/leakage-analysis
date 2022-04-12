import os, sys
import subprocess
import argparse
from .main import main

parser = argparse.ArgumentParser(description='Run analysis in batch')
parser.add_argument('dir', help='the directory of python files to be analyzed')
parser.add_argument('-s', '--sort', help='sort python files by number', action="store_true")
parser.add_argument('-r', '--recursive', help='search for python files recursively', action="store_true")
parser.add_argument('-f', '--file', help='analyze file dir', action="store")
parser.add_argument('-o', '--output-flag', help='output html file', action="store_true")
args = parser.parse_args()

def print_red(msg):
    print("\033[91m {}\033[00m".format(msg))

def write_to_log(filename, msg=""):
    log.write(filename + "\t" + msg + "\n")

def analyze(file, file_path):
    result = subprocess.run(["2to3", "-w", file_path]) 
    if result.returncode:
        print_red("Conversion failed!")
        write_to_log(file, "Conversion failed")
    
    msg = main(file_path)
    if type(msg) == str:
        print_red("Analysis failed!")
        write_to_log(file, msg)
    else:
        write_to_log(file, "Success!\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t".format(msg[0]+msg[1]+msg[3]+msg[4], msg[2], msg[5], sum(msg)))

if __name__ == "__main__":
    if args.file:
        with open(args.file) as f:
            files = f.read().splitlines()
        log_path = args.file + "-log.txt"
    else:
        dir_path = args.dir
        files = filter(lambda file: file.endswith(".py") and not file.endswith(".ir.py"), os.listdir(dir_path))
        if args.recursive:
            files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(dir_path)) for f in fn]
        else:
            files = os.listdir(dir_path)
        files = [f for f in files if f.endswith(".py") and not f.endswith(".ir.py")]
        log_path = os.path.join(dir_path, "log.txt")
    log = open(log_path, "a")
    if args.sort:
        sorted_files = sorted(files, key=lambda x: int(x.split('_')[1].split('.')[0]))
    else:
        sorted_files = files
    for file in sorted_files:
        file_path  = os.path.join("..", "GitHubAPI-Crawler", file) if args.file else os.path.join(dir_path, file) 
        analyze(file, file_path)
    log.close()

    # main.main(test_file_path)