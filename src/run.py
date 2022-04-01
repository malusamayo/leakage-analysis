import os, sys
import subprocess
import argparse

parser = argparse.ArgumentParser(description='Run analysis in batch')
parser.add_argument('dir', help='the directory of python files to be analyzed')
parser.add_argument('-s', '--sort', help='sort python files by number', action="store_true")
parser.add_argument('-r', '--recursive', help='search for python files recursively', action="store_true")
args = parser.parse_args()

def print_red(msg):
    print("\033[91m {}\033[00m".format(msg))

def write_to_log(filename, msg=""):
    log.write(filename + "\t" + msg + "\n")

if __name__ == "__main__":
    dir_path = sys.argv[1]
    log = open(os.path.join(dir_path, "log.txt"), "a")
    files = filter(lambda file: file.endswith(".py") and not file.endswith(".ir.py"), os.listdir(dir_path))
    if args.recursive:
        files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(dir_path)) for f in fn]
    else:
        files = os.listdir(dir_path)
    files = [f for f in files if f.endswith(".py") and not f.endswith(".ir.py")]
    if args.sort:
        sorted_files = sorted(files, key=lambda x: int(x.split('_')[1].split('.')[0]))
    else:
        sorted_files = files
    for file in sorted_files:
        file_path  = os.path.join(dir_path, file)
        result = subprocess.run(["2to3", "-w", file_path]) 
        if result.returncode:
            print_red("Conversion failed!")
            write_to_log(file, "Conversion failed")
        result = subprocess.run(["python3", "-m", "src.main", file_path]) 
        if result.returncode:
            print_red("Analysis failed!")
            if result.returncode == 37:
                write_to_log(file, "Failed to parse!")
            else:
                write_to_log(file, "Unknown: " + str(result.returncode))
            continue
        write_to_log(file, "Success!")
    log.close()

    # main.main(test_file_path)