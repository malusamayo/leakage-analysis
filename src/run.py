import os, sys
import subprocess

def print_red(msg):
    print("\033[91m {}\033[00m".format(msg))

syntax_error_files = ["nb_9024.py", "nb_30750.py", "nb_112899.py", "nb_164489.py", "nb_321208.py", "nb_377254.py",
    "nb_383709.py", "nb_477435.py", "nb_499964.py", "nb_645563.py", "nb_658949.py", "nb_662657.py", # written in js 
    "nb_383709.py", "nb_669803.py", "nb_704442.py", "nb_707224.py", 3]

if __name__ == "__main__":
    dir_path = sys.argv[1]
    for file in os.listdir(dir_path):
        if not file.endswith(".py") or file.endswith(".ir.py"):
            continue
        file_path  = os.path.join(dir_path, file)
        result = subprocess.run(["2to3", "-w", file_path]) 
        if result.returncode:
            print_red("Conversion failed!")
            print(file_path)
        result = subprocess.run(["python3", "-m", "src.main", file_path]) 
        if result.returncode:
            print_red("Analysis failed!")
            print(file_path)

    # main.main(test_file_path)