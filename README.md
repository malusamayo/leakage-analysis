# Leakage Analysis
A static analysis tool to detect test data leakage in Python notebooks

## How to build
1. Install [souffle](https://souffle-lang.github.io/install), the datalog engine we use for our main analysis. Make sure that souffle could be directly invoked in command line.
2. Pull and build our customized version of [pyright], the type inference engine we use: ```git submodule update --init --recursive ``` (please refer to the submodule for building the project).
3. Install required Python packages in requirements.txt.

## How to use
1. Run analysis for a single Python file: ```python3 -m src.main /path/to/file```
2. Run analysis for all Python files in a directory: ```python3 -m src.run /path/to/dir```
3. More information could be found using the `-h` flag.

## How to build and run Docker image
1. Pull our customized version of pyright, the type inference engine we use: ```git submodule update --init --recursive```.
2. Add all used Python libraries to `requirements.txt`, which will be installed in the container and used by pyright.
3. Build Docker image: ```docker build -t leakage-analysis .```
4. Run Docker image: ```docker run -v /path/to/dir:/path/to/dir leakage-analysis /path/to/dir/$FILE -o```. All to-be-analyzed notebooks should be converted to Python files and stored in `/path/to/dir`.

## How to read output
For a given input file `test.py`, an output html file `test.html` will be generated if `-o` flag is specified. 

In `test.html`, we show the analysis results alongside input code. A summary table on detected leakage issues is shown on the top. Users could also utilize the interactive buttons to highlight relevant code and navigate through different code segments.

## Internal Structure

Given a Python file, `src/main.py` first parses the input into AST. 
Then it feeds AST to a GlobalCollector instance (from `global_collector.py`) that collects global variables we could not rename in later transformations, which we will ignore later. 

Next, it feeds AST to a CodeTransformer instance (from `irgen.py`) that translates original Python code to a simpler version that 1) breaks down complex statements to multiple simpler ones, and 2) translates code to the static single assignment (SSA) form. 

Then it calls the type inference engine on the transformed code file. With type inference information, it converts the code file to datalog facts the final analysis could read, using FactGenerator from `factgen.py`.

Finally, it performs datalog analysis (`main.dl`) on generated facts and outputs results in the same directory.

## Directory Structure

```
src
├── factgen.py: convert transformed code to datalog facts
├── global_collector.py: collect global variables
├── __init__.py
├── irgen.py: transform code to simpler SSA form
├── main.dl: main datalog analysis that analyzes leakage
├── main.py: run analysis on a single file
├── render.py: output a html file based on analysis results and original code
├── run.py: run analysis on multiple files
└── scope.py: manage variable scopes for renaming purposes
```
