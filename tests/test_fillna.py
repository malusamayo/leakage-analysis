import os, sys
import pandas as pd
from src import main


def template(test_file, shouldBeTainted):
    test_file_path = os.path.join(".", "tests", "inputs", test_file)
    test_fact_path = os.path.join("tests", "inputs", test_file.replace(".py", "-fact"), "Leak.csv")
    main.main(test_file_path)
    assert os.path.exists(test_fact_path)
    df = pd.read_csv(test_fact_path, sep="\t", names=["heap", "invo", "method"])
    isTainted = df["method"].map(lambda m: m == "LogisticRegression.fit").any()
    assert isTainted == shouldBeTainted, "Leak undetected!" if shouldBeTainted else "False leak detected!"


def test0():
    template("test0.py", shouldBeTainted=True)

def test1():
    template("test1.py", shouldBeTainted=False)

def test2():
    template("test2.py", shouldBeTainted=True)

def test_titanic():
    template("titanic0.py", shouldBeTainted=True)