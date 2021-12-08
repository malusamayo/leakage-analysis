import os, sys
import pandas as pd
sys.path.append('../src')
from main import main
# from irgen import CodeTransformer


def template(test_file, shouldBeTainted):
    test_file_path = os.path.join("..", "tests", "inputs", test_file)
    test_fact_path = os.path.join("inputs", test_file.replace(".py", "-fact"), "Leak.csv")
    main(test_file_path)
    assert os.path.exists(test_fact_path)
    df = pd.read_csv(test_fact_path, sep="\t", names=["heap", "invo", "method"])
    isTainted = df["method"].map(lambda m: m == "LogisticRegression.fit").any()
    assert isTainted == shouldBeTainted, "Leak undetected!" if shouldBeTainted else "False leak detected!"

class TestClass:
    def test0(self):
        template("test0.py", shouldBeTainted=True)

    def test1(self):
        template("test1.py", shouldBeTainted=False)

    def test2(self):
        template("test2.py", shouldBeTainted=True)
