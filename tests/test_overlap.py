import os, sys
import pandas as pd
from src import main


def template(test_file, taintMethods):
    test_file_path = os.path.join(".", "tests", "inputs", test_file)
    test_fact_path = os.path.join("tests", "inputs", test_file.replace(".py", "-fact"), "OverlapLeak.csv")
    main.main(test_file_path)

    assert os.path.exists(test_fact_path), "Leak result not found!"
    df = pd.read_csv(test_fact_path, sep="\t", names=["model1", "ctx1", "model2", "ctx2", "invo", "method"])
    # isTainted = df["method"].map(lambda m: m in taintMethods).any()
    print(df["method"])

    def report():
        print(set(taintMethods).difference(set(df["method"])))
        hasFalseNeg = len(set(taintMethods).difference(set(df["method"]))) > 0
        hasFalsePos = len(set(df["method"]).difference(set(taintMethods))) > 0
        if hasFalseNeg:
            return "Leak undetected!!!"
        if hasFalsePos:
            return "False leak detected!!!"
        assert False, "Should not reach here"
    
    assert set(df["method"]) == set(taintMethods), report()


def test_basic():
    template("testOversampler.py", ["HistGradientBoostingClassifier.fit"])
    template("testOversampler2.py", [])

def test_smote():
    template("yogmoh_news-category.py", ["RandomForestClassifier.fit", "Unknown.fit", "XGBClassifier.fit"])

def test_advanced():
    template("dktalaicha_credit-card-fraud-detection-using-smote-adasyn.py", ["LogisticRegression.fit", "Unknown.fit"])
    template("nb_598984.py",[])

def test_funcdef():
    template("shahules_tackling-class-imbalance.py", ["LogisticRegression.fit"])



