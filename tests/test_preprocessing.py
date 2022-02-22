import os, sys
import pandas as pd
from src import main


def template(test_file, taintMethods):
    test_file_path = os.path.join(".", "tests", "inputs", test_file)
    test_fact_path = os.path.join("tests", "inputs", test_file.replace(".py", "-fact"), "PreProcessingLeak.csv")
    main.main(test_file_path)

    assert os.path.exists(test_fact_path), "Leak result not found!"
    df = pd.read_csv(test_fact_path, sep="\t", names=["heap", "invo", "method"])
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
    template("test0.py", ["LogisticRegression.fit"])
    template("test1.py", ["LogisticRegression.fit"]) # should not use distribution of whole test set as well!
    template("test2.py", ["LogisticRegression.fit"])
    template("test3.py", [])
    template("test4.py", [])

def test_fillna():
    template("titanic0.py", ["LogisticRegression.fit"])
    template("nb_344873.py", ["Sequential.fit", "LassoCV.fit", "XGBRegressor.fit"])
    

def test_tfidf():
    template("nb_100841.py", ["MultinomialNB | Unbound.fit"])
    template("nb_334422.py", ['AdaBoostRegressor.fit', 'DecisionTreeRegressor.fit', 
                            'GradientBoostingRegressor.fit', 'KNeighborsRegressor.fit', 
                            'LassoCV.fit', 'LinearRegression.fit',
                            'RandomForestRegressor.fit', 'RidgeCV.fit'])

def test_dataFrameMapper():
    template("nb_132929.py", ["GaussianNB.fit", "Unbound.fit"])

def test_scaler():
    template("nb_194503.py", ["Unknown.fit"])
    template("nb_362989.py", ["GaussianNB.fit", "Unbound.fit"])
    template("nb_292583.py", ["GridSearchCV.fit", "AdaBoostClassifier.fit", "Any | Unknown | type.fit", "Unknown.fit"]) 
    template("nb_473437.py", [])

def test_pca():
    template("nb_205857.py", ["KMeans.fit", "RandomForestClassifier.fit", "Unbound.fit"]) 
    template("nb_471253.py", [])

def test_countvec():
    template("nb_303674.py", ["Unknown.fit"])
    template("nb_1020535.py", [])

def test_pipeline():
    template("nb_276778.py", ["RandomizedSearchCV.fit"])
    template("nb_277256.py", [])

def test_feature_selection():
    template("nb_387986.py", ["RandomForestRegressor.fit", "LinearRegression.fit", "Pipeline.fit", "RidgeCV.fit"])

def test_applymap():
    template("nb_344814.py", ["LogisticRegression.fit"])

def test_equiv_edge():
    template("nb_282393.py", ["Unbound | Sequential.fit_generator", "Sequential.fit_generator"])

def test_loop():
    template("nb_175471.py", ["MultinomialNB.fit"]) # , "Unknown.fit"]) [could not handle indexing]
    template("nb_248151.py", ["Unbound.fit"]) 

def test_context_sensitivity():
    template("nb_273933.py", ["GridSearchCV.fit", "Unknown.fit", "KNeighborsClassifier.fit"])

def test_classdef():
    template("nb_424904.py", []) #'Unknown | type.fit']) [could not distinguish non-dataflow & flow insensitivity]

def test_funcdef():
    template("nb_481597.py", ["RandomForestRegressor.fit"]) 

def test_branch():
    template("nb_1080148.py", ["Unknown.fit"]) 

def test_cut():
    template("nb_1090319.py", ["Unknown.fit", "LogisticRegression.fit"]) 
    template("nb_1255091.py", ["DecisionTreeClassifier.fit", "KNeighborsClassifier.fit", 'RandomForestClassifier.fit'])