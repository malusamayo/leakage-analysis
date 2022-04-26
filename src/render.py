from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter
import os
import pandas as pd
from collections import defaultdict

script_code = '''<script>
    let highlighted = [];
    function highlight_lines(lines) {
        for (let line of highlighted) {
            let ele = document.getElementById(String(line));
            ele.style.backgroundColor = '';
        }
        highlighted = lines;
        for (let line of highlighted) {
            let ele = document.getElementById(String(line));
            ele.style.backgroundColor = 'yellow';
        }
    }
    let marked = [];
    function mark_leak_lines(lines) {
        for (let line of marked) {
            let ele = document.getElementById(String(line));
            ele.style.backgroundColor = '';
        }
        marked = lines;
        for (let line of marked) {
            let ele = document.getElementById(String(line));
            ele.style.backgroundColor = ele.style.backgroundColor = 'lightgreen';
        }
    }
    function show_infos(lines) {
        for (let line of lines) {
            let ele = document.getElementById(String(line) + "-info");
            if (ele) {
                ele.style.display = ele.style.display == 'none'? '': 'none'
            }
        }
    }
</script>
    <style type="text/css">
    .sum table {
    font-family: arial, sans-serif;
    border-collapse: collapse;
    width: 100%;
    }

    .sum td, .sum th {
    border: 1px solid #dddddd;
    text-align: left;
    padding: 8px;
    }

    .sum tr:hover {background-color: #D6EEEE;}
</style>
'''

SUMMARY_TEMP = '''<center>
<table class="sum">
  <tbody><tr>
    <th>Leakage</th>
    <th>#Detected</th>
    <th>Locations</th>
  </tr>
  <tr>
    <td>Pre-processing leakage</td>
    <td>#NUMPRE</td>
    <td>#LOCPRE</td>
  </tr>
  <tr>
    <td>Overlap leakage</td>
    <td>#NUMOVERLAP</td>
    <td>#LOCOVERLAP</td>
  </tr>
  <tr>
    <td>No independence test data</td>
    <td>#NUMMULTI</td>
    <td>#LOCMULTI</td>
  </tr>
</tbody></table></center>
'''

REMIND_STYLE = "background-color: green; color: white; border:none;"
WARN_STYLE = "background-color: red; color: white; border:none;"
def get_button(content, style=None, onclick=None):
    return f'''<button type="button" style="line-height: 85%; {style}" onclick="{onclick}">{content}</button>'''
def wrap_in_link(ele, link_id):
    return f'''<a href="#{link_id}">{ele}</a>'''

def translate_labels(label, invos, invo2lineno):
    allInvo_str = ', '.join(sorted([invo2lineno[invo] for invo in invos]))
    if label == "train":
        return get_button(label, REMIND_STYLE) 
    elif label == "test":
        return get_button(label, REMIND_STYLE)
    elif label == "train-test":
        return get_button("highlight train/test sites", onclick=f"highlight_lines([{allInvo_str}])")
    elif label == "test-train":
        return get_button("highlight train/test sites", onclick=f"highlight_lines([{allInvo_str}])")
    elif label == "test_overlap":
        if len(invos) > 0:
            return get_button("overlap with training data", WARN_STYLE) + ' ' + get_button("potential leak src", onclick=f"mark_leak_lines([{allInvo_str}])")
        return get_button("overlap with training data", WARN_STYLE)
    elif label == "train_overlap":
        return get_button("overlap with all test data", WARN_STYLE)
    elif label == "preprocessing_leak":
        return get_button("potential preprocessing leakage", WARN_STYLE) + ' ' + wrap_in_link(get_button("show and go to first leak src", onclick=f"mark_leak_lines([{allInvo_str}])"), invo2lineno[invos[0]])
    elif label == "test_multiuse":
        return get_button("used multiple times", WARN_STYLE) + ' ' + get_button("highlight other usage", onclick=f"highlight_lines([{allInvo_str}])")
    elif label == "validation":
        return get_button(label, REMIND_STYLE)
    elif label == "no_test":
        return get_button("no independent test data", WARN_STYLE)

def get_columns(filename):
    d = {
        "Telemetry_ModelPair.csv": ['trainModel', 'train', 'trainInvo', 'trainMeth', 'ctx1', 'testModel', 'test', 'testInvo', 'testMeth', 'ctx2'],
        "TrainingDataWithModel.csv": ['model', 'data', 'invo', 'meth', 'ctx'],
        "ValDataWithModel.csv": ['model', 'data', 'invo', 'meth', 'ctx'],
        "ValOrTestDataWithModel.csv": ['model', 'data', 'invo', 'meth', 'ctx'],
        "TaintStartsTarget.csv": ['to', 'toCtx', 'from', 'fromCtx', 'invo', 'meth', 'label'],
        "Telemetry_OverlapLeak.csv": ['trainModel', 'train', 'trainInvo', 'trainMeth', 'ctx1', 'testModel', 'test', 'invo', 'testMeth', 'ctx2'],
        "FinalOverlapLeak.csv": ['trainModel', 'train', 'invo', 'trainMeth', 'ctx', 'cnt'],
        "Telemetry_PreProcessingLeak.csv": ['trainModel', 'train', 'trainInvo', 'trainMeth', 'ctx1', 'testModel', 'test', 'testInvo', 'testMeth', 'ctx2', 'des', 'src'],
        "Telemetry_MultiUseTestLeak.csv": ['testModel', 'test', 'invo', 'meth', 'ctx1', 'testModel2', 'test2', 'invo2', 'meth2', 'ctx2'],
        "NoTestData.csv": ['trainModel', 'train', 'invo', 'trainMeth', 'ctx'],
        "FinalNoTestDataWithMultiUse.csv": ['msg', 'cnt']
    }
    return d[filename]

def read_fact(fact_path, filename):
    return pd.read_csv(os.path.join(fact_path, filename), sep="\t", names=get_columns(filename))

def load_info(fact_path, filename, labels, info, invos=()):
    df = read_fact(fact_path, filename)
    def append_info(row):
        labels[row['invo']][(info, invos)] = None
    df.apply(append_info, axis=1)
    return df

def to_html(input_path, fact_path, html_path, lineno_map):
    with open(input_path) as f:
        code = f.read()
    html = highlight(code, PythonLexer(), HtmlFormatter(full=True, linenos=True))
    html_lines = html.split('\n')

    # locate code area
    st = [i for i, line in enumerate(html_lines) if '<td class="code">' in line][0]
    ed = [i for i, line in enumerate(html_lines) if '</pre>' in line][-1]

    # add lineno tags
    for i in range(st+1, ed):
        # lineno = i - st + 3
        html_lines[i] = f'<span id="{i-st+1}">' + html_lines[i] + '</span>'

    invo2lineno = {}
    with open(os.path.join(fact_path, "InvokeLineno.facts")) as f:
        lines = f.readlines()
        for line in lines:
            invo, lineno = line.strip().split("\t")
            if lineno in lineno_map:
                invo2lineno[invo] = lineno_map[lineno]

    labels = defaultdict(dict) # for each line of code

    # from invocation to html line number
    def invo_idx(invo):
        return int(invo2lineno[invo]) + st - 1

    # return unique invos
    def sorted_invo(invos):
        return tuple(sorted(set(invos)))

    # find train/val/test data
    load_info(fact_path, "TrainingDataWithModel.csv", labels, "train")
    valortests = load_info(fact_path, "ValOrTestDataWithModel.csv", labels, "test")
    load_info(fact_path, "ValDataWithModel.csv", labels, "validation")

    # find train/test pairs
    modelpairs = read_fact(fact_path, "Telemetry_ModelPair.csv") 
    def append_info(row):
        labels[row['trainInvo']][("train-test", sorted_invo(row['testInvo'] + [row['trainInvo']]))] = None
        for testInvo in row['testInvo']:
            labels[testInvo][("test-train", sorted_invo(row['testInvo'] + [row['trainInvo']]))] = None
    modelpairs.groupby("trainInvo")["testInvo"].apply(list).reset_index().apply(append_info, axis=1)


    leaksrc = read_fact(fact_path, "TaintStartsTarget.csv")
    # overlap info
    overlapsrcInvos = set(leaksrc.loc[leaksrc['label'] == "dup"]["invo"])
    load_info(fact_path, "Telemetry_OverlapLeak.csv", labels, "test_overlap", sorted_invo(overlapsrcInvos))
    finaloverlap = load_info(fact_path, "FinalOverlapLeak.csv", labels, "train_overlap")

    # pre-processing info
    preleaks = read_fact(fact_path, "Telemetry_PreProcessingLeak.csv") 
    merged =  pd.merge(preleaks, leaksrc, left_on="src", right_on="from")
    def append_info(row):
        labels[row['testInvo']][("preprocessing_leak", sorted_invo(row['invo']))] = None
    merged.groupby("testInvo")['invo'].apply(list).reset_index().apply(append_info, axis=1)

    # multi-test info
    multileaks1 = read_fact(fact_path, "Telemetry_MultiUseTestLeak.csv") 
    def append_info(row):
        labels[row['invo']][("test_multiuse", sorted_invo(row['invo2'] + [row['invo']]))] = None
    multileaks1.groupby("invo")['invo2'].apply(list).reset_index().apply(append_info, axis=1)

    # no test info
    load_info(fact_path, "NoTestData.csv", labels, "no_test")

    # print('\n'.join(html_lines))

    # adding buttons
    for invo, tags in labels.items():
        for (label, invos) in tags.keys():
            html_lines[invo_idx(invo)] +=  ' ' + translate_labels(label, invos, invo2lineno)

    def invos2buttons(invos):
        return ' '.join([wrap_in_link(get_button(str(invo2lineno[invo])), str(invo2lineno[invo])) for invo in invos])

    def gen_summary():
        summary = SUMMARY_TEMP
        summary = summary.replace("#NUMPRE", str(preleaks["testInvo"].nunique()))
        summary = summary.replace("#LOCPRE", invos2buttons(sorted_invo(preleaks["testInvo"])))
        summary = summary.replace("#NUMOVERLAP", str(finaloverlap["invo"].nunique()))
        summary = summary.replace("#LOCOVERLAP", invos2buttons(sorted_invo(finaloverlap["invo"])))

        notests = read_fact(fact_path, "FinalNoTestDataWithMultiUse.csv")
        summary = summary.replace("#NUMMULTI", str(len(notests)))
        if len(notests) > 0:
            summary = summary.replace("#LOCMULTI", invos2buttons(sorted_invo(valortests["invo"])))
        else:
            summary = summary.replace("#LOCMULTI",  "")
        return summary

    html_lines.insert(0, script_code + gen_summary())
    html_lines[html_lines.index('pre { line-height: 125%; }')] = 'pre { line-height: 145%; }'
    with open(html_path, "w") as f:
        f.write('\n'.join(html_lines))
