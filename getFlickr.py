import numpy as np

FLICKR_DATA_PATH= "/shared/flickr_style/"
STYLE_NAME_PATH = FLICKR_DATA_PATH + "style_names.txt"
OBJS_BY_FILENAME_PATH = FLICKR_DATA_PATH + "flickr_obj_label.txt"
NUM_OF_STYLES = 20
TOP_N = 10
OBJ_WEIGHT = [0.9, 0.7, 0.5, 0.3, 0.1]
STYLE_WEIGHT = [0.9, 0.7, 0.5, 0.3, 0.1]

flickr_test_set = np.loadtxt(FLICKR_DATA_PATH+'test.txt', str, delimiter='\t')
flickr_test_filename = [readline.split()[0].split('/')[-1] for readline in flickr_test_set]
flickr_test_label = [int(readline.split()[1]) for readline in flickr_test_set]
flickr_test_dict = dict(zip(flickr_test_filename, flickr_test_label))

flickr_train_set = np.loadtxt(FLICKR_DATA_PATH+'train.txt', str, delimiter='\t')
flickr_train_filename = [readline.split()[0].split('/')[-1] for readline in flickr_train_set]
flickr_train_label = [int(readline.split()[1]) for readline in flickr_train_set]
flickr_train_dict = dict(zip(flickr_train_filename, flickr_train_label))

def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z

flickr_all_dict = merge_two_dicts(flickr_train_dict, flickr_test_dict)

list = {'predict_obj': [{'score': '0.121914', 'name': 'gown', 'label': 'n03450230'}, {'score': '0.119266', 'name': 'jersey,T-shirt,teeshirt', 'label': 'n03595614'}, {'score': '0.0851326', 'name': 'maillot', 'label': 'n03710637'}, {'score': '0.0685063', 'name': 'website,website,internetsite,site', 'label': 'n06359193'}, {'score': '0.0598228', 'name': 'sunglass', 'label': 'n04355933'}], 'predict_style': [{'score': '0.252059', 'name': 'Bright'}, {'score': '0.212549', 'name': 'Pastel'}, {'score': '0.110541', 'name': 'Vintage'}, {'score': '0.0962888', 'name': 'Romantic'}, {'score': '0.0546268', 'name': 'Depth of Field'}]}

def getStyleList():
    f = open(STYLE_NAME_PATH, "r")
    style_list = []
    while 1:
        line = f.readline()
        if not line: break
        line = line.split('\n')
        style_list.append(line[0])

    f.close()
    return style_list


style_list = getStyleList()

pred_obj = list["predict_obj"]
pred_sty = list["predict_style"]

pred_objcol = np.ones((5,1))
pred_styrow = np.ones((1,5))

for i in range(0, len(pred_sty)):
    pred_objcol[i][0] = float(pred_obj[i]["score"])*OBJ_WEIGHT[i]
    pred_styrow[0][i] = float(pred_sty[i]["score"])*STYLE_WEIGHT[i]

mat = np.dot(pred_objcol, pred_styrow)
mat = mat/mat.flatten().sum()       # normalize
idxs = np.argsort(mat.flatten())    # 0 to 24

# sort from the greatest
for i in range(1, TOP_N+1):
    row_n = idxs[-i]/5
    col_n = idxs[-i]%5
    print pred_sty[col_n]["name"]
    print pred_obj[row_n]["name"],"(", pred_obj[row_n]["label"],")"
    print mat[row_n][col_n]
    print style_list.index(pred_sty[col_n]["name"])
    print "---------------"

# get obj list
f = open(OBJS_BY_FILENAME_PATH, "r")
styleClassifier = [None]*NUM_OF_STYLES
for i in xrange(0, NUM_OF_STYLES):
    styleClassifier[i] = {}

while 1:
    line = f.readline()
    if not line: break

    lineEle = line.split(',')
    filePath = lineEle[0]

    fileName = filePath.split('/')[-1]
    fileObjs = []
    for j in xrange(1, len(lineEle), 2):
        try:
            labelCode = lineEle[j]
            labelScore = lineEle[j+1].split('\n')[0]
            tmp = {}
            tmp[labelCode] = labelScore
            fileObjs.append(tmp)

            styleClassifier[flickr_all_dict[fileName]][fileName] = fileObjs

        except:
            print "error!"
            print fileName
            print flickr_all_dict[fileName]
            print fileObjs

f.close()

print styleClassifier[0]
