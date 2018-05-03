"""
# Created by Lukas on 10.12.2017#
Topic:
Task:
Description
"""
import datetime
import data_preparation
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix

from sklearn.metrics import f1_score,accuracy_score
import numpy as np

from StorageLogic import runLogic, simple_signal

def calculate_load_cyles(bat):
    """ a Ray is better than a bug
    This function estimates the load cycles by adding all the storage input together """
    a_ray = np.array(bat.history["delta_storage_level"])
    feed_ins = a_ray[a_ray > 0]
    feed_out = a_ray[a_ray < 0]

    return feed_ins.sum()


def heatmap(values, xlabel, ylabel, xticklabels, yticklabels, cmap=None,
            vmin=None, vmax=None, ax=None, fmt="%0.2f"):
    if ax is None:
        ax = plt.gca()
    font = {'family': 'monospace',
            'size': 15}
    import matplotlib
    matplotlib.rc('font', **font)
    # plot the mean cross-validation scores
    img = ax.pcolor(values, cmap=cmap, vmin=vmin, vmax=vmax)
    img.update_scalarmappable()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(xticklabels)) + .5)
    ax.set_yticks(np.arange(len(yticklabels)) + .5)
    ax.set_xticklabels(xticklabels, rotation=0)
    ax.set_yticklabels(yticklabels)
    ax.set_aspect(1)

    for p, color, value in zip(img.get_paths(), img.get_facecolors(),
                               img.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.mean(color[:3]) > 0.5:
            c = 'k'
        else:
            c = 'w'
        ax.text(x, y, fmt % value, color=c, ha="center", va="center")

    return img

##################################################
#             >>  OUTPUT SET UP  <<              #
##################################################

with open("summary_table_{}.csv".format(datetime.date.today()), "w") as file:
    """
    Create a table for the Output
    """
    columns = ["name",
               "Accuracy (Train)",
               "Accuracy (Test)",
               "F1 (Train)",
               "F1 (Test)",
               "profit 2014",
               "rel profit 2014",
               "profit 2015",
               "rel profit 2015",
               "load cycles",
               "profit per load cycle",
               "LCOE"]
    for column in columns:
        file.write(column)
        file.write(";")
    file.write("\n")


def write_infos_in_table(name, clf, Mode=None):
    if Mode == "NN":
        pred_14 = clf["X_14"]
        pred_15 = clf["X_15"]

    elif Mode == "back2back":
        pred_14 = list(y_14)[clf:] + list(y_14)[:clf]
        pred_15 = list(y_15)[clf:] + list(y_15)[:clf]

    elif Mode == "simple":
        pred_14 = simple_signal(prices_14)
        pred_15 = simple_signal(prices_15)

    elif Mode == "GAMS":
        pred_14 = y_14
        pred_15 = y_15

    else:
        pred_14 = clf.predict(X_14)
        pred_15 = clf.predict(X_15)

    report_14 = runLogic(name + "_14",
                         price_series=prices_14,
                         signals=pred_14,
                         signal_format="1")

    report_15 = runLogic(name + "_15",
                         price_series=prices_15,
                         signals=pred_15,
                         signal_format="1")

    lc = calculate_load_cyles(report_15)
    name = name
    acc_train = accuracy_score(pred_14, y_14)
    acc_test = accuracy_score(pred_15, y_15)
    f1_train = f1_score(pred_14, y_14, average="weighted")
    f1_test = f1_score(pred_15, y_15, average="weighted")
    profit_14 = report_14.balance
    rel_profit_14 = report_14.balance / GAMS_14
    profit_15 = report_15.balance
    rel_profit_15 = report_15.balance / GAMS_15
    load_cycles_15 = lc
    profit_per_loadcycle = profit_15 / lc
    LCOE = "NOT Implemented"

    report_15.mk_report_dataFrame(suffix=name, filename=name, date=True)
    output = [name,
              acc_train,
              acc_test,
              f1_train,
              f1_test,
              profit_14,
              rel_profit_14,
              profit_15,
              rel_profit_15,
              load_cycles_15,
              profit_per_loadcycle]

    with open("summary_table_{}.csv".format(datetime.date.today()), "a") as file:
        for cell in output:
            file.write(str(cell))
            file.write(";")
        file.write("\n")

    scores_image = heatmap(confusion_matrix(y_15, pred_15),
                           xlabel='Predicted label',
                           ylabel='True label',
                           xticklabels=["discharge", "wait", "charge"],
                           yticklabels=["discharge", "wait", "charge"],
                           cmap=plt.cm.gray_r, fmt="%d")

    plt.title("{}\n{:.2f}".format(name.replace("_","").title(),f1_test))
    plt.gca().invert_yaxis()
    plt.tight_layout()
    # plt.subplots_adjust(left=0.0, right=0.75, top=0.92, bottom=0.09)
    # plt.show()
    plt.savefig("pictures\\Confusion Martix\\{}".format(name))
    plt.clf()


##################################################
#                 >>  IMPORT  <<                 #
##################################################

if True:
    INFOS = data_preparation.get_dict()
    pickle.dump(INFOS, open("INFOS.p", "wb"))

INFOS = pickle.load(open("INFOS.p", "rb"))

X = INFOS["X"]
y = INFOS["y"]

X_14 = INFOS["X_14"]
y_14 = INFOS["y_14"]

X_15 = INFOS["X_eval"]
y_15 = INFOS["y_eval"]

GAMS_14 = INFOS["GAMS_result_14"]
GAMS_15 = INFOS["GAMS_result_15"]

prices_14 = INFOS["prices_eval_14"]
prices_15 = INFOS["prices_eval_15"]

##################################################
#            >>  CLASSIFIER SETUP  <<            #
##################################################
KNN = 1
DecisionTree = 1
RandomForest = 1
LogReg = 1
SVM = 1
NN = 1
if KNN:
    print(">>> KNN <<<")
    from sklearn.neighbors import KNeighborsClassifier

    pipe = make_pipeline(QuantileTransformer(),
                         PCA(100),
                         KNeighborsClassifier(n_neighbors=11,
                                              weights="distance",
                                              p=1,
                                              n_jobs=-1
                                              ))

    pipe.fit(X_14, y_14)
    write_infos_in_table("KNN", pipe)

if DecisionTree:
    print(">>> DT <<<")
    from sklearn.tree import DecisionTreeClassifier

    pipe = make_pipeline(DecisionTreeClassifier(class_weight="balanced",
                                                criterion="entropy",
                                                max_depth=50,
                                                min_samples_leaf=1,
                                                min_samples_split=50))

    pipe.fit(X, y)
    write_infos_in_table("Decision Tree", pipe)

if RandomForest:
    print(">>> Random Forest <<<")
    from sklearn.ensemble import RandomForestClassifier
    pipe = make_pipeline(RandomForestClassifier(class_weight="balanced",
                                                n_estimators=1000,
                                                criterion="entropy",
                                                # max_features=90,
                                                n_jobs=-1))

    pipe.fit(X, y)
    write_infos_in_table("Random Forest", pipe)

if LogReg:
    print(">>> LogReg <<<")
    from sklearn.linear_model import LogisticRegression

    pipe = make_pipeline(QuantileTransformer(),
                         PolynomialFeatures(2),
                         PCA(1000),
                         LogisticRegression(class_weight="balanced",
                                            solver="saga",
                                            penalty="L2",
                                            C=0.1,
                                            max_iter=100000))

    pipe.fit(X_14, y_14)
    print("LOG still 14")
    write_infos_in_table("Logistic Regression", pipe)

if SVM:
    print(">>> SVM - RBF <<<")
    from sklearn.svm import SVC

    pipe = make_pipeline(StandardScaler(),
                         SVC(class_weight="balanced",
                             C=10,
                             gamma = 0.0001,
                             kernel= "rbf",
                             max_iter=100000))

    pipe.fit(X_14, y_14)
    write_infos_in_table("SVC - RBF", pipe)

if SVM:
    print(">>> SVM - Linear <<<")
    from sklearn.svm import SVC

    pipe = make_pipeline(QuantileTransformer(),
                         SVC(kernel="linear",
                             C = 10,
                             max_iter=1000000))

    pipe.fit(X_14, y_14)
    write_infos_in_table("SVC - Linear", pipe)

if NN:
    print(">>> NN <<<")
    import feed_forward_neuronal_network
    pred = feed_forward_neuronal_network.create_and_train(INFOS)
    write_infos_in_table("Neuronal Network", clf = pred, Mode="NN")



##################################################
#             >>  TRAIN DUMMIES  <<              #
##################################################
_168 = True
_24 = True
threshold = True
if _168:
    pred = -168
    write_infos_in_table("Shift Week", clf =pred , Mode="back2back")
if _24:
    pred = -24
    write_infos_in_table("Shift Day", clf =pred, Mode="back2back")

if threshold:
    write_infos_in_table("Simple_signal", clf=pred, Mode="simple")


write_infos_in_table("LP", clf=None, Mode="GAMS")

