"""Self-paced Ensemble

Usage:
    executable.py [options]
    executable.py --help

 
Options:
    -h --help                           help with SPE useage                 
    --nestimators = no-of-estimators    no of base estimators
    --run = num-of-run                  no of runs
    --bins = num-of-bins                no of bins
    --dataset = dataset-name            'creditcard', 'forest', 'finance'
"""

from docopt import docopt
import sklearn
import time
from spe.dataset import load_custom_data, load_finance_data, load_sklearn_data
from spe.spe import ClassifierSPE
from spe.performance_measure import *
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    print("\n")
    print("==============================================================")
    print("Welcome to Self-paced Ensemble")
    print("==============================================================")
    arguments = docopt(__doc__, version="0.0.rc1")

    RANDOM_STATE = 41
    """
    Setting up the metdata from the commandline arguments
    """
    method = "SPEnsemble"
    number_of_estimators = int(arguments["--nestimators"])
    total_run = int(arguments["--run"])
    dataset = arguments["--dataset"]
    k_bins = int(arguments["--bins"])

    print("\nLoading data.... ")
    """
    Choosing dataset and getting trainset and testset
    """
    if dataset == "creditcard":
        train_X, train_y, test_X, test_y = load_custom_data("spe/creditcard.csv")
    elif dataset == "forest":
        train_X, train_y, test_X, test_y = load_sklearn_data(
            test_size=0.1, minority_label=7, random_state=RANDOM_STATE
        )
    elif dataset == "finance":
        train_X, train_y, test_X, test_y = load_finance_data("spe/finance.csv")

    print("\n Data loaded!")

    print("Method:", method)
    print("Number of estimators: ", number_of_estimators)
    print("Independent runs: ", total_run)
    print()

    scores = []
    aucprc_scores = []
    f1_scores = []
    g_scores = []
    mcc_scores = []
    accuracy_scores = []
    for r in range(total_run):
        print("\nRun: ", r + 1)
        print("-------------------------------------------------------------")
        print()
        """
        Loading the Classifier
        """
        model = ClassifierSPE(
            n_estimators=number_of_estimators,
            base_estimator=sklearn.tree.DecisionTreeClassifier(),
            random_seed_value=RANDOM_STATE,
            number_of_bins=k_bins,
        )
        start_time = time.time()
        print("Fitting model...")
        """
        Fitting the traindata
        """
        model.fit(train_X, train_y)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Model fitted. Time: {}".format(elapsed_time))
        print("\nPredicting test input...")
        """
        Predicting the testdata
        """
        predict_y = model.predict_proba(test_X)
        prob_of_min = predict_y[:, 1]
        print("Prediction done!")
        print("\nAppending prediction socres...")
        accuracy_scores.append(accuracy(test_y, np.rint(prob_of_min)))
        aucprc_scores.append(aucprc(test_y, prob_of_min))
        f1_scores.append(f1_score_optimal(test_y, prob_of_min))
        g_scores.append(g_mean_score_optimal(test_y, prob_of_min))
        mcc_scores.append(aucprc(test_y, prob_of_min))

    accuracy_scores = np.array(accuracy_scores)
    aucprc_scores = np.array(aucprc_scores)
    f1_scores = np.array(f1_scores)
    g_scores = np.array(g_scores)
    mcc_scores = np.array(mcc_scores)

    print(
        "\nMean prediction scores after {} runs with {} estimators: ".format(
            total_run, number_of_estimators
        )
    )
    print("******************************************************************")
    print("\nAccuracy: Mean = {:0.4f}".format(accuracy_scores.mean()))
    print("\nAUCPRC Score: Mean = {:0.4f}".format(aucprc_scores.mean()))
    print("\nF1 Score: Mean = {:0.4f}".format(f1_scores.mean()))
    print("\nG-mean Score: Mean = {:0.4f}".format(g_scores.mean()))
    print("\nMCC Score: Mean = {:0.4f}".format(mcc_scores.mean()))
