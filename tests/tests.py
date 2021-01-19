import numpy as np
import random as rnd
from sklearn.datasets import make_classification, load_boston, fetch_california_housing, load_breast_cancer, load_wine, fetch_kddcup99
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import warnings
from sklearn.metrics import f1_score, mean_squared_error, accuracy_score
from prettytable import PrettyTable
from gafs import SklearnGeneticSelection, bcolors
warnings.simplefilter('ignore')

SEED = 1337

if __name__ == '__main__':
    final_score = []
    initial_score = []
    final_n_features = []
    initial_n_features = [50, 13, 30, 13, 8, 41]
    datasets = ['Sintetic (4 informative features)', 'Boston', 'Breast cancer', 'Wine', 'California housing']

    params = {
            'epochs':100,
            'population_size':20,
            'verbose':1,
            'mutation_proba':.8,
            'mode':'max',
            'do_val':.2,
            'needs_proba':False,
            'selection_type':'k_best',
            'parents_selection':'k_way',
            'crossover_type':'random',
            'mutation_type':'random',
            'k':5,
            'k_best':.5,
            'init_type':'random',
            'random_state':SEED
        }

    params['score_f'] = f1_score
    print(f'{bcolors.OKCYAN}Sintetic classification test (4 informative features, 1000 samples) | F1 score:{bcolors.ENDC}')
    X, y = make_classification(n_samples=1000, n_features=50, n_informative=4, n_repeated=0, n_classes=2, shuffle=True, random_state=0)
    sgs = SklearnGeneticSelection(RandomForestClassifier(random_state=SEED), params)
    sgs.fit(X, y)
    final_n_features.append(int(sgs.get_best_params().sum()))
    initial_score.append(sgs.initial_score)
    final_score.append(sgs.best_score)

    params['score_f'] = mean_squared_error
    params['mode'] = 'min'
    print(f'{bcolors.OKCYAN}Boston house prices dataset (506 samples) | MSE:{bcolors.ENDC}')
    X, y = load_boston(return_X_y=True)
    sgs = SklearnGeneticSelection(RandomForestRegressor(random_state=SEED), params)
    sgs.fit(X, y)
    final_n_features.append(int(sgs.get_best_params().sum()))
    initial_score.append(sgs.initial_score)
    final_score.append(sgs.best_score)

    params['score_f'] = f1_score
    params['mode'] = 'max'
    print(f'{bcolors.OKCYAN}Breast cancer dataset (569 samples) | F1 score:{bcolors.ENDC}')
    X, y = load_breast_cancer(return_X_y=True)
    sgs = SklearnGeneticSelection(RandomForestClassifier(random_state=SEED), params)
    sgs.fit(X, y)
    final_n_features.append(int(sgs.get_best_params().sum()))
    initial_score.append(sgs.initial_score)
    final_score.append(sgs.best_score)

    params['score_f'] = accuracy_score
    print(f'{bcolors.OKCYAN}Wine quality dataset (178 samples, multiclass) | Accuracy score:{bcolors.ENDC}')
    X, y = load_wine(return_X_y=True)
    sgs = SklearnGeneticSelection(RandomForestClassifier(random_state=SEED), params)
    sgs.fit(X, y)
    final_n_features.append(int(sgs.get_best_params().sum()))
    initial_score.append(sgs.initial_score)
    final_score.append(sgs.best_score)

    params['score_f'] = mean_squared_error
    params['mode'] = 'min'
    print(f'{bcolors.OKCYAN}California housing dataset (20640 samples) | MSE:{bcolors.ENDC}')
    X, y = fetch_california_housing(return_X_y=True)
    sgs = SklearnGeneticSelection(RandomForestRegressor(n_jobs=-1, random_state=SEED), params)
    sgs.fit(X, y)
    final_n_features.append(int(sgs.get_best_params().sum()))
    initial_score.append(sgs.initial_score)
    final_score.append(sgs.best_score)

    x = PrettyTable()

    x.field_names = [f'{bcolors.HEADER}Dataset', "Initial score", "Initial features count", "Final score", f"Final features count{bcolors.ENDC}"]
    t = list(zip(datasets, initial_score, initial_n_features, final_score, final_n_features))
    for row in t:
        x.add_row(row)
    
    print(x)