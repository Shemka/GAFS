# GAFS
Genetic Algorithm Feature Selection is a library for ML solutions better perfomance. It use GA for iterative model-based feature selection.
[Good article about GA](https://towardsdatascience.com/introduction-to-genetic-algorithms-including-example-code-e396e98d8bf3).

## Installation
```git clone  https://github.com/Shemka/GAFS.git```
### Pip
```pip install -r requirements.txt```
### Conda
```conda install --file requirements.txt```

## Quick start
Let's check few examples of library usage. First of all, import all modules and create dataset:
```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier  
from sklearn.metrics import f1_score
from gafs import SklearnGeneticSelection

X, y = make_classification(n_samples=1000, n_features=50, n_informative=4, n_repeated=0, n_classes=2, shuffle=True, random_state=0)
```
Define SGS object, its params and fit it:
```python
params = {
  'epochs':100,
  'population_size':20,
  'score_f':f1_score,
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
  'random_state':1337
}
sgs = SklearnGeneticSelection(RandomForestClassifier(random_state=SEED), params)
sgs.fit(X, y)
```
Output:
```
 Initial score: 0.7942583732057416 
 Initial features count: 50 
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [05:56<00:00,  3.56s/it, best_score=0.901, features_count=9]
 Final score: 0.9014084507042254 
 Final features count: 9.0 
```
Get features mask:
```python
mask = sgs.get_best_params()
```
## Tests
  All the tests can be found in `tests.py` where algorithm was evaluated in some simple tasks. Tests results:
|Dataset|Initial score|Initial features count|Final score|Final features count|
|:-----:|:-----------:|:--------------------:|:---------:|:------------------:|
|Sintetic (4 informative features)|0.794|50|0.901|9|
|Boston|10.9195|13|10.5168|12|
|Breast cancer|0.952|30|1.0|11|
|Wine|0.916|13|1.0|8|
|California housing|0.2449|8|0.2158|4|
## TODO
- Documentation;
- CatBoost, XGBoost, LightGBM special classes with early stoppings.
