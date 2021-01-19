import numpy as np
import random as rnd
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from tqdm import trange
import matplotlib.pyplot as plt

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class SklearnGeneticSelection(object):
    def __init__(self, model, params):

        '''
            PARAMETERS:
            • model — sklearn-like model which will be used for scoring;
            • params — parameters dictionary:
                ○ epochs — iterations count (required);
                ○ population_size — number of chromosomes in population (required);
                ○ score_f — sklearn-like metric function name (required);
                ○ mode — do maximization or minimization of objective function (default is 'max').
                    Possible values: 'min', 'max';
                ○ selection_type — chromosomes selection algorithm (default is 'k_best').
                    Possible values: 'k_best', 'luckers_best';
                ○ parents_selection — parents selection algorithm (default is 'k_way');
                ○ crossover_type — crossover algorithm (default is 'random_parts').
                    Possible values: 'random_parts', 'one_split', 'two_splits', 'random';
                ○ mutation_type — mutation algorithm (default is 'flip').
                    Possible values: 'flip', 'remove', 'swap', 'reverse', 'shuffle', 'random';
                ○ init_type — initialization algorithm (default is 'random');
                ○ k_best — percentage of the best chromosomes in population (default is 0.5);
                ○ mutation_proba — mutation probability (default is 0.8);
                ○ do_val — if validation is required then do_val must be a value in (0;1) 
                            else do_val must equal to 0 (default is 0.1);
                ○ k — number of chromosomes choose from is parents selection stage (default is 3);
                ○ needs_proba — if there requirements in probability prediction or not (default False);
                ○ verbose — if there verbosity is required (default 1);
                ○ random_state — random state (default 0).
        '''

        keys = list(params.keys())

        assert 'epochs' in keys
        assert 'population_size' in keys
        assert 'score_f' in keys

        self.base_model = model
        
        config = {
            'verbose':1,
            'mutation_proba':.8,
            'mode':'max',
            'do_val':.1,
            'needs_proba':False,
            'selection_type':'k_best',
            'parents_selection':'k_way',
            'crossover_type':'random_parts',
            'mutation_type':'flip',
            'k':3,
            'k_best':.5,
            'init_type':'random',
            'random_state':0
        }
        _keys = list(config.keys())+['epochs', 'population_size', 'score_f']

        for el in keys:
            if not el in _keys:
                raise ValueError(f"Unknown parameter {bcolors.BOLD}\'{el}\'{bcolors.ENDC}")

        config.update(params)

        self.epochs = config['epochs']
        self.population_size = config['population_size']
        self.score_f = config['score_f']
        
        self.verbose = config['verbose']
        self.mutation_proba = config['mutation_proba']
        self.mode = config['mode']
        self.do_val = config['do_val']
        self.needs_proba = config['needs_proba']
        self.selection_type = config['selection_type']
        self.parents_selection = config['parents_selection']
        self.crossover_type = config['crossover_type']
        self.mutation_type = config['mutation_type']
        self.k = config['k']
        self.k_best = config['k_best']
        self.init_type = config['init_type']
        self.random_state = config['random_state']
        self.scores_history = []


        self.best_score = None
        self.best_chromosome = None
        self.initial_score = None
        self.population_history = {
            'populations': None,
            'scores': None
        }

        assert self.mutation_proba > 0 and self.mutation_proba <= 1, 'Parameter \'mutation_proba\' must be defined on (0;1].'
        assert self.do_val >= 0 and self.do_val <= 1, 'Parameter \'do_val\' must be defined on [0;1].'
        assert self.needs_proba in [True, False], 'Parameter \'needs_proba\' accepts only booleans.'
        assert self.selection_type in ['k_best', 'luckers_best'], 'Parameter \'selection_type\' accepts only \'k_best\' and \'luckers_best\'.'
        assert self.parents_selection in ['k_way'], 'Parameter \'parents_selection\' accepts only \'k_way\'.'
        assert self.crossover_type in ['random_parts', 'one_split', 'two_splits', 'random'], 'Parameter \'crossover_type\' accepts only \'random_parts\', \'one_split\', \'two_splits\', \'random\'.'
        assert self.mutation_type in ['flip', 'swap', 'reverse', 'remove', 'shuffle', 'random'], 'Parameter \'mutation_type\' accepts only \'flip\', \'swap\', \'reverse\', \'remove\', \'shuffle\', \'random\'.'
        assert self.selection_type == 'k_best' and isinstance(self.k, int), 'Parameter \'k\' must be interger.'
        assert self.parents_selection == 'k_way' and (int(self.population_size*self.k_best) - self.k) > 0, 'Parameter \'k\' must be less than population_size*k_best-1.'
        assert self.k_best > 0 and self.k_best < 1, 'Parameter \'k_best\' must be defined on (0;1).'
        assert self.init_type in ['random'], 'Parameter \'init_type\' accepts only \'random\''
        
        # SEED INITIALIZATION
        np.random.seed(self.random_state)
        rnd.seed(self.random_state)

    # SYSTEM UTILS
    def _population_init(self, size):
        if self.init_type == 'random':
            population = np.random.randint(0, 2, (self.population_size, size))
            for i in range(population.shape[0]):
                if population[i].sum() == 0:
                    population[i, rnd.randint(0, population.shape[1]-1)] = 1
            return population
        else:
            return self.init_type(size)
    def _save_population(self, population, scores):
        if self.population_history['populations'] is None:
            self.population_history['populations'] = population
            self.population_history['scores'] = scores
        else:
            mask = np.isin(population, self.population_history['populations']).sum(axis=1) == population.shape[1]
            self.population_history['populations'] = np.append(self.population_history['populations'], population[mask], axis=0)
            self.population_history['scores'] = np.append(self.population_history['scores'], scores[mask], axis=0)

    # SCORING PHASE
    # Fit and evaluate model for getting chromosome score.
    def _compute_score(self, X, y, eval_set=None):
        model = clone(self.base_model)
        model.fit(X, y)
        
        score = None
        if self.needs_proba:
            if not eval_set is None:
                preds = model.predict_proba(eval_set[0])
                score = self.score_f(eval_set[1], preds)
            else:
                preds = model.predict_proba(X)
                score = self.score_f(y, preds)
        else:
            if not eval_set is None:
                preds = model.predict(eval_set[0])
                score = self.score_f(eval_set[1], preds)
            else:
                preds = model.predict(X)
                score = self.score_f(y, preds)
        
        return score

    # Wrapper for method above for getting population score vector.
    def _get_score_vector(self, population, X, y, eval_set=None):
        scores = np.zeros((population.shape[0]))
        for i in range(population.shape[0]):

            # This one is used for loading from cache.
            if not self.population_history['populations'] is None and population[i].tolist() in self.population_history['populations'].tolist():
                scores[i] = self.population_history['scores'][self.population_history['populations'].tolist().index(population[i].tolist())]
            else:
                mask = population[i] == 1
                if eval_set is None:
                    scores[i] = self._compute_score(X[:, mask], y, eval_set)
                else:
                    scores[i] = self._compute_score(X[:, mask], y, (eval_set[0][:, mask], eval_set[1]))
        return scores
    
    # POPULATION TRANSFORMATION PHASE
    # Selecting best chromosomes.
    def _select_k_best(self, score_vector, population):
        idx = np.argsort(score_vector)
        best_population = None
        if self.mode == 'max':
            best_population = population[idx[int(idx.shape[0]*(1-self.k_best)):]]
            score_vector = score_vector[idx[int(idx.shape[0]*(1-self.k_best)):]]
        else:
            best_population = population[idx[:int(idx.shape[0]*self.k_best)]]
            score_vector = score_vector[idx[:int(idx.shape[0]*self.k_best)]]
        return best_population, score_vector
    
    # This method is used for choosing parents for further crossover.
    def _choose_parents(self, population, scores):
        n_generate = self.population_size-population.shape[0]
        parents = []
        if self.parents_selection == 'k_way':
            while n_generate > 0:
                # Choose first parent
                idx = np.arange(0, population.shape[0])
                np.random.shuffle(idx)
                tmp = idx[:self.k]
                parent1 = tmp[np.argmax(scores[tmp])]

                # Choose second parent
                idx = np.delete(idx, parent1)
                np.random.shuffle(idx)
                tmp = idx[:self.k]
                parent2 = tmp[np.argmax(scores[tmp])]

                if not (parent1, parent2) in parents:
                    parents.append((parent1, parent2))
                    n_generate -= 1
        
        return parents

    # Crossover is used for crossovering parents.
    def _crossover(self, population, parents):
        
        # This method is used for choosing random genes from each of parents merging of which leads to son creation.
        def random_parts():
            nonlocal population
            idx = np.arange(population.shape[1])
            mask = np.isin(idx, np.random.choice(idx, idx.shape[0]//2, replace=False))
            son = np.zeros((1, idx.shape[0]))
            son[:, mask] = population[parent1, mask]
            son[:, ~mask] = population[parent2, ~mask]
            if son.sum() == 0:
                son = np.random.randint(0, 2, son.shape[1])
            population = np.append(population, son, axis=0)
        # Same as above, but here we essentially crop both by half and take halfs from them.
        def one_split():
            nonlocal population
            spl = population.shape[1]//2
            son = np.zeros((1, population.shape[1]))
            if rnd.randint(0, 1):
                son[:, :spl] = population[parent1, :spl]
                son[:, spl:] = population[parent2, spl:]
            else:
                son[:, spl:] = population[parent1, spl:]
                son[:, :spl] = population[parent2, :spl]
            if son.sum() == 0:
                son = np.random.randint(0, 2, son.shape[1])
            population = np.append(population, son, axis=0)
        # Same as above, but not half but a third.
        def two_splits():
            nonlocal population
            son = np.zeros((1, population.shape[1]))
            if rnd.randint(0, 1):
                son[:, :son.shape[1]//3] = population[parent1, :son.shape[1]//3]
                son[:, son.shape[1]//3:son.shape[1]*2//3] = population[parent2, son.shape[1]//3:son.shape[1]*2//3]
                son[:, son.shape[1]*2//3:] = population[parent1, son.shape[1]*2//3:]
            else:
                son[:, :son.shape[1]//3] = population[parent2, :son.shape[1]//3]
                son[:, son.shape[1]//3:son.shape[1]*2//3] = population[parent1, son.shape[1]//3:son.shape[1]*2//3]
                son[:, son.shape[1]*2//3:] = population[parent2, son.shape[1]*2//3:]
            if son.sum() == 0:
                son = np.random.randint(0, 2, son.shape[1])
            population = np.append(population, son, axis=0)

        for (parent1, parent2) in parents:
            if self.crossover_type == 'random_parts':
                random_parts()
            elif self.crossover_type == 'one_split':
                one_split()
            elif self.crossover_type == 'two_splits':
                two_splits()
            elif self.crossover_type == 'random':
                n = rnd.randint(0,2)
                if n == 0:
                    random_parts()
                elif n == 1:
                    one_split()
                else:
                    two_splits()
        return population

    # This functions is used for doing mutation operations on population
    def _mutation(self, population):
        # Mutation probability mask
        mask = (np.random.choice(2, population.shape[0], p=[1-self.mutation_proba, self.mutation_proba]) == 1)

        # Reset is used for approaching 0 features by randomly replacing problem place.
        def reset(x):
            if population[x].sum() == 0:
                population[x] = np.random.randint(0, 2, population[x].shape[0])
                if population[x].sum() == 0:
                    population[x, rnd.randint(0, population.shape[1]-1)] = 1
        
        # Flip is used for binary invertation. It choose random number of genes to invert. 
        def flip():
            lens = np.random.randint(1, population.shape[1]//2, population.shape[0])
            idx = np.array([rnd.randint(0, population.shape[1]-l) for l in lens])

            for i, (id, m, l) in enumerate(zip(idx, mask, lens)):
                if not m:
                    continue

                if (l == 1 and population[i, id] != 1) or l > 1:
                    population[i, id:id+l] = (~(population[i, id:id+l]==1))*1
                elif id+1 < population.shape[1]:
                    population[i, id+1] = 1
                elif id+1 >= population.shape[1]:
                    population[i, id-1] = 1
                reset(i)
        
        # Swap is used for swapping 2 random elements.
        def swap():
            for i, m in enumerate(mask):
                if not m:
                    continue
                
                # Getting idx of 1s and 0s
                idx_sort = np.argsort(population[i])
                sorted_records_array = population[i][idx_sort]
                vals, idx_start, count = np.unique(sorted_records_array, return_counts=True, return_index=True)
                res = np.split(idx_sort, idx_start[1:])

                if len(res) == 1:
                    population[i, rnd.randint(0, population.shape[1]-1)] = (not bool(population[i, rnd.randint(0, population.shape[1]-1)]))*1
                    continue
                
                id0 = rnd.choice(res[0])
                id1 = rnd.choice(res[1])
                population[i, id0], population[i, id1] = population[i, id1], population[i, id0]

                reset(i)   
        
        # Reverse is used for random genes order invertation.
        def reverse():
            lens = np.random.randint(2, population.shape[1]//2, population.shape[0])
            idx = np.array([rnd.randint(0, population.shape[1]-l) for l in lens])

            for i, (id, m, l) in enumerate(zip(idx, mask, lens)):
                if not m:
                    continue
                population[i, id:id+l] = population[i, id:id+l][::-1]

                reset(i)
        
        # Shuffle is used for random genes order shuffling.
        def shuffle():
            lens = np.random.randint(2, population.shape[1]//2, population.shape[0])
            idx = np.array([rnd.randint(0, population.shape[1]-l) for l in lens])

            for i, (id, m, l) in enumerate(zip(idx, mask, lens)):
                if not m:
                    continue
                np.random.shuffle(population[i, id:id+l])

                reset(i)
        
        # Remove is used for random non-zero genes removal.
        def remove():
            for i, m in enumerate(mask):
                if not m:
                    continue
                
                # Getting idx of 1s and 0s
                idx_sort = np.argsort(population[i])
                sorted_records_array = population[i][idx_sort]
                vals, idx_start, count = np.unique(sorted_records_array, return_counts=True, return_index=True)
                res = np.split(idx_sort, idx_start[1:])
                if len(res) == 1 and population[i, 0] == 0:
                    population[i, rnd.randint(0, population.shape[1]-1)] = 1
                    continue
                
                idx = res[1]
                if len(idx) > 1:
                    l = rnd.randint(1, len(idx))
                    population[i, np.random.choice(idx, l, replace=False)] = 0
                
                reset(i)
        
        # This one is used for random using methods above.
        def random_operation():
            for i, m in enumerate(mask):
                if not m:
                    continue

                _type = np.random.choice(5, None, p=[.2, .2, .2, .2, .2])
                if _type == 0: # FLIP
                    l = rnd.randint(1, population.shape[1]//2-1)
                    id = rnd.randint(0, population.shape[1]-l)

                    if (l == 1 and population[i, id] != 1) or l > 1:
                        population[i, id:id+l] = (~(population[i, id:id+l]==1))*1
                    elif id+1 < population.shape[1]:
                        population[i, id+1] = 1
                    elif id+1 >= population.shape[1]:
                        population[i, id-1] = 1

                elif _type == 1: # SWAP
                    # Getting idx of 1s and 0s
                    idx_sort = np.argsort(population[i])
                    sorted_records_array = population[i][idx_sort]
                    vals, idx_start, count = np.unique(sorted_records_array, return_counts=True, return_index=True)
                    res = np.split(idx_sort, idx_start[1:])

                    if len(res) == 1:
                        population[i, rnd.randint(0, population.shape[1]-1)] = (not bool(population[i, rnd.randint(0, population.shape[1]-1)]))*1
                        continue
                    
                    id0 = rnd.choice(res[0])
                    id1 = rnd.choice(res[1])
                    population[i, id0], population[i, id1] = population[i, id1], population[i, id0]

                elif _type == 2: # REVERSE
                    l = rnd.randint(2, population.shape[1]//2-1)
                    id = rnd.randint(0, population.shape[1]-l)
                    population[i, id:id+l] = population[i, id:id+l][::-1]

                elif _type == 3: # SHUFFLE
                    l = rnd.randint(2, population.shape[1]//2-1)
                    id = rnd.randint(0, population.shape[1]-l)
                    np.random.shuffle(population[i, id:id+l])

                else: # REMOVE
                    # Getting idx of 1s and 0s
                    idx_sort = np.argsort(population[i])
                    sorted_records_array = population[i][idx_sort]
                    vals, idx_start, count = np.unique(sorted_records_array, return_counts=True, return_index=True)
                    res = np.split(idx_sort, idx_start[1:])
                    if len(res) == 1 and population[i, 0] == 0:
                        population[i, rnd.randint(0, population.shape[1]-1)] = 1
                        continue
                    elif len(res) == 1 and population[i, 0] == 1:
                        population[i, rnd.randint(0, population.shape[1]-1)] = 0
                        continue

                    idx = res[1]
                    if len(idx) > 1:
                        l = rnd.randint(1, len(idx))
                        population[i, np.random.choice(idx, l, replace=False)] = 0
                reset(i)
        
        if self.mutation_type == 'flip':
            flip()
        elif self.mutation_type == 'swap':
            swap()
        elif self.mutation_type == 'reverse':
            reverse()
        elif self.mutation_type == 'shuffle':
            shuffle()
        elif self.mutation_type == 'remove':
            remove()
        elif self.mutation_type == 'random':
            random_operation()

        return population

    def plot_history(self):
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(len(self.scores_history)), self.scores_history, linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.show()

    def get_best_params(self):
        return self.best_chromosome == 1

    # Return scores history
    def fit(self, X_train, y_train):

        # Initialization
        population = self._population_init(X_train.shape[1])
        eval_set = None
        if self.do_val > 0:
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=self.do_val, random_state=self.random_state, shuffle=True)
            eval_set = (X_val, y_val)

        if self.verbose:
            self.initial_score = self._compute_score(X_train, y_train, eval_set)
            print(bcolors.HEADER, 'Initial score:', self.initial_score, bcolors.ENDC)
            print(bcolors.HEADER, 'Initial features count:', X_train.shape[1], bcolors.ENDC)

        pbar = trange(self.epochs) if self.verbose else range(self.epochs)

        for epoch in pbar:
            # Getting score vector of the population
            scores = self._get_score_vector(population, X_train, y_train, eval_set)
            self.scores_history.append(max(scores))

            # Best score and chromosome saving
            if self.best_score is None:
                self.best_score = scores.max() if self.mode == 'max' else scores.min()
                self.best_chromosome = population[scores.tolist().index(self.best_score)]
            else:
                self.best_score = (scores.max() if scores.max() > self.best_score else self.best_score) if self.mode == 'max' else (scores.min() if scores.min() < self.best_score else self.best_score)
                if self.best_score in scores:
                    self.best_chromosome = population[scores.tolist().index(self.best_score)]

            self._save_population(population, scores)

            # Select, Crossover, Mutation
            population, scores = self._select_k_best(scores, population)
            parents = self._choose_parents(population, scores)
            population = self._crossover(population, parents)
            population = self._mutation(population)

            if self.verbose:
                pbar.set_postfix({"best_score":self.best_score, "features_count":self.best_chromosome.sum()})
        
        if self.verbose:
            print(bcolors.HEADER, 'Final score:', self._compute_score(X_train[:, self.best_chromosome==1], y_train, (X_val[:, self.best_chromosome==1], y_val)), bcolors.ENDC)
            print(bcolors.HEADER, 'Final features count:', self.best_chromosome.sum(), bcolors.ENDC)
            print()
        
        return self.scores_history

class CatBoostGeneticSelection(SklearnGeneticSelection):
    def __init__(self, model, params):
        '''
            PARAMETERS:
            • model — sklearn-like model which will be used for scoring;
            • params — parameters dictionary:
                ○ epochs — iterations count (required);
                ○ population_size — number of chromosomes in population (required);
                ○ score_f — sklearn-like metric function name (required);
                ○ mode — do maximization or minimization of objective function (default is 'max').
                    Possible values: 'min', 'max';
                ○ selection_type — chromosomes selection algorithm (default is 'k_best').
                    Possible values: 'k_best', 'luckers_best';
                ○ parents_selection — parents selection algorithm (default is 'k_way');
                ○ crossover_type — crossover algorithm (default is 'random_parts').
                    Possible values: 'random_parts', 'one_split', 'two_splits', 'random';
                ○ mutation_type — mutation algorithm (default is 'flip').
                    Possible values: 'flip', 'remove', 'swap', 'reverse', 'shuffle', 'random';
                ○ init_type — initialization algorithm (default is 'random');
                ○ k_best — percentage of the best chromosomes in population (default is 0.5);
                ○ mutation_proba — mutation probability (default is 0.8);
                ○ do_val — if validation is required then do_val must be a value in (0;1) 
                            else do_val must equal to 0 (default is 0.1);
                ○ k — number of chromosomes choose from is parents selection stage (default is 3);
                ○ needs_proba — if there requirements in probability prediction or not (default False);
                ○ verbose — if there verbosity is required (default 1);
                ○ cat_features — categorical features passed in CatBoost model (default is None);
                ○ random_state — random state (default 0).
        '''

        keys = list(params.keys())

        assert 'epochs' in keys
        assert 'population_size' in keys
        assert 'score_f' in keys

        self.base_model = model
        
        config = {
            'cat_features':None,
            'verbose':1,
            'mutation_proba':.8,
            'mode':'max',
            'do_val':.1,
            'needs_proba':False,
            'selection_type':'k_best',
            'parents_selection':'k_way',
            'crossover_type':'random_parts',
            'mutation_type':'flip',
            'k':3,
            'k_best':.5,
            'init_type':'random',
            'random_state':0
        }
        _keys = list(config.keys())+['epochs', 'population_size', 'score_f']

        for el in keys:
            if not el in _keys:
                raise ValueError(f"Unknown parameter {bcolors.BOLD}\'{el}\'{bcolors.ENDC}")

        config.update(params)

        self.epochs = config['epochs']
        self.population_size = config['population_size']
        self.score_f = config['score_f']
        
        self.verbose = config['verbose']
        self.mutation_proba = config['mutation_proba']
        self.mode = config['mode']
        self.do_val = config['do_val']
        self.needs_proba = config['needs_proba']
        self.selection_type = config['selection_type']
        self.parents_selection = config['parents_selection']
        self.crossover_type = config['crossover_type']
        self.mutation_type = config['mutation_type']
        self.k = config['k']
        self.k_best = config['k_best']
        self.init_type = config['init_type']
        self.cat_features = config['cat_features']
        self.random_state = config['random_state']
        self.scores_history = []


        self.best_score = None
        self.best_chromosome = None
        self.initial_score = None
        self.population_history = {
            'populations': None,
            'scores': None
        }

        assert self.mutation_proba > 0 and self.mutation_proba <= 1, 'Parameter \'mutation_proba\' must be defined on (0;1].'
        assert self.do_val >= 0 and self.do_val <= 1, 'Parameter \'do_val\' must be defined on [0;1].'
        assert self.needs_proba in [True, False], 'Parameter \'needs_proba\' accepts only booleans.'
        assert self.selection_type in ['k_best', 'luckers_best'], 'Parameter \'selection_type\' accepts only \'k_best\' and \'luckers_best\'.'
        assert self.parents_selection in ['k_way'], 'Parameter \'parents_selection\' accepts only \'k_way\'.'
        assert self.crossover_type in ['random_parts', 'one_split', 'two_splits', 'random'], 'Parameter \'crossover_type\' accepts only \'random_parts\', \'one_split\', \'two_splits\', \'random\'.'
        assert self.mutation_type in ['flip', 'swap', 'reverse', 'remove', 'shuffle', 'random'], 'Parameter \'mutation_type\' accepts only \'flip\', \'swap\', \'reverse\', \'remove\', \'shuffle\', \'random\'.'
        assert self.selection_type == 'k_best' and isinstance(self.k, int), 'Parameter \'k\' must be interger.'
        assert self.parents_selection == 'k_way' and (int(self.population_size*self.k_best) - self.k) > 0, 'Parameter \'k\' must be less than population_size*k_best-1.'
        assert self.k_best > 0 and self.k_best < 1, 'Parameter \'k_best\' must be defined on (0;1).'
        assert self.init_type in ['random'], 'Parameter \'init_type\' accepts only \'random\''
        
        # SEED INITIALIZATION
        np.random.seed(self.random_state)
        rnd.seed(self.random_state)

    # SCORING PHASE
    # Fit and evaluate model for getting chromosome score.
    def _compute_score(self, X, y, eval_set=None):
        model = clone(self.base_model)
        model.fit(X, y, self.cat_features, eval_set=eval_set, use_best_model=True)
        
        score = None
        if self.needs_proba:
            if not eval_set is None:
                preds = model.predict_proba(eval_set[0])
                score = self.score_f(eval_set[1], preds)
            else:
                preds = model.predict_proba(X)
                score = self.score_f(y, preds)
        else:
            if not eval_set is None:
                preds = model.predict(eval_set[0])
                score = self.score_f(eval_set[1], preds)
            else:
                preds = model.predict(X)
                score = self.score_f(y, preds)
        
        return score

if __name__ == '__main__':
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score
    from sklearn.datasets import make_classification
    SEED=1337
    params = {
            'epochs':20,
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
    sgs.plot_history()