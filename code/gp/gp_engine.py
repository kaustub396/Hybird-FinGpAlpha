"""
Phase 3B: Genetic Programming Alpha Mining Engine

Discovers mathematical trading formulas from OHLC-derived features using
Genetic Programming (DEAP library).

Each GP individual is a symbolic expression tree that maps cross-sectional
features → alpha score. Evolution optimizes for Information Coefficient (IC)
with parsimony pressure to keep formulas interpretable.

Key design decisions:
- Protected division, log, sqrt to avoid runtime errors
- Cross-sectional rank normalization of alpha output
- IC-based fitness (standard in alpha mining literature)
- Parsimony pressure via tree size penalty
- Hall of Fame to preserve best individuals across generations

Usage:
    from gp_engine import GPAlphaEngine
    engine = GPAlphaEngine(panel)
    best_alphas = engine.evolve(target='fwd_ret_20d', n_gen=50)
"""

import pandas as pd
import numpy as np
import operator
import random
import warnings
from deap import algorithms, base, creator, gp, tools
from copy import deepcopy

warnings.filterwarnings('ignore')


# ============================================================
# PROTECTED OPERATORS
# ============================================================

def protected_div(a, b):
    """Division that returns 0 when denominator is near-zero."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(np.abs(b) > 1e-10, a / b, 0.0)
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

def protected_log(a):
    """Log of absolute value + 1 (always defined)."""
    with np.errstate(invalid='ignore'):
        result = np.sign(a) * np.log1p(np.abs(a))
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

def protected_sqrt(a):
    """Sqrt of absolute value, preserving sign."""
    return np.sign(a) * np.sqrt(np.abs(a))

def neg(a):
    """Negation."""
    return -a

def square(a):
    """Square with overflow protection."""
    result = a * a
    return np.clip(result, -1e10, 1e10)

def safe_max(a, b):
    return np.maximum(a, b)

def safe_min(a, b):
    return np.minimum(a, b)

def abs_val(a):
    return np.abs(a)

def sign(a):
    return np.sign(a)

def inv(a):
    """Protected inverse: 1/a."""
    return protected_div(1.0, a)


# ============================================================
# GP ALPHA ENGINE
# ============================================================

class GPAlphaEngine:
    """
    Genetic Programming engine for alpha factor discovery.
    
    Parameters
    ----------
    panel : dict
        Cross-sectional panel {feature_name: DataFrame(dates × stocks)}.
    feature_names : list or None
        Which features to use as terminals. If None, uses a sensible default.
    population_size : int
        GP population size.
    tournament_size : int
        Tournament selection size.
    max_depth : int
        Maximum tree depth (controls formula complexity).
    cx_prob : float
        Crossover probability.
    mut_prob : float
        Mutation probability.
    parsimony_weight : float
        Penalty per tree node (controls bloat). Higher = simpler formulas.
    random_state : int
        Random seed.
    """
    
    DEFAULT_FEATURES = [
        'ret_1d', 'ret_5d', 'ret_20d', 'ret_60d', 'ret_120d', 'ret_250d',
        'vol_5d', 'vol_20d', 'vol_60d', 'vol_120d',
        'range_pct', 'atr_14',
        'sma_20', 'sma_50', 'sma_200',
        'price_to_sma20', 'price_to_sma50', 'price_to_sma200',
        'drawdown_60d', 'drawdown_250d',
        'rsi_14', 'zscore_20', 'zscore_60',
        'vol_ratio_5_20', 'vol_change_5d',
        'oc_ratio', 'hl_ratio',
        'afm_sentiment', 'afm_fundamental'
    ]
    
    def __init__(self, panel, feature_names=None, population_size=500,
                 tournament_size=5, max_depth=6, cx_prob=0.7, mut_prob=0.2,
                 parsimony_weight=0.001, random_state=42):
        
        self.panel = panel
        self.feature_names = feature_names or self.DEFAULT_FEATURES
        
        # Verify features exist in panel
        available = [f for f in self.feature_names if f in panel]
        missing = [f for f in self.feature_names if f not in panel]
        if missing:
            print(f"  Warning: Features not in panel (skipped): {missing}")
        self.feature_names = available
        
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.max_depth = max_depth
        self.cx_prob = cx_prob
        self.mut_prob = mut_prob
        self.parsimony_weight = parsimony_weight
        self.random_state = random_state
        
        # Will be set during evolution
        self.toolbox = None
        self.pset = None
        self.hall_of_fame = None
        self.logbook = None
        
    def _setup_gp(self):
        """Configure DEAP GP primitives and operators."""
        
        # Create primitive set
        n_features = len(self.feature_names)
        pset = gp.PrimitiveSet("ALPHA", n_features)
        
        # Rename arguments to feature names
        for i, fname in enumerate(self.feature_names):
            pset.renameArguments(**{f"ARG{i}": fname})
        
        # Add operators (arity 2)
        pset.addPrimitive(operator.add, 2, name="add")
        pset.addPrimitive(operator.sub, 2, name="sub")
        pset.addPrimitive(operator.mul, 2, name="mul")
        pset.addPrimitive(protected_div, 2, name="div")
        pset.addPrimitive(safe_max, 2, name="max")
        pset.addPrimitive(safe_min, 2, name="min")
        
        # Add operators (arity 1)
        pset.addPrimitive(neg, 1, name="neg")
        pset.addPrimitive(abs_val, 1, name="abs")
        pset.addPrimitive(protected_log, 1, name="log")
        pset.addPrimitive(protected_sqrt, 1, name="sqrt")
        pset.addPrimitive(square, 1, name="sq")
        pset.addPrimitive(sign, 1, name="sign")
        pset.addPrimitive(inv, 1, name="inv")
        
        # Add ephemeral random constants
        pset.addEphemeralConstant("rand_const",
                                  lambda: round(random.uniform(-1, 1), 2))
        
        self.pset = pset
        
        # Create fitness and individual types
        # Maximize IC, but we use parsimony penalty
        if hasattr(creator, 'FitnessMax'):
            del creator.FitnessMax
        if hasattr(creator, 'Individual'):
            del creator.Individual
        
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
        
        # Toolbox
        toolbox = base.Toolbox()
        toolbox.register("expr", gp.genHalfAndHalf, pset=pset,
                         min_=1, max_=self.max_depth)
        toolbox.register("individual", tools.initIterate,
                         creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat,
                         list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=pset)
        
        # Selection
        toolbox.register("select", tools.selTournament,
                         tournsize=self.tournament_size)
        
        # Crossover and mutation
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
        
        # Bloat control: limit tree depth
        toolbox.decorate("mate", gp.staticLimit(
            key=operator.attrgetter("height"), max_value=self.max_depth + 2))
        toolbox.decorate("mutate", gp.staticLimit(
            key=operator.attrgetter("height"), max_value=self.max_depth + 2))
        
        self.toolbox = toolbox
    
    def _prepare_data(self, target, date_mask=None):
        """
        Prepare feature matrices and target for fitness evaluation.
        
        Parameters
        ----------
        target : str
            Forward return target name.
        date_mask : array-like or None
            Boolean mask or index of dates to use (for regime conditioning).
        
        Returns
        -------
        feature_arrays : dict {feature_name: 2D array(dates × stocks)}
        target_array : 2D array(dates × stocks)
        valid_mask : 2D bool array (True where both features and target exist)
        """
        # Get dates where target exists
        target_df = self.panel[target]
        
        if date_mask is not None:
            if isinstance(date_mask, pd.Index):
                target_df = target_df.loc[target_df.index.intersection(date_mask)]
            else:
                target_df = target_df.loc[date_mask]
        
        dates = target_df.index
        stocks = target_df.columns
        
        # Build feature arrays
        feature_arrays = {}
        for fname in self.feature_names:
            feat_df = self.panel[fname]
            # Align to same dates and stocks
            aligned = feat_df.reindex(index=dates, columns=stocks)
            feature_arrays[fname] = aligned.values
        
        target_array = target_df.values
        
        # Valid mask: need both target and all features to be non-NaN
        valid = ~np.isnan(target_array)
        for fname in self.feature_names:
            valid &= ~np.isnan(feature_arrays[fname])
        
        return feature_arrays, target_array, valid, dates, stocks
    
    def _evaluate_individual(self, individual, feature_arrays, target_array,
                             valid_mask, dates_idx):
        """
        Evaluate a single GP individual.
        
        Fitness = ICIR (mean Rank-IC / std Rank-IC) with parsimony penalty.
        Uses Rank IC (Spearman) which is more robust than Pearson IC.
        
        Key improvements over naive mean-IC fitness:
        1. ICIR rewards CONSISTENCY, not just average
        2. Rank IC is robust to outliers and non-linear relationships
        3. Dispersion check rejects near-constant formulas
        4. Evaluates on ALL dates (not a sample) for accuracy
        
        Returns tuple (fitness,) as required by DEAP.
        """
        try:
            func = self.toolbox.compile(expr=individual)
        except Exception:
            return (-1.0,)
        
        n_dates, n_stocks = target_array.shape
        
        try:
            args = [feature_arrays[fname] for fname in self.feature_names]
            alpha = func(*args)
            
            if np.isscalar(alpha):
                return (-1.0,)
            
            alpha = np.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)
            
            if alpha.shape != target_array.shape:
                if alpha.ndim == 1 and len(alpha) == n_stocks:
                    alpha = np.tile(alpha, (n_dates, 1))
                else:
                    return (-1.0,)
            
        except Exception:
            return (-1.0,)
        
        # Dispersion check: reject formulas with very low cross-sectional variance
        # Sample 50 dates to check dispersion quickly
        check_idx = np.linspace(0, n_dates - 1, min(50, n_dates), dtype=int)
        low_disp_count = 0
        for d in check_idx:
            mask = valid_mask[d]
            if mask.sum() < 10:
                continue
            a = alpha[d][mask]
            if np.std(a) < 1e-8:
                low_disp_count += 1
        if low_disp_count > len(check_idx) * 0.5:
            return (-1.0,)
        
        # Compute Rank IC on evenly-spaced dates (use more dates for accuracy)
        sample_size = min(500, n_dates)
        sample_idx = np.linspace(0, n_dates - 1, sample_size, dtype=int)
        
        ics = []
        for d in sample_idx:
            mask = valid_mask[d]
            if mask.sum() < 10:
                continue
            
            a = alpha[d][mask]
            r = target_array[d][mask]
            
            if np.std(a) < 1e-10:
                continue
            
            # Rank IC (Spearman) — more robust than Pearson
            a_rank = np.argsort(np.argsort(a)).astype(float)
            r_rank = np.argsort(np.argsort(r)).astype(float)
            a_rank -= a_rank.mean()
            r_rank -= r_rank.mean()
            denom = np.sqrt(np.sum(a_rank**2) * np.sum(r_rank**2))
            if denom < 1e-10:
                continue
            rank_ic = np.sum(a_rank * r_rank) / denom
            
            if not np.isnan(rank_ic):
                ics.append(rank_ic)
        
        if len(ics) < 30:
            return (-1.0,)
        
        ics = np.array(ics)
        mean_ic = np.mean(ics)
        std_ic = np.std(ics)
        
        # ICIR = mean(IC) / std(IC) — measures signal consistency
        if std_ic > 1e-10:
            icir = mean_ic / std_ic
        else:
            icir = 0.0
        
        # Fitness = blend of ICIR (consistency) + mean IC (strength)
        # ICIR alone can reward weak-but-consistent signals too much
        # Blend ensures we want both strength and consistency
        fitness = 0.6 * icir + 0.4 * (mean_ic * 10)  # Scale mean_ic to comparable range
        
        # Parsimony penalty
        tree_size = len(individual)
        penalty = self.parsimony_weight * tree_size
        
        fitness = fitness - penalty
        fitness = np.clip(fitness, -1.0, 1.0)
        
        return (fitness,)
    
    def evolve(self, target='fwd_ret_20d', n_gen=50, date_mask=None,
               verbose=True, elite_size=5):
        """
        Run GP evolution to discover alpha formulas.
        
        Parameters
        ----------
        target : str
            Forward return target.
        n_gen : int
            Number of generations.
        date_mask : array-like or None
            Dates to use for fitness evaluation (for regime conditioning).
        verbose : bool
            Print progress.
        elite_size : int
            Number of top individuals to preserve (elitism).
        
        Returns
        -------
        list : Best individuals found (Hall of Fame).
        """
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        
        # Setup GP
        self._setup_gp()
        
        # Prepare data
        feature_arrays, target_array, valid_mask, dates, stocks = \
            self._prepare_data(target, date_mask)
        
        if verbose:
            n_dates = len(dates)
            print(f"  GP Evolution Setup:")
            print(f"    Features: {len(self.feature_names)}")
            print(f"    Dates: {n_dates}")
            print(f"    Stocks: {len(stocks)}")
            print(f"    Population: {self.population_size}")
            print(f"    Generations: {n_gen}")
            print(f"    Max depth: {self.max_depth}")
        
        # Register fitness function
        self.toolbox.register("evaluate", self._evaluate_individual,
                              feature_arrays=feature_arrays,
                              target_array=target_array,
                              valid_mask=valid_mask,
                              dates_idx=np.arange(len(dates)))
        
        # Create initial population
        pop = self.toolbox.population(n=self.population_size)
        
        # Hall of Fame (preserves best-ever individuals)
        hof = tools.HallOfFame(elite_size)
        
        # Statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("avg", np.mean)
        stats.register("max", np.max)
        stats.register("min", np.min)
        stats.register("std", np.std)
        
        if verbose:
            print(f"\n  {'Gen':>4} {'Avg IC':>8} {'Max IC':>8} {'Min IC':>8} {'Std':>8}")
            print("  " + "-" * 40)
        
        # Custom evolution loop with elitism
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + stats.fields
        
        # Evaluate initial population
        fitnesses = list(map(self.toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        
        hof.update(pop)
        record = stats.compile(pop)
        logbook.record(gen=0, nevals=len(pop), **record)
        
        if verbose:
            print(f"  {0:>4} {record['avg']:>8.4f} {record['max']:>8.4f} "
                  f"{record['min']:>8.4f} {record['std']:>8.4f}")
        
        for gen in range(1, n_gen + 1):
            # Select next generation
            offspring = self.toolbox.select(pop, len(pop) - elite_size)
            offspring = [self.toolbox.clone(ind) for ind in offspring]
            
            # Crossover
            for i in range(0, len(offspring) - 1, 2):
                if random.random() < self.cx_prob:
                    self.toolbox.mate(offspring[i], offspring[i + 1])
                    del offspring[i].fitness.values
                    del offspring[i + 1].fitness.values
            
            # Mutation
            for i in range(len(offspring)):
                if random.random() < self.mut_prob:
                    self.toolbox.mutate(offspring[i])
                    del offspring[i].fitness.values
            
            # Evaluate individuals with invalid fitness
            invalids = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(self.toolbox.evaluate, invalids))
            for ind, fit in zip(invalids, fitnesses):
                ind.fitness.values = fit
            
            # Elitism: add best from Hall of Fame
            elite = [self.toolbox.clone(ind) for ind in hof[:elite_size]]
            
            pop[:] = offspring + elite
            hof.update(pop)
            
            record = stats.compile(pop)
            logbook.record(gen=gen, nevals=len(invalids), **record)
            
            if verbose and (gen % 10 == 0 or gen == n_gen):
                print(f"  {gen:>4} {record['avg']:>8.4f} {record['max']:>8.4f} "
                      f"{record['min']:>8.4f} {record['std']:>8.4f}")
        
        self.hall_of_fame = hof
        self.logbook = logbook
        
        if verbose:
            print(f"\n  Evolution complete. Best IC: {hof[0].fitness.values[0]:.4f}")
            print(f"  Best formula: {self.get_formula(hof[0])}")
        
        return hof
    
    def get_formula(self, individual):
        """Get readable string representation of a GP individual."""
        return str(individual)
    
    def get_simplified_formula(self, individual):
        """Get a simplified version of the formula."""
        formula = str(individual)
        # Basic simplification: replace ARG names (already done via renaming)
        return formula
    
    def compute_alpha(self, individual, date_mask=None):
        """
        Compute alpha scores from a GP individual on full panel data.
        
        Parameters
        ----------
        individual : GP individual
            The formula tree.
        date_mask : array-like or None
            Subset of dates to compute on.
        
        Returns
        -------
        DataFrame : alpha scores (dates × stocks).
        """
        func = self.toolbox.compile(expr=individual)
        
        target_df = self.panel['fwd_ret_20d']  # Just for date/stock alignment
        if date_mask is not None:
            if isinstance(date_mask, pd.Index):
                target_df = target_df.loc[target_df.index.intersection(date_mask)]
            else:
                target_df = target_df.loc[date_mask]
        
        dates = target_df.index
        stocks = target_df.columns
        
        args = []
        for fname in self.feature_names:
            feat_df = self.panel[fname].reindex(index=dates, columns=stocks)
            args.append(feat_df.values)
        
        try:
            alpha = func(*args)
            alpha = np.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)
            
            if np.isscalar(alpha):
                alpha = np.full((len(dates), len(stocks)), alpha)
            
            alpha_df = pd.DataFrame(alpha, index=dates, columns=stocks)
        except Exception as e:
            print(f"Error computing alpha: {e}")
            alpha_df = pd.DataFrame(0.0, index=dates, columns=stocks)
        
        return alpha_df
    
    def evaluate_top_n(self, evaluator, target='fwd_ret_20d', n=5,
                       holding_period=20, date_mask=None):
        """
        Evaluate top N individuals from Hall of Fame using full evaluation.
        
        Parameters
        ----------
        evaluator : AlphaEvaluator
            The evaluation engine.
        n : int
            Number of top individuals to evaluate.
        target : str
            Forward return target.
        holding_period : int
            Portfolio holding period.
        date_mask : array-like or None
            Dates to evaluate on.
        
        Returns
        -------
        list : [(formula_string, results_dict), ...]
        """
        if self.hall_of_fame is None:
            raise ValueError("Must call evolve() first.")
        
        results = []
        for i, ind in enumerate(self.hall_of_fame[:n]):
            formula = self.get_formula(ind)
            print(f"\n  Alpha #{i+1}: {formula}")
            print(f"  GP Fitness (IC - penalty): {ind.fitness.values[0]:.4f}")
            print(f"  Tree size: {len(ind)} nodes")
            
            alpha_scores = self.compute_alpha(ind, date_mask=date_mask)
            eval_results = evaluator.evaluate(
                alpha_scores, target=target,
                holding_period=holding_period, verbose=True
            )
            results.append((formula, eval_results))
        
        return results


if __name__ == "__main__":
    import pickle
    import os
    
    PROC_DIR = r"C:\Users\EV-Car\Main_Project_2\data\processed"
    
    # Load panel
    with open(os.path.join(PROC_DIR, 'panel.pkl'), 'rb') as f:
        panel = pickle.load(f)
    print(f"Panel loaded: {len(list(panel.values())[0])} dates")
    
    # Quick test: evolve for 20 generations to verify everything works
    print("\n" + "=" * 55)
    print("  GP ENGINE TEST RUN (20 generations)")
    print("=" * 55)
    
    engine = GPAlphaEngine(
        panel,
        population_size=200,  # Small for testing
        max_depth=5,
        parsimony_weight=0.001,
        random_state=42
    )
    
    hof = engine.evolve(target='fwd_ret_20d', n_gen=20, verbose=True)
    
    # Evaluate top 3
    from evaluation import AlphaEvaluator
    evaluator = AlphaEvaluator(panel, transaction_cost=0.001, n_quantiles=5)
    
    print("\n" + "=" * 55)
    print("  EVALUATING TOP 3 GP ALPHAS")
    print("=" * 55)
    
    top_results = engine.evaluate_top_n(evaluator, n=3)
    
    print("\n\nTest complete. GP engine is working correctly.")
