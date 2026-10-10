"""The underlying data structure used in gplearn.

The :mod:`gplearn._program` module contains the underlying representation of a
computer program. It is used for creating and evolving programs used in the
:mod:`gplearn.genetic` module.
"""

# Author: Trevor Stephens <trevorstephens.com>
#
# License: BSD 3 clause

from copy import copy

import numpy as np
from sklearn.utils.random import sample_without_replacement

from .functions import _Function
from .typed_functions import (
    DAILY_MATRIX,
    MINUTE_TENSOR,
    TypedFunction,
)
from .utils import check_random_state


class _Program(object):

    """A program-like representation of the evolved program.

    This is the underlying data-structure used by the public classes in the
    :mod:`gplearn.genetic` module. It should not be used directly by the user.

    Parameters
    ----------
    function_set : list
        A list of valid functions to use in the program.

    arities : dict
        A dictionary of the form `{arity: [functions]}`. The arity is the
        number of arguments that the function takes, the functions must match
        those in the `function_set` parameter.

    init_depth : tuple of two ints
        The range of tree depths for the initial population of naive formulas.
        Individual trees will randomly choose a maximum depth from this range.
        When combined with `init_method='half and half'` this yields the well-
        known 'ramped half and half' initialization method.

    init_method : str
        - 'grow' : Nodes are chosen at random from both functions and
          terminals, allowing for smaller trees than `init_depth` allows. Tends
          to grow asymmetrical trees.
        - 'full' : Functions are chosen until the `init_depth` is reached, and
          then terminals are selected. Tends to grow 'bushy' trees.
        - 'half and half' : Trees are grown through a 50/50 mix of 'full' and
          'grow', making for a mix of tree shapes in the initial population.

    n_features : int
        The number of features in `X`.

    const_range : tuple of two floats
        The range of constants to include in the formulas.

    metric : _Fitness object
        The raw fitness metric.

    p_point_replace : float
        The probability that any given node will be mutated during point
        mutation.

    parsimony_coefficient : float
        This constant penalizes large programs by adjusting their fitness to
        be less favorable for selection. Larger values penalize the program
        more which can control the phenomenon known as 'bloat'. Bloat is when
        evolution is increasing the size of programs without a significant
        increase in fitness, which is costly for computation time and makes for
        a less understandable final result. This parameter may need to be tuned
        over successive runs.

    random_state : RandomState instance
        The random number generator. Note that ints, or None are not allowed.
        The reason for this being passed is that during parallel evolution the
        same program object may be accessed by multiple parallel processes.

    transformer : _Function object, optional (default=None)
        The function to transform the output of the program to probabilities,
        only used for the SymbolicClassifier.

    feature_names : list, optional (default=None)
        Optional list of feature names, used purely for representations in
        the `print` operation or `export_graphviz`. If None, then X0, X1, etc
        will be used for representations.

    program : list, optional (default=None)
        The flattened tree representation of the program. If None, a new naive
        random tree will be grown. If provided, it will be validated.

    Attributes
    ----------
    program : list
        The flattened tree representation of the program.

    raw_fitness_ : float
        The raw fitness of the individual program.

    fitness_ : float
        The penalized fitness of the individual program.

    oob_fitness_ : float
        The out-of-bag raw fitness of the individual program for the held-out
        samples. Only present when sub-sampling was used in the estimator by
        specifying `max_samples` < 1.0.

    parents : dict, or None
        If None, this is a naive random program from the initial population.
        Otherwise it includes meta-data about the program's parent(s) as well
        as the genetic operations performed to yield the current program. This
        is set outside this class by the controlling evolution loops.

    depth_ : int
        The maximum depth of the program tree.

    length_ : int
        The number of functions and terminals in the program.

    """

    def __init__(self,
                 function_set,
                 arities,
                 init_depth,
                 init_method,
                 n_features,
                 const_range,
                 metric,
                 p_point_replace,
                 parsimony_coefficient,
                 random_state,
                 transformer=None,
                 feature_names=None,
                 feature_types=None,
                 program=None,
                 typed_root_reducer_mode=False):

        self.function_set = function_set
        self.arities = arities
        self.init_depth = (init_depth[0], init_depth[1] + 1)
        self.init_method = init_method
        self.n_features = n_features
        self.const_range = const_range
        self.metric = metric
        self.p_point_replace = p_point_replace
        self.parsimony_coefficient = parsimony_coefficient
        self.transformer = transformer
        self.feature_names = feature_names
        self.feature_types = feature_types
        self.program = program
        self.typed_root_reducer_mode = typed_root_reducer_mode

        if self.program is not None:
            if not self.validate_program():
                raise ValueError('The supplied program is incomplete.')
        else:
            # Create a naive random program
            self.program = self.build_program(random_state)

        self.raw_fitness_ = None
        self.fitness_ = None
        self.parents = None
        self._n_samples = None
        self._max_samples = None
        self._indices_state = None

    def _typed_feature_pools(self):
        minute_features = []
        daily_features = []
        if not self.feature_types:
            minute_features = list(range(self.n_features))
            daily_features = list(range(self.n_features))
            return minute_features, daily_features

        for idx, feature_type in enumerate(self.feature_types):
            if feature_type == MINUTE_TENSOR:
                minute_features.append(idx)
            elif feature_type == DAILY_MATRIX:
                daily_features.append(idx)
        return minute_features, daily_features

    def _sample_typed_terminal(self, random_state, expected_type):
        minute_features, daily_features = self._typed_feature_pools()
        if expected_type == DAILY_MATRIX:
            candidates = daily_features
            allow_const = False
        else:
            candidates = minute_features or list(range(self.n_features))
            allow_const = self.const_range is not None

        if allow_const:
            pick = random_state.randint(len(candidates) + 1)
            if pick == len(candidates):
                return random_state.uniform(*self.const_range)
            return candidates[pick]

        if not candidates:
            raise ValueError(f'No terminals available for expected type {expected_type}')
        return candidates[random_state.randint(len(candidates))]

    def _typed_functions_for_output(self, output_type):
        return [
            function for function in self.function_set
            if isinstance(function, TypedFunction) and function.output_type == output_type
        ]

    def _typed_subtree_info(self, program, start_index=0):
        node = program[start_index]
        if not isinstance(node, TypedFunction):
            output_type = self._node_output_type(node)
            if output_type is None:
                return start_index + 1, False, None, []
            return start_index + 1, True, output_type, [(start_index, start_index + 1, output_type, False)]

        next_index = start_index + 1
        spans = []
        for expected_type in node.input_types:
            next_index, is_valid, child_output_type, child_spans = self._typed_subtree_info(program, next_index)
            if not is_valid or child_output_type != expected_type:
                return next_index, False, node.output_type, []
            spans.extend(child_spans)

        spans.append((start_index, next_index, node.output_type, True))
        return next_index, True, node.output_type, spans

    def _typed_subtree_spans(self, program):
        if not program:
            return []

        try:
            end_index, is_valid, _, spans = self._typed_subtree_info(program, 0)
        except Exception:
            return []
        if not is_valid or end_index != len(program):
            return []
        return spans

    def _select_typed_subtree(self, random_state, program=None, expected_output_type=None, allow_root=True):
        program = self.program if program is None else program
        spans = self._typed_subtree_spans(program)
        if expected_output_type is not None:
            spans = [span for span in spans if span[2] == expected_output_type]
        if not allow_root:
            spans = [span for span in spans if span[0] != 0]
        if not spans:
            return None

        weights = np.array([0.9 if span[3] else 0.1 for span in spans], dtype=float)
        probs = np.cumsum(weights / weights.sum())
        choice = np.searchsorted(probs, random_state.uniform())
        return spans[choice]

    def _sample_typed_span(self, random_state, spans):
        if not spans:
            return None
        weights = np.array([0.9 if span[3] else 0.1 for span in spans], dtype=float)
        probs = np.cumsum(weights / weights.sum())
        choice = np.searchsorted(probs, random_state.uniform())
        return spans[choice]

    def _typed_replacement_candidates(self, node):
        if not isinstance(node, TypedFunction):
            return []

        return [
            function for function in self.function_set
            if isinstance(function, TypedFunction)
            and function.arity == node.arity
            and function.input_types == node.input_types
            and function.output_type == node.output_type
        ]

    def _build_typed_subtree(self, random_state, expected_type, current_depth, max_depth, method):
        function_candidates = self._typed_functions_for_output(expected_type)
        can_choose_function = bool(function_candidates) and current_depth < max_depth
        choose_function = False
        if can_choose_function:
            if method == 'full':
                choose_function = True
            else:
                choose_function = bool(random_state.randint(2))

        if not choose_function:
            return [self._sample_typed_terminal(random_state, expected_type)]

        function = function_candidates[random_state.randint(len(function_candidates))]
        program = [function]
        for child_type in function.input_types:
            program.extend(self._build_typed_subtree(random_state, child_type, current_depth + 1, max_depth, method))
        return program

    def _node_output_type(self, node):
        if isinstance(node, TypedFunction):
            return node.output_type
        if isinstance(node, float):
            return MINUTE_TENSOR
        if isinstance(node, int):
            if self.feature_types and 0 <= node < len(self.feature_types):
                return self.feature_types[node]
            return MINUTE_TENSOR
        return None

    def _terminal_runtime_value(self, X, node):
        if isinstance(node, float):
            return np.full(X.shape[:-1], node, dtype=float)
        if isinstance(node, int):
            terminal_type = self._node_output_type(node)
            if terminal_type == DAILY_MATRIX:
                return X[..., node][..., 0]
            return X[..., node]
        raise TypeError(f'Unsupported terminal node type: {type(node)}')

    def _validate_typed_subtree(self, start_index):
        node = self.program[start_index]
        if not isinstance(node, TypedFunction):
            output_type = self._node_output_type(node)
            return start_index + 1, output_type is not None, output_type

        next_index = start_index + 1
        child_output_types = []
        for expected_type in node.input_types:
            next_index, is_valid, output_type = self._validate_typed_subtree(next_index)
            if not is_valid or output_type != expected_type:
                return next_index, False, node.output_type
            child_output_types.append(output_type)
        return next_index, True, node.output_type

    def build_program(self, random_state):
        """Build a naive random program.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        if self.init_method == 'half and half':
            method = ('full' if random_state.randint(2) else 'grow')
        else:
            method = self.init_method
        max_depth = random_state.randint(*self.init_depth)

        # Start a program with a function to avoid degenerative programs
        if self.typed_root_reducer_mode:
            root_functions = [
                function for function in self.function_set
                if isinstance(function, TypedFunction) and function.output_type == DAILY_MATRIX
            ]
            if not root_functions:
                raise ValueError('typed_root_reducer_mode requires at least one daily-output function')
            function = root_functions[random_state.randint(len(root_functions))]
            program = [function]
            for child_type in function.input_types:
                program.extend(self._build_typed_subtree(random_state, child_type, 1, max_depth, method))
            return program
        else:
            function = random_state.randint(len(self.function_set))
            function = self.function_set[function]
        program = [function]
        terminal_stack = [function.arity]

        while terminal_stack:
            depth = len(terminal_stack)
            choice = self.n_features + len(self.function_set)
            choice = random_state.randint(choice)
            # Determine if we are adding a function or terminal
            if (depth < max_depth) and (method == 'full' or
                                        choice <= len(self.function_set)):
                function = random_state.randint(len(self.function_set))
                function = self.function_set[function]
                program.append(function)
                terminal_stack.append(function.arity)
            else:
                # We need a terminal, add a variable or constant
                if self.const_range is not None:
                    terminal = random_state.randint(self.n_features + 1)
                else:
                    terminal = random_state.randint(self.n_features)
                if terminal == self.n_features:
                    terminal = random_state.uniform(*self.const_range)
                    if self.const_range is None:
                        # We should never get here
                        raise ValueError('A constant was produced with '
                                         'const_range=None.')
                program.append(terminal)
                terminal_stack[-1] -= 1
                while terminal_stack[-1] == 0:
                    terminal_stack.pop()
                    if not terminal_stack:
                        return program
                    terminal_stack[-1] -= 1

        # We should never get here
        return None

    def validate_program(self):
        """Rough check that the embedded program in the object is valid."""
        if self.typed_root_reducer_mode:
            if not self.program:
                return False
            if not isinstance(self.program[0], TypedFunction) or self.program[0].output_type != DAILY_MATRIX:
                return False
            end_index, is_valid, output_type, _ = self._typed_subtree_info(self.program, 0)
            return bool(is_valid and end_index == len(self.program) and output_type == DAILY_MATRIX)

        terminals = [0]
        for index, node in enumerate(self.program):
            if isinstance(node, (_Function, TypedFunction)):
                terminals.append(node.arity)
            else:
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
        return terminals == [-1]

    def __str__(self):
        """Overloads `print` output of the object to resemble a LISP tree."""
        terminals = [0]
        output = ''
        for i, node in enumerate(self.program):
            if isinstance(node, (_Function, TypedFunction)):
                terminals.append(node.arity)
                output += node.name + '('
            else:
                if isinstance(node, int):
                    if self.feature_names is None:
                        output += 'X%s' % node
                    else:
                        output += self.feature_names[node]
                else:
                    output += '%.3f' % node
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
                    output += ')'
                if i != len(self.program) - 1:
                    output += ', '
        return output

    def export_graphviz(self, fade_nodes=None):
        """Returns a string, Graphviz script for visualizing the program.

        Parameters
        ----------
        fade_nodes : list, optional
            A list of node indices to fade out for showing which were removed
            during evolution.

        Returns
        -------
        output : string
            The Graphviz script to plot the tree representation of the program.

        """
        terminals = []
        if fade_nodes is None:
            fade_nodes = []
        output = 'digraph program {\nnode [style=filled]\n'
        for i, node in enumerate(self.program):
            fill = '#cecece'
            if isinstance(node, (_Function, TypedFunction)):
                if i not in fade_nodes:
                    fill = '#136ed4'
                terminals.append([node.arity, i])
                output += ('%d [label="%s", fillcolor="%s"] ;\n'
                           % (i, node.name, fill))
            else:
                if i not in fade_nodes:
                    fill = '#60a6f6'
                if isinstance(node, int):
                    if self.feature_names is None:
                        feature_name = 'X%s' % node
                    else:
                        feature_name = self.feature_names[node]
                    output += ('%d [label="%s", fillcolor="%s"] ;\n'
                               % (i, feature_name, fill))
                else:
                    output += ('%d [label="%.3f", fillcolor="%s"] ;\n'
                               % (i, node, fill))
                if i == 0:
                    # A degenerative program of only one node
                    return output + '}'
                terminals[-1][0] -= 1
                terminals[-1].append(i)
                while terminals[-1][0] == 0:
                    output += '%d -> %d ;\n' % (terminals[-1][1],
                                                terminals[-1][-1])
                    terminals[-1].pop()
                    if len(terminals[-1]) == 2:
                        parent = terminals[-1][-1]
                        terminals.pop()
                        if not terminals:
                            return output + '}'
                        terminals[-1].append(parent)
                        terminals[-1][0] -= 1

        # We should never get here
        return None

    def _depth(self):
        """Calculates the maximum depth of the program tree."""
        terminals = [0]
        depth = 1
        for node in self.program:
            if isinstance(node, (_Function, TypedFunction)):
                terminals.append(node.arity)
                depth = max(len(terminals), depth)
            else:
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
        return depth - 1

    def _length(self):
        """Calculates the number of functions and terminals in the program."""
        return len(self.program)

    def execute(self, X):
        """Execute the program according to X.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        y_hats : array-like, shape = [n_samples]
            The result of executing the program on X.

        Notes
        -----
        In 3D typed mode, terminals tagged as `daily_matrix` are materialized
        as 2D daily matrices with shape `(date, stock)`. Minute terminals keep
        the minute axis with shape `(date, stock, minute)`.

        """
        # Check for single-node programs
        node = self.program[0]
        if isinstance(node, (float, int)):
            return self._terminal_runtime_value(X, node)

        apply_stack = []

        for node in self.program:

            if isinstance(node, (_Function, TypedFunction)):
                apply_stack.append([node])
            else:
                # Lazily evaluate later
                apply_stack[-1].append(node)

            while len(apply_stack[-1]) == apply_stack[-1][0].arity + 1:
                # Apply functions that have sufficient arguments
                function = apply_stack[-1][0]
                terminals = [self._terminal_runtime_value(X, t) if isinstance(t, (float, int)) else t for t in apply_stack[-1][1:]]
                intermediate_result = function(*terminals)
                if len(apply_stack) != 1:
                    apply_stack.pop()
                    apply_stack[-1].append(intermediate_result)
                else:
                    return intermediate_result

        # We should never get here
        return None

    def get_all_indices(self, n_samples=None, max_samples=None,
                        random_state=None):
        """Get the indices on which to evaluate the fitness of a program.

        Parameters
        ----------
        n_samples : int
            The number of samples.

        max_samples : int
            The maximum number of samples to use.

        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        indices : array-like, shape = [n_samples]
            The in-sample indices.

        not_indices : array-like, shape = [n_samples]
            The out-of-sample indices.

        """
        if self._indices_state is None and random_state is None:
            raise ValueError('The program has not been evaluated for fitness '
                             'yet, indices not available.')

        if n_samples is not None and self._n_samples is None:
            self._n_samples = n_samples
        if max_samples is not None and self._max_samples is None:
            self._max_samples = max_samples
        if random_state is not None and self._indices_state is None:
            self._indices_state = random_state.get_state()

        indices_state = check_random_state(None)
        indices_state.set_state(self._indices_state)

        not_indices = sample_without_replacement(
            self._n_samples,
            self._n_samples - self._max_samples,
            random_state=indices_state)
        sample_counts = np.bincount(not_indices, minlength=self._n_samples)
        indices = np.where(sample_counts == 0)[0]

        return indices, not_indices

    def _indices(self):
        """Get the indices used to measure the program's fitness."""
        return self.get_all_indices()[0]

    def raw_fitness(self, X, y, sample_weight):
        """Evaluate the raw fitness of the program according to X, y.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        sample_weight : array-like, shape = [n_samples]
            Weights applied to individual samples.

        Returns
        -------
        raw_fitness : float
            The raw fitness of the program.

        """
        y_pred = self.execute(X)
        if self.transformer:
            y_pred = self.transformer(y_pred)
        raw_fitness = self.metric(y, y_pred, sample_weight)

        return raw_fitness

    def fitness(self, parsimony_coefficient=None):
        """Evaluate the penalized fitness of the program according to X, y.

        Parameters
        ----------
        parsimony_coefficient : float, optional
            If automatic parsimony is being used, the computed value according
            to the population. Otherwise the initialized value is used.

        Returns
        -------
        fitness : float
            The penalized fitness of the program.

        """
        if parsimony_coefficient is None:
            parsimony_coefficient = self.parsimony_coefficient
        penalty = parsimony_coefficient * len(self.program) * self.metric.sign
        return self.raw_fitness_ - penalty

    def get_subtree(self, random_state, program=None, expected_output_type=None, allow_root=True):
        """Get a random subtree from the program.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        program : list, optional (default=None)
            The flattened tree representation of the program. If None, the
            embedded tree in the object will be used.

        Returns
        -------
        start, end : tuple of two ints
            The indices of the start and end of the random subtree.

        """
        if program is None:
            program = self.program

        if self.typed_root_reducer_mode:
            span = self._select_typed_subtree(
                random_state,
                program=program,
                expected_output_type=expected_output_type,
                allow_root=allow_root,
            )
            if span is None:
                raise ValueError('No valid typed subtree available for selection')
            return span[0], span[1]

        # Choice of crossover points follows Koza's (1992) widely used approach
        # of choosing functions 90% of the time and leaves 10% of the time.
        probs = np.array([0.9 if isinstance(node, (_Function, TypedFunction)) else 0.1
                          for node in program])
        probs = np.cumsum(probs / probs.sum())
        start = np.searchsorted(probs, random_state.uniform())

        stack = 1
        end = start
        while stack > end - start:
            node = program[end]
            if isinstance(node, (_Function, TypedFunction)):
                stack += node.arity
            end += 1

        return start, end

    def reproduce(self):
        """Return a copy of the embedded program."""
        return copy(self.program)

    def crossover(self, donor, random_state):
        """Perform the crossover genetic operation on the program.

        Crossover selects a random subtree from the embedded program to be
        replaced. A donor also has a subtree selected at random and this is
        inserted into the original parent to form an offspring.

        Parameters
        ----------
        donor : list
            The flattened tree representation of the donor program.

        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        if self.typed_root_reducer_mode:
            parent_spans = self._typed_subtree_spans(self.program)
            donor_spans = self._typed_subtree_spans(donor)
            if not parent_spans or not donor_spans:
                return self.reproduce(), [], []

            common_types = tuple(sorted({span[2] for span in parent_spans}.intersection({span[2] for span in donor_spans})))
            if not common_types:
                return self.reproduce(), [], []

            target_output_type = common_types[random_state.randint(len(common_types))]
            parent_candidates = [span for span in parent_spans if span[2] == target_output_type]
            donor_candidates = [span for span in donor_spans if span[2] == target_output_type]
            parent_span = self._sample_typed_span(random_state, parent_candidates)
            donor_span = self._sample_typed_span(random_state, donor_candidates)
            if parent_span is None or donor_span is None:
                return self.reproduce(), [], []

            start, end, _, _ = parent_span
            donor_start, donor_end, _, _ = donor_span
        else:
            # Get a subtree to replace
            start, end = self.get_subtree(random_state)
            # Get a subtree to donate
            donor_start, donor_end = self.get_subtree(random_state, donor)

        removed = range(start, end)
        donor_removed = list(set(range(len(donor))) -
                             set(range(donor_start, donor_end)))
        # Insert genetic material from donor
        return (self.program[:start] +
                donor[donor_start:donor_end] +
                self.program[end:]), removed, donor_removed

    def subtree_mutation(self, random_state):
        """Perform the subtree mutation operation on the program.

        Subtree mutation selects a random subtree from the embedded program to
        be replaced. A donor subtree is generated at random and this is
        inserted into the original parent to form an offspring. This
        implementation uses the "headless chicken" method where the donor
        subtree is grown using the initialization methods and a subtree of it
        is selected to be donated to the parent.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        if self.typed_root_reducer_mode:
            parent_span = self._select_typed_subtree(random_state, self.program)
            if parent_span is None:
                return self.reproduce(), [], []
            start, end, expected_output_type, _ = parent_span
            method = ('full' if random_state.randint(2) else 'grow') if self.init_method == 'half and half' else self.init_method
            max_depth = random_state.randint(*self.init_depth)
            chicken = self._build_typed_subtree(random_state, expected_output_type, 0, max_depth, method)
            donor_span = self._select_typed_subtree(random_state, chicken, expected_output_type=expected_output_type)
            if donor_span is None:
                return self.reproduce(), [], []
            donor_start, donor_end, _, _ = donor_span
            removed = range(start, end)
            donor_removed = list(set(range(len(chicken))) -
                                 set(range(donor_start, donor_end)))
            return (self.program[:start] +
                    chicken[donor_start:donor_end] +
                    self.program[end:]), removed, donor_removed

        # Build a new naive program
        chicken = self.build_program(random_state)
        # Do subtree mutation via the headless chicken method!
        return self.crossover(chicken, random_state)

    def hoist_mutation(self, random_state):
        """Perform the hoist mutation operation on the program.

        Hoist mutation selects a random subtree from the embedded program to
        be replaced. A random subtree of that subtree is then selected and this
        is 'hoisted' into the original subtrees location to form an offspring.
        This method helps to control bloat.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        if self.typed_root_reducer_mode:
            outer_span = self._select_typed_subtree(random_state, self.program)
            if outer_span is None:
                return self.reproduce(), []
            start, end, outer_output_type, _ = outer_span
            subtree = self.program[start:end]
            inner_spans = self._typed_subtree_spans(subtree)
            inner_spans = [span for span in inner_spans if span[2] == outer_output_type]
            non_root_inner_spans = [span for span in inner_spans if span[0] != 0]
            if non_root_inner_spans:
                inner_spans = non_root_inner_spans
            if not inner_spans:
                return self.reproduce(), []
            weights = np.array([0.9 if span[3] else 0.1 for span in inner_spans], dtype=float)
            probs = np.cumsum(weights / weights.sum())
            choice = inner_spans[np.searchsorted(probs, random_state.uniform())]
            sub_start, sub_end, _, _ = choice
            hoist = subtree[sub_start:sub_end]
            removed = list(set(range(start, end)) -
                           set(range(start + sub_start, start + sub_end)))
            return self.program[:start] + hoist + self.program[end:], removed

        # Get a subtree to replace
        start, end = self.get_subtree(random_state)
        subtree = self.program[start:end]
        # Get a subtree of the subtree to hoist
        sub_start, sub_end = self.get_subtree(random_state, subtree)
        hoist = subtree[sub_start:sub_end]
        # Determine which nodes were removed for plotting
        removed = list(set(range(start, end)) -
                       set(range(start + sub_start, start + sub_end)))
        return self.program[:start] + hoist + self.program[end:], removed

    def point_mutation(self, random_state):
        """Perform the point mutation operation on the program.

        Point mutation selects random nodes from the embedded program to be
        replaced. Terminals are replaced by other terminals and functions are
        replaced by other functions that require the same number of arguments
        as the original node. The resulting tree forms an offspring.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        program = copy(self.program)

        # Get the nodes to modify
        mutate = np.where(random_state.uniform(size=len(program)) <
                          self.p_point_replace)[0]

        for node in mutate:
            if isinstance(program[node], (_Function, TypedFunction)):
                arity = program[node].arity
                if self.typed_root_reducer_mode:
                    candidates = self._typed_replacement_candidates(program[node])
                    if not candidates:
                        continue
                    replacement = candidates[random_state.randint(len(candidates))]
                else:
                    replacement = len(self.arities[arity])
                    replacement = random_state.randint(replacement)
                    replacement = self.arities[arity][replacement]
                program[node] = replacement
            else:
                # We've got a terminal, add a const or variable
                if self.typed_root_reducer_mode:
                    expected_type = self._node_output_type(program[node])
                    if expected_type is None:
                        continue
                    terminal = self._sample_typed_terminal(random_state, expected_type)
                elif self.const_range is not None:
                    terminal = random_state.randint(self.n_features + 1)
                else:
                    terminal = random_state.randint(self.n_features)
                if terminal == self.n_features:
                    terminal = random_state.uniform(*self.const_range)
                    if self.const_range is None:
                        # We should never get here
                        raise ValueError('A constant was produced with '
                                         'const_range=None.')
                program[node] = terminal

        return program, list(mutate)

    depth_ = property(_depth)
    length_ = property(_length)
    indices_ = property(_indices)
