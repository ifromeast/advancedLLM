from __future__ import annotations

import math
import random
import numpy as np
from enum import Enum
from collections import deque
from collections.abc import Generator
from abc import ABC, ABCMeta, abstractmethod
from typing import Any, Generic, TypeVar, Union

State = TypeVar("State")
Action = TypeVar("Action")
Reward = TypeVar("Reward", bound=Union[float, Any])

class MCTSNode(Generic[State, Reward]):
    __slots__ = ["S", "parent", "children", "N", "Q", "G"]

    def __init__(self, S: State, parent: MCTSNode | None = None):
        self.S = S
        self.parent = parent
        self.children = []
        self.N = 0     # total visit count for the node
        self.Q = 0.0   # total reward of the node
        self.G = []    # list of rewards for the node

    def add_child(self, child_node: MCTSNode):
        self.children.append(child_node)

    def update(self, reward: Reward):
        self.N += 1
        self.G.append(reward)
        self.Q += (reward - self.Q) / self.N  # incremental moving average

    def is_fully_expanded(self, legal_actions: list[Any]) -> bool:
        return len(self.children) == len(legal_actions)


class MCTS(ABC, Generic[State, Action, Reward]):
    def __init__(self, max_rollouts: int = 4, c: float = 1.414, default_uct_score: float = float("inf"), *args, **kwargs):
        self.max_rollouts = max_rollouts
        self.c = c
        self.default_uct_score = default_uct_score

    @abstractmethod
    def get_actions(self, S: State) -> list[Action]:
        """Return a list of legal actions for the given state."""
        pass

    @abstractmethod
    def get_next_state(self, S: State, action: Action) -> State:
        """Return the next state after taking the given action."""
        pass

    @abstractmethod
    def is_terminal(self, S: State) -> bool:
        """Check if the given state is terminal."""
        pass

    @abstractmethod
    def get_reward(self, S: State) -> Reward:
        """Return the reward for the given state."""
        pass

    def initialize(self, S: State) -> MCTSNode:
        return MCTSNode(S=S)

    def search(self, S: State) -> State:
        root = self.initialize(S)
        for _ in range(self.max_rollouts):
            leaf = self.select(root)
            child = self.expand(leaf)
            result = self.simulate(child)
            self.backpropagate(child, result)
        return self._best_child(root).S

    def select(self, node: MCTSNode) -> MCTSNode:
        while not self.is_terminal(node.S):
            if not node.is_fully_expanded(self.get_actions(node.S)):
                return node
            node = self._select_child(node)
        return node

    def expand(self, node: MCTSNode) -> MCTSNode:
        actions = self.get_actions(node.S)
        unexpanded_actions = [
            action
            for action in actions
            if not any(child.S == self.get_next_state(node.S, action) for child in node.children)
        ]
        if unexpanded_actions:
            action = random.choice(unexpanded_actions)
            S_next = self.get_next_state(node.S, action)
            child = MCTSNode(S_next, parent=node)
            node.add_child(child)
            return child
        return node

    def simulate(self, node: MCTSNode) -> Reward:
        S_next = node.S
        while not self.is_terminal(S_next):
            action = self._simulate_policy(S_next)
            S_next = self.get_next_state(S_next, action)
        return self.get_reward(S_next)

    def _simulate_policy(self, S: State) -> Action:
        return random.choice(self.get_actions(S))  # pragma: no cover

    def backpropagate(self, node: MCTSNode, reward: Reward):
        while node:
            node.update(reward)
            node = node.parent

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        return max(node.children, key=lambda child: self._uct(child))

    def _uct(self, node: MCTSNode) -> float:
        if node.N == 0:
            return self.default_uct_score
        return (node.Q / node.N) + self.c * math.sqrt(math.log(node.parent.N) / node.N)

    def _best_child(self, node: MCTSNode) -> MCTSNode:
        return max(node.children, key=lambda child: child.N)


class MCTSrState:
    __slots__ = ["problem", "answer"]

    def __init__(self, problem: str, answer: str):
        self.problem = problem
        self.answer = answer


class MCTSrNode(MCTSNode):
    def __init__(self, S: MCTSrState, parent: MCTSrNode | None = None):
        super().__init__(S, parent)

    def update(self, reward: int):
        self.G.append(reward)
        self.Q = (min(self.G) + np.mean(self.G)) / 2  # formula (2) in paper https://arxiv.org/pdf/2406.07394


class Policy(Enum):
    GREEDY = 1
    IMPORTANCE_SAMPLING = 2

class InitializeStrategy(Enum):
    ZERO_SHOT = 1
    DUMMY_ANSWER = 2


import dspy
from .utils import parse_integer_answer

class ZeroShotAnswer(dspy.Signature):
    problem: str = dspy.InputField()
    answer: str = dspy.OutputField()


class CritiqueAnswer(dspy.Signature):
    problem: str = dspy.InputField()
    current_answer: str = dspy.InputField()
    critique: str = dspy.OutputField()


class RefineAnswer(dspy.Signature):
    """[[ ## proposed_instruction ## ]] Given a mathematical problem, a current answer, and a critique of that answer,
    refine the current answer to provide a more accurate and well-reasoned solution. Begin by carefully analyzing the
    problem and the critique, then think step by step to derive the correct answer. Ensure that your reasoning is clear
    and logical, and that the final answer is justified by the steps taken.

    [[ ## completed ## ]]
    """

    problem: str = dspy.InputField()
    current_answer: str = dspy.InputField()
    critique: str = dspy.InputField()
    answer: str = dspy.OutputField()


class EvaluateAnswer(dspy.Signature):
    problem: str = dspy.InputField()
    answer: str = dspy.InputField()
    score: int = dspy.OutputField(ge=-100, le=100)


class ZeroShotCoT(dspy.Module):
    def __init__(self):
        self.cot = dspy.TypedChainOfThought(ZeroShotAnswer)

    def forward(self, problem) -> dspy.Prediction:
        return dspy.Prediction(answer=self.cot(problem=problem).answer)
    
class MultipleTurnSelfRefine(dspy.Module):
    def __init__(self, num_turns: int = 1):
        super().__init__()
        self.zero_shot_cot = ZeroShotCoT()
        self.critique_answer = dspy.TypedChainOfThought(CritiqueAnswer)
        self.refine_answer = dspy.TypedChainOfThought(RefineAnswer)
        self.num_turns = num_turns

    def forward(self, problem) -> dspy.Prediction:
        current_answer = self.zero_shot_cot(problem=problem).answer

        for _ in range(self.num_turns):
            critique_result = self.critique_answer(problem=problem, current_answer=current_answer)
            refined_result = self.refine_answer(
                problem=problem, current_answer=current_answer, critique=critique_result.critique
            )
            current_answer = refined_result.answer

        return dspy.Prediction(answer=current_answer)

class MCTSMeta(ABCMeta):
    pass


class ModuleMeta(type(dspy.Module)):
    pass


class CombinedMeta(MCTSMeta, ModuleMeta):
    pass

# paper https://arxiv.org/pdf/2406.07394
class MCTSr(MCTS, dspy.Module, metaclass=CombinedMeta):
    def __init__(
        self,
        max_rollouts: int = 4,
        c: float = math.sqrt(2),
        max_children: int = 2,
        eps: float = 1e-8,
        reward_ub: int = 95,
        reward_penalty: int = 50,
        default_uct_score: float = 1000,
        dummy_answer: str = "I don't know.",
        policy: Policy = Policy.GREEDY,
        initialize_strategy: InitializeStrategy = InitializeStrategy.DUMMY_ANSWER,
        num_turns: int = 1,
        samples_per_node: int = 3,
    ):
        MCTS.__init__(self, max_rollouts=max_rollouts, c=c, default_uct_score=default_uct_score)
        dspy.Module.__init__(self)
        self.max_children = max_children
        self.eps = eps
        self.reward_ub = reward_ub
        self.reward_penalty = reward_penalty
        self.dummy_answer = dummy_answer
        self.policy = policy
        self.num_turns = num_turns
        self.initialize_strategy = initialize_strategy
        self.samples_per_node = samples_per_node

        self.zero_shot = ZeroShotCoT()
        self.critique = dspy.TypedChainOfThought(CritiqueAnswer)
        self.evaluate = dspy.TypedChainOfThought(EvaluateAnswer)
        self.refine = dspy.TypedChainOfThought(RefineAnswer)

    def initialize(self, S: MCTSrState) -> MCTSrNode:
        if self.initialize_strategy == InitializeStrategy.ZERO_SHOT:
            root = MCTSrNode(S=MCTSrState(problem=S.problem, answer=self.zero_shot.forward(problem=S.problem).answer))
        elif self.initialize_strategy == InitializeStrategy.DUMMY_ANSWER:
            root = MCTSrNode(S=MCTSrState(problem=S.problem, answer=self.dummy_answer))
        else:
            raise ValueError(f"Initialize Strategy `{self.initialize_strategy}` does not exist")

        return root

    def forward(self, problem) -> dspy.Prediction:
        S_best = self.search(S=MCTSrState(problem=problem, answer=self.dummy_answer))
        return dspy.Prediction(answer=S_best.answer)

    def is_terminal(self, node: MCTSNode) -> bool:
        return len(node.children) >= self.max_children or any(child.Q > node.Q for child in node.children)

    def get_actions(self, S: MCTSrState) -> list[str]:
        pass

    def get_next_state(self, S: MCTSrState, action=None) -> MCTSrState:
        current_answer = S.answer
        for _ in range(self.num_turns):
            critique = self.critique(problem=S.problem, current_answer=current_answer).critique
            refined = self.refine(problem=S.problem, current_answer=current_answer, critique=critique)
            current_answer = refined.answer
        S_next = MCTSrState(problem=S.problem, answer=current_answer)
        return S_next

    def get_reward(self, S: MCTSrState) -> int:
        reward = self.evaluate(problem=S.problem, answer=S.answer).score
        reward = parse_integer_answer(reward) if not isinstance(reward, int) else reward
        return min(reward, self.reward_ub) - self.reward_penalty if reward > self.reward_ub else reward

    def select(self, root: MCTSrNode) -> MCTSrNode:
        children = [child for child in self._traverse_tree(root) if not self.is_terminal(child)]
        if not children:
            return root

        uct_scores = np.array([self._uct(child) for child in children])

        if self.policy == Policy.GREEDY:
            node = children[np.argmax(uct_scores)]
        elif self.policy == Policy.IMPORTANCE_SAMPLING:
            probabilities = uct_scores / np.sum(uct_scores)
            node = np.random.choice(children, p=probabilities)
        else:
            raise ValueError(f"Selection Policy `{self.policy}` does not exist")

        return node

    def expand(self, node: MCTSrNode) -> MCTSrNode:
        S_next = self.get_next_state(S=node.S)
        child = MCTSrNode(S=S_next, parent=node)
        node.add_child(child)
        return child

    def simulate(self, node: MCTSrNode) -> list[int]:
        rewards = [self.get_reward(S=node.S) for _ in range(self.samples_per_node)]
        node.update(np.mean(rewards))
        return rewards

    def backpropagate(self, node: MCTSrNode, rewards: list[int] = None):
        while node.parent:
            node.parent.N += 1
            node.parent.Q = (node.parent.Q + max(child.Q for child in node.parent.children)) / 2
            node = node.parent

    def _uct(self, node: MCTSrNode) -> float:
        if not node.parent:
            return self.default_uct_score
        return node.Q + self.c * math.sqrt(math.log(node.parent.N + 1) / (node.N + self.eps))

    def _traverse_tree(self, root: MCTSNode) -> Generator[MCTSNode, None, None]:
        queue = deque([root])
        while queue:
            node = queue.popleft()
            yield node
            queue.extend(node.children)

    def _best_child(self, root: MCTSNode) -> MCTSNode:
        return max(self._traverse_tree(root), key=lambda node: node.Q)
