# Copyright (c) 2024 Magik Compute Pte. Ltd. All rights reserved.
import os
import math
import requests
import numpy as np
from collections import deque
from collections.abc import Generator
from abc import ABC, ABCMeta, abstractmethod
from typing import Any, Generic, TypeVar
from action import load_llm_config, MCTSAgent

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"  # For 4090*6

class State:
    __slots__ = ["problem", "answer"]

    def __init__(self, problem: str, answer: str):
        self.problem = problem
        self.answer = answer

class TreeNode(ABC):
    def __init__(self, state: State, parent=None, depth=0):
        super().__init__()

        self.state = state
        self.parent = parent
        self.children = []
        self.N = 0     # total visit count for the node
        self.Q = 0.0   # total reward of the node
        self.G = []    # list of rewards for the node
        self.depth = depth

        self._terminated = False

    def add_child(self, child_node):
        self.children.append(child_node)

    def update(self, reward):
        self.N += 1
        self.G.append(reward)
        # self.Q += (reward - self.Q) / self.N  # incremental moving average
        self.Q = (min(self.G) + np.mean(self.G)) / 2  # formula (2) in paper https://arxiv.org/pdf/2406.07394

    @property
    def is_fully_expanded(self, legal_actions: list[Any]) -> bool:
        return len(self.children) == len(legal_actions)
    
    @property
    def is_terminal(self) -> bool:
        return self._terminated
    
    def set_as_terminate_node(self):
        self._terminated = True


class MCTS(ABC):
    """Monte Carlo Tree Search"""
    def __init__(self, 
                 max_rollouts: int = 4, 
                 c: float = 1.414, 
                 max_children: int = 2,
                 eps: float = 1e-8,
                 reward_ub: int = 10,
                 reward_penalty: int = 0.,
                 default_uct_score: float = 1000,
                 dummy_answer: str = "I don't know.",
                 policy: str = 'GREEDY',
                 initialize_strategy: str = 'ZERO_SHOT',
                 num_turns: int = 1,
                 samples_per_node: int = 3,
                 llm_cfg: dict = None,
                ):
        self.max_rollouts = max_rollouts
        self.c = c
        self.max_children = max_children
        self.default_uct_score = default_uct_score
        self.eps = eps
        self.reward_ub = reward_ub
        self.reward_penalty = reward_penalty
        self.dummy_answer = dummy_answer
        self.policy = policy
        self.initialize_strategy = initialize_strategy
        self.num_turns = num_turns
        self.samples_per_node = samples_per_node

        # explored = expanded + simulated, i.e. has seen terminal at least once, i.e. we can calculate its UCT value, i.e. has Q and N
        self.explored_nodes = set()

        llm_cfg,_ = load_llm_config('http://localhost:8001/v1')

        self.agent = MCTSAgent(llm_cfg)
        self.evaluator_url = "http://localhost:8110/get_score"

    def calc_score(self, sys_prompt: str="", question: str="", response: str=""):
        payload = {'sys_prompt': sys_prompt,'question': question, 'response': response}
        r = requests.post(self.evaluator_url, json=payload, timeout=1000)
        return r.json()['score']

    def initialize(self, S: State) -> TreeNode:
        if self.initialize_strategy == 'ZERO_SHOT':
            root = TreeNode(state=State(problem=S.problem, answer=self.agent.zero_shot(S.problem)))
        elif self.initialize_strategy == 'DUMMY_ANSWER':
            root = TreeNode(S=State(problem=S.problem, answer=self.dummy_answer))
        else:
            raise ValueError(f"Initialize Strategy `{self.initialize_strategy}` does not exist")

        return root

    def forward(self, problem):
        S_best = self.search(S=State(problem=problem, answer=self.dummy_answer))
        return S_best.answer

    def is_terminal(self, node: TreeNode) -> bool:
        return len(node.children) >= self.max_children or any(child.Q > node.Q for child in node.children)

    def get_actions(self, S: State, action: str='CoT') -> list[str]:
        if action == 'CoT':
            return self.agent.zero_shot(S.problem)
        elif action == 'TIR':
            return self.agent.tir_answer(S.problem)
        else:
            raise ValueError(f"Action name `{action}` does not exist!")

    def get_next_state(self, S: State, action=None) -> State:
        cot_answer = self.get_actions(S=S, action='CoT')
        tir_answer = self.get_actions(S=S, action='TIR')
        cot_score = self.calc_score(sys_prompt=self.agent.cot_prompt, question=S.problem, response=cot_answer)
        tir_score = self.calc_score(sys_prompt=self.agent.tir_prompt, question=S.problem, response=tir_answer)

        current_answer = cot_answer if cot_score > tir_score else tir_answer
        S_next = State(problem=S.problem, answer=current_answer)
        return S_next

    def get_reward(self, S: State) -> int:
        reward = self.calc_score(question=S.problem, response=S.answer)
        return min(reward, self.reward_ub) - self.reward_penalty if reward > self.reward_ub else reward
    
    def search(self, S: State) -> State:
        root = self.initialize(S)
        for _ in range(self.max_rollouts):
            self.rollout(root)

        return self._best_child(root).state
    
    def rollout(self, node: TreeNode):
        leaf = self.select(node)
        child = self.expand(leaf)
        result = self.simulate(child)
        self.backpropagate(child, result)

    def select(self, root: TreeNode) -> TreeNode:
        children = [child for child in self._traverse_tree(root) if not self.is_terminal(child)]
        if not children:
            return root

        uct_scores = np.array([self._uct(child) for child in children])

        if self.policy == 'GREEDY':
            node = children[np.argmax(uct_scores)]
        elif self.policy == 'IMPORTANCE_SAMPLING':
            probabilities = uct_scores / np.sum(uct_scores)
            node = np.random.choice(children, p=probabilities)
        else:
            raise ValueError(f"Selection Policy `{self.policy}` does not exist")
        return node

    def expand(self, node: TreeNode) -> TreeNode:
        S_next = self.get_next_state(S=node.state)
        child = TreeNode(state=S_next, parent=node)
        node.add_child(child)
        return child

    def simulate(self, node: TreeNode) -> list[int]:
        rewards = [self.get_reward(S=node.state) for _ in range(self.samples_per_node)]
        # node.update(np.mean(rewards))
        return rewards

    def backpropagate(self, node: TreeNode, rewards: list[int] = None):
        while node.parent:
            node.parent.N += 1
            node.parent.Q = (node.parent.Q + max(child.Q for child in node.parent.children)) / 2
            node = node.parent

    def _uct(self, node: TreeNode) -> float:
        if not node.parent:
            return self.default_uct_score
        return node.Q + self.c * math.sqrt(math.log(node.parent.N + 1) / (node.N + self.eps))

    def _traverse_tree(self, root: TreeNode) -> Generator[TreeNode, None, None]:
        queue = deque([root])
        while queue:
            node = queue.popleft()
            yield node
            queue.extend(node.children)

    def _best_child(self, root: TreeNode) -> TreeNode:
        return max(self._traverse_tree(root), key=lambda node: node.Q)



def test_mcts(problem: str):
    mcts = MCTS()
    answer = mcts.forward(problem)
    print(answer)

if __name__ == '__main__':
    # problem = "Alice has 3 sisters and she also has 4 brothers. How many sisters does Alice’s brother have?"
    problem = "A box contains m black balls and n white balls. Each time I draw a white ball, I do not return it to the box. Conversely, if I draw a black ball, I return it to the box. What is the expected time to draw all n white balls?"
    test_mcts(problem)
