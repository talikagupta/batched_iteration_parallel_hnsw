# Try to import CuPy for GPU acceleration; fall back to NumPy if unavailable
try:
    import cupy as cp
    xp = cp
    CUPY_ENABLED = True
except ImportError:
    import numpy as np
    xp = np
    CUPY_ENABLED = False

from math import log2
from random import random
from collections import defaultdict

class Node:
    def __init__(self, node_id, L, m, ef, ep, m0=None):
        self.node_id = node_id
        self._m = m
        self._m0 = 2 * m if m0 is None else m0
        self._level_mult = 1 / log2(m)
        self.l = int(-log2(random()) * self._level_mult) + 1
        # print(self.l)
        # SEARCH-LAYER state
        self.visited = set()
        self.W = set()
        self.candidates = set()
        self._ef = ef
        self.ep = [ep]
        self.current_layer = L
        self.level_stack = [i for i in range(L, self.l+1 + 1, -1)]
        self.final_W = {}
        self.search_phase = 1 # flag signifies we are in search phase 1 (where ef = 1) or search phase 2 (0 value)
        self.is_done = 0

class InsertScheduler:
    def __init__(self, X, graph, batch_size):
        self.X = X          # Dataset vectors
        self.graph = graph     # HNSW Graph, a dict mapping each layer to a graph 
        self.batch_size = batch_size

        self.active_nodes = []      # Nodes currently being processed
        self.waiting_queue = []     # Nodes waiting for being processed
        self.search_phase = SearchLayerPhase(ef=1)      # Search phase handler

    def add_node(self, node):
        # Add node to active set or waiting queue
        if len(self.active_nodes) < self.batch_size:
            self.active_nodes.append(node)
        else:
            self.waiting_queue.append(node)

    def step(self):
        # Run one search layer step for active nodes
        if self.active_nodes:
            self.search_phase.run(self.active_nodes, self.X, self.graph)

        # Remove completed nodes
        done_nodes = [s for s in self.active_nodes if s.is_done]
        self.active_nodes = [s for s in self.active_nodes if not s.is_done]

        # Fill batch from waiting queue
        available_slots = self.batch_size - len(self.active_nodes)
        for _ in range(available_slots):
            if self.waiting_queue:
                self.active_nodes.append(self.waiting_queue.pop())
        
        return done_nodes

    def run_until_done(self):
        # Keep running steps until all nodes are processed
        all_done = []
        while self.active_nodes or self.waiting_queue:
            done = self.step()
            all_done.extend(done)
        return all_done

class SearchLayerPhase:
    def __init__(self, ef):
        self.ef = ef        # ef parameter for candidate list size

    def run(self, active_nodes, X, graphs):
        # Performs one iteration of the search layer algorithm in HNSW.

        queries = [node.node_id for node in active_nodes]
        B = len(active_nodes)
        X_batch = X[queries]
        ef = self.ef

        # Step 1: Find nearest node in candidate set
        candidates = [node.candidates for node in active_nodes]
        all_cand, valid_mask = pad_sets(candidates, B)
        cand_vectors = X[all_cand.clip(min=0)]
        dist_q_to_C = batched_l2_distances(X_batch, cand_vectors)
        dist_q_to_C[~valid_mask] = xp.inf
        nearest_idxs = xp.argmin(dist_q_to_C, axis=1)
        nearest_nodes = all_cand[xp.arange(B), nearest_idxs]
        nearest_dists = dist_q_to_C[xp.arange(B), nearest_idxs]

        # Remove nearest nodes from candidate sets to prevent re-visiting
        for i in range(B):
            candidates[i].discard(int(nearest_nodes[i]))

        # Step 2: Compute distances to working set W
        W = [node.W for node in active_nodes]
        all_W, W_mask = pad_sets(W, B)
        dist_q_to_W = batched_l2_distances(X_batch, X[all_W.clip(min=0)])
        dist_q_to_W[~W_mask] = -1
        furthest_idxs = xp.argmax(dist_q_to_W, axis=1)
        furthest_dists = dist_q_to_W[xp.arange(B), furthest_idxs]

        # Step 3: Check if we should continue expanding
        continue_mask = nearest_dists <= furthest_dists

        for i, cont in enumerate(continue_mask):
            if not cont:
                node = active_nodes[i]
                node.final_W[node.current_layer] = node.W
                if node.level_stack: node.level_stack.pop(0)
                if node.level_stack:
                    node.current_layer = node.level_stack[0]
                    node.visited = set(node.ep)
                    node.candidates = set(node.ep)
                    node.W = set(node.ep)
                else:
                    node.is_done = 1

        # Step 4: Expand from nearest node
        expanding_nodes = [active_nodes[i] for i in range(B) if continue_mask[i]]
        expanding_nearest = [nearest_nodes[i] for i in range(B) if continue_mask[i]]
        expanding_idxs = [i for i, m in enumerate(continue_mask) if m]

        # Get unvisited neighbors of nearest nodes
        batch_nbrs = [
            list(graphs.get(node.current_layer, {}).get(int(n), set()) - node.visited)
            for node, n in zip(expanding_nodes, expanding_nearest)
        ]
        batch_nbrs_padded, nbrs_mask = pad_sets(batch_nbrs, len(expanding_nodes))

        nbrs_vecs = X[batch_nbrs_padded.clip(min=0)]
        dist_q_to_nbrs = batched_l2_distances(
            X[[n.node_id for n in expanding_nodes]],
            nbrs_vecs
        )
        dist_q_to_nbrs[~nbrs_mask] = xp.inf

        # Step 5: Accept neighbors closer than furthest in W
        furthest_expanding = furthest_dists[expanding_idxs]
        accept_mask = dist_q_to_nbrs < furthest_expanding[:, None]

        for i, node in enumerate(expanding_nodes):
            accepted = batch_nbrs_padded[i][accept_mask[i]]
            node.visited.update(batch_nbrs[i])
            node.candidates.update(map(int, accepted))
            node.W.update(map(int, accepted))

        # Step 6: Re-sort working set W and trim to ef
        W = [node.W for node in active_nodes]
        all_W, W_mask = pad_sets(W, B)
        dist_to_W = batched_l2_distances(X_batch, X[all_W.clip(min=0)])
        dist_to_W[~W_mask] = xp.inf

        sorted_idxs = xp.argsort(dist_to_W, axis=1)
        for i, node in enumerate(active_nodes):
            keep = sorted_idxs[i, :min(ef, int(W_mask[i].sum()))]
            sorted_nodes = all_W[i, keep]
            node.W = set(map(int, sorted_nodes))
            
            # Save the closest node as entry point for the next layer
            if node.search_phase == 1 and len(sorted_nodes) > 0:
                node.ep = [int(sorted_nodes[0])]
            elif node.search_phase == 0 and len(sorted_nodes) > 0:
                node.ep = list(node.W)

            if not node.candidates: # check for |C| > 0
                # transition to the next layer
                node.final_W[node.current_layer] = node.W
                if node.level_stack: node.level_stack.pop(0)
                if node.level_stack:
                    node.current_layer = node.level_stack[0]
                    node.visited = set(node.ep)
                    node.candidates = set(node.ep)
                    node.W = set(node.ep)
                else:
                    node.is_done = 1

def batched_l2_distances(X, B):
    """
        Compute Euclidean distances between a batch of vectors X and batch of candidates B. Each vector is of dimension D.

        Parameters:
        - X: shape (B, D)
        - B: shape (B, N, D)

        Returns:
        - distances: shape (B, N)
    """
    x_norm_sq = xp.sum(X**2, axis=1, keepdims=True)
    b_norm_sq = xp.sum(B**2, axis=2)
    cross_term = 2 * xp.einsum("bd,bnd->bn", X, B)
    dist_sq = x_norm_sq + b_norm_sq - cross_term
    return xp.sqrt(xp.maximum(dist_sq, 0))

def pad_sets(sets, B, pad_size=None):
    """
    Pad a list of variable-length sets to create a uniform 2D matrix and a mask indicating valid entries.

    Parameters:
    - sets: list of sets, each containing node IDs
    - B: int - Number of sets (typically batch size)
    - pad_size: int (optional) - Max size to pad to; inferred from largest set if not given

    Returns:
    - padded: array of shape (B, pad_size) with -1 for padding
    - mask: boolean array of shape (B, pad_size) with True for valid entries
    """
    if pad_size is None:
        pad_size = max(len(s) for s in sets) if sets else 0
    padded = xp.full((B, pad_size), -1, dtype=int)
    mask = xp.zeros((B, pad_size), dtype=bool)
    for i, s in enumerate(sets):
        items = xp.array(list(s), dtype=int)
        padded[i, :len(s)] = items
        mask[i, :len(s)] = True
    return padded, mask

# ------------------ TESTING -------------------

# # 2D dataset
# X = np.array([
#     [0.0, 0.0],   # 0
#     [1.0, 0.0],   # 1
#     [0.0, 1.0],   # 2
#     [1.0, 1.0],   # 3
#     [0.5, 0.5],   # 4 (entry point)
#     [2.0, 2.0],   # 5 (new node to insert)
#     [3.0, 2.0],   # 6 (new node to insert)
#     [0.0, 0.0],   # 7 (new node to insert)
# ])

# # Dummy layered graph: dictionary per layer
# graph = {
#     0: {  # Layer 0
#         0: {1, 2, 4},
#         1: {0, 3, 4},
#         2: {0, 3, 4},
#         3: {1, 2},
#         4: {2},
#         5: set()
#     },
#     1: {  # Layer 1
#         0: {4},
#         3: {4},
#         4: {0, 3}
#     }
# }



# scheduler = InsertScheduler(X, graph, batch_size=3)
# nodes = [Node(node_id=i+5, L=1, m=5, ef=4, ep=4) for i in range(3)]

# for node in nodes:
#     node.visited = {4}
#     node.candidates = {4}
#     node.W = {4}
#     scheduler.add_node(node)

# done_nodes = scheduler.run_until_done()
# done_node_states = [
#     {
#         "id": node.node_id,
#         "final_W": sorted(list(node.W)),
#         "visited": sorted(list(node.visited))
#     }
#     for node in done_nodes
# ]

# print(done_node_states)
# print("Running with:", "CuPy (GPU)" if CUPY_ENABLED else "NumPy (CPU)")


import time

# Parameters
N = 100000            # Total number of nodes in the dataset
D = 128              # Vector dimension
L = 4                # Number of HNSW layers
M = 32                # Maximum number of connections
INSERT_COUNT = 10000  # Number of new nodes to insert
BATCH_SIZE = 1000     # Insert scheduler batch size
EF = 32              # ef parameter for search
ENTRY_POINT = 0      # Use first node as entry point

# Create synthetic dataset
X = xp.random.randn(N, D).astype(xp.float32)
X = X / xp.linalg.norm(X, axis=1, keepdims=True)  # normalize for fair distances

# Random graph generation for each layer
def generate_random_hnsw_graph(num_nodes, num_layers, max_connections):
    graph = defaultdict(dict)
    
    for l in range(num_layers + 1):
        # Generate neighbor indices in batch
        all_neighbors = xp.random.randint(0, num_nodes, size=(num_nodes, 2 * max_connections))
        
        for i in range(num_nodes):
            nbrs = all_neighbors[i]
            # Remove self-connections and duplicates
            nbrs = xp.unique(nbrs[nbrs != i])[:max_connections]
            graph[l][i] = set(map(int, nbrs.get() if CUPY_ENABLED else nbrs))

    return graph

graph = generate_random_hnsw_graph(N - INSERT_COUNT, L, M)

# Create InsertScheduler and nodes to insert
scheduler = InsertScheduler(X, graph, batch_size=BATCH_SIZE)
start_id = N - INSERT_COUNT
new_nodes = [
    Node(node_id=i, L=L, m=M, ef=EF, ep=ENTRY_POINT)
    for i in range(start_id, N)
]

# Initialize each node with ep state
for node in new_nodes:
    node.visited = {ENTRY_POINT}
    node.candidates = {ENTRY_POINT}
    node.W = {ENTRY_POINT}
    scheduler.add_node(node)

# Measure insertion time
start_time = time.time()
scheduler.run_until_done()
end_time = time.time()

print(f"Insertion of {INSERT_COUNT} nodes completed in {end_time - start_time:.2f} seconds.")
print("Running on:", "CuPy (GPU)" if CUPY_ENABLED else "NumPy (CPU)")

