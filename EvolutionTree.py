import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
class EvolutionTree:
    def __init__(self, depth=2, sigma_0=1.0, sigma=1.0, delta=0.1, lower=0.5, upper=2.0):
        self.delta = delta  # Observation noise variance    
        self.depth = depth  # Depth of the tree
        self.num_leaves = 2 ** depth  # Number of leaf nodes, ensuring a full binary tree
        self.sigma_0 = sigma_0  # Variance of the root node
        self.sigma = sigma  # Variance scaling parameter for branches
        # self.delta = delta  # Observation noise variance
        self.lower = lower  # Lower bound for branch lengths
        self.upper = upper  # Upper bound for branch lengths
        self.tree = self._build_tree()
        self.branch_lengths = self._assign_branch_lengths()
        self.mu = 0.0  # Mean value for all nodes
        self.X = None  # True values for all nodes
        # self.Y = None  # Observed values for all nodes
        self.X_obs = None  # Observed values at leaf nodes

    def _build_tree(self):
        # Construct a full binary tree with the given depth
        tree = nx.balanced_tree(r=2, h=self.depth)
        return tree

    def _assign_branch_lengths(self, lower=0.01, upper=2.0):
        # Assign random branch lengths to each edge
        branch_lengths = {}
        for edge in self.tree.edges():
            branch_lengths[edge] = np.random.uniform(lower, upper)
        return branch_lengths
    
    def find_lowest_common_ancestor(self, i, j):
        # Find the lowest common ancestor (LCA) of nodes i and j
        path_i = nx.shortest_path(self.tree, source=0, target=i)
        path_j = nx.shortest_path(self.tree, source=0, target=j)
        lca = None
        for u, v in zip(path_i, path_j):
            if u == v:
                lca = u
            else:
                break
        return lca
    
    def compute_covariance(self, i, j):
        # Compute the covariance between nodes i and j based on the LCA
        lca = self.find_lowest_common_ancestor(i, j)
        path_to_lca = nx.shortest_path(self.tree, source=0, target=lca)
        variance_lca = self.sigma_0 ** 2 + self.sigma ** 2 * sum(
            self.branch_lengths[(path_to_lca[k], path_to_lca[k + 1])] for k in range(len(path_to_lca) - 1))
        return variance_lca
    
    def validate_covariance(self):
        # Validate the covariance formula for all pairs of leaves
        leaves = [node for node in self.tree.nodes() if self.tree.degree(node) == 1 and node != 0]
        for i in range(len(leaves)):
            for j in range(i, len(leaves)):
                cov_ij = self.compute_covariance(leaves[i], leaves[j])
                print(f"Covariance between leaf {leaves[i]} and leaf {leaves[j]}: {cov_ij}")

    def generate_data(self):
        # Generate the true values X for all nodes
        X_dict = {}
        root = 0
        X_dict[root] = np.random.normal(self.mu, self.sigma_0)
        
        # Perform a BFS to generate values for the rest of the nodes
        for node in nx.bfs_tree(self.tree, root):
            for child in self.tree.neighbors(node):
                if child not in X_dict:  # Ensure child is not already visited
                    branch_length = self.branch_lengths[(node, child)]
                    X_dict[child] = X_dict[node] + np.random.normal(0, self.sigma * np.sqrt(branch_length))
        
        # Store true values for all nodes
        self.X = np.array([X_dict[node] for node in self.tree.nodes()])
        
        # Extract leaf nodes
        leaves = [node for node in self.tree.nodes() if self.tree.degree(node) == 1 and node != 0]
        X_leaf = np.array([X_dict[leaf] for leaf in leaves])
        
        # Generate observed values Y by adding Gaussian noise
        # self.X_obs = X_leaf + np.random.normal(0, self.delta, size=X_leaf.shape)
        # self.Y = self.X.copy()
        # self.Y[leaves] = self.Y_obs
        self.X_obs = X_leaf
        
        return self.X, self.X_obs, leaves


    def plot_tree(self):
        # Plot the tree structure
        pos = nx.spring_layout(self.tree)
        nx.draw(self.tree, pos, with_labels=True, node_size=500, node_color="lightblue")
        labels = {edge: f'{length:.2f}' for edge, length in self.branch_lengths.items()}
        nx.draw_networkx_edge_labels(self.tree, pos, edge_labels=labels)
        plt.show()