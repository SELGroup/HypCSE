

import numpy as np

#from mst import reorder
from mst import mst
# from mst import setup
from utils.tree import descendants_traversal, descendants_count, descendants_volume
import utils.tree as tree_utils
import itertools
from tqdm import tqdm


def dasgupta_cost_iterative(tree, similarities):
    """ Non-recursive version of DC. Also works on non-binary trees """
    n = len(list(tree.nodes()))
    root = n - 1

    cost = [0] * n

    desc = [None] * n  # intermediate computation: children of node

    children = [list(tree.neighbors(node)) for node in range(n)]  # children remaining to process
    stack = [root]
    while len(stack) > 0:
        node = stack[-1]
        if len(children[node]) > 0:
            stack.append(children[node].pop())
        else:
            children_ = list(tree.neighbors(node))

            if len(children_) == 0:
                desc[node] = [node]

            else:
                # Intermediate computations
                desc[node] = [d for c in children_ for d in desc[c]]

                # Cost at this node
                # cost_ = similarities[desc[node]].T[desc[node]].sum()
                # cost_ -= sum([similarities[desc[c]].T[desc[c]].sum() for c in children_])
                # cost_ = cost_ / 2.0
                # This is much faster for imbalanced trees
                cost_ = sum([similarities[desc[c0]].T[desc[c1]].sum() for i, c0 in enumerate(children_) for c1 in
                             children_[i + 1:]])
                cost_ *= len(desc[node])

                cost[node] = cost_ + sum([cost[c] for c in children_])  # recursive cost

                # Free intermediate computations (otherwise, up to n^2 space for recursive descendants)
                for c in children_:
                    desc[c] = None

            assert node == stack.pop()
    return 2 * cost[root]


def dasgupta_cost(tree, similarities):
    """ Non-recursive version of DC for binary trees.

    Optimized for speed by reordering similarity matrix for locality
    """
    n = len(list(tree.nodes()))
    root = n - 1
    n_leaves = len(similarities)

    leaves = descendants_traversal(tree)
    n_desc, left_desc = descendants_count(tree)

    cost = [0] * n  # local cost for every node

    # reorder similarity matrix for locality
    # similarities = similarities[leaves].T[leaves] # this is the bottleneck; is there a faster way?
    similarities = mst.reorder(similarities, np.array(leaves), n_leaves)  # this is the bottleneck; is there a faster way?

    # Recursive computation
    children = [list(tree.neighbors(node)) for node in range(n)]  # children remaining to process
    stack = [root]
    while len(stack) > 0:
        node = stack[-1]
        if len(children[node]) > 0:
            stack.append(children[node].pop())
        else:
            children_ = list(tree.neighbors(node))

            if len(children_) < 2:
                pass
            elif len(children_) == 2:
                left_c = children_[0]
                right_c = children_[1]

                left_range = [left_desc[left_c], left_desc[left_c] + n_desc[left_c]]
                right_range = [left_desc[right_c], left_desc[right_c] + n_desc[right_c]]
                cost_ = np.add.reduceat(
                    np.add.reduceat(
                        similarities[
                        left_range[0]:left_range[1],
                        right_range[0]:right_range[1]
                        ], [0], axis=1
                    ), [0], axis=0
                )
                cost[node] = cost_[0, 0]

            else:
                assert False, "tree must be binary"
            assert node == stack.pop()

    return 2 * sum(np.array(cost) * np.array(n_desc))

def se_cost_iterative(tree, similarities, degrees=None):
    similarities = similarities - np.diag(np.diag(similarities))
    if degrees is None:
        degrees = np.sum(similarities, axis=-1)
    n = len(list(tree.nodes()))
    root = n - 1
    cost = [0] * n
    desc = [None] * n

    children = [list(tree.neighbors(node)) for node in range(n)]  # children remaining to process
    stack = [root]
    while len(stack) > 0:
        node = stack[-1]
        if len(children[node]) > 0:
            stack.append(children[node].pop())
        else:
            children_ = list(tree.neighbors(node))
            if len(children_) == 0:
                desc[node] = [node]
            else:
                desc[node] = [d for c in children_ for d in desc[c]]

                cost_ = sum([similarities[desc[c0]].T[desc[c1]].sum() for i, c0 in enumerate(children_) for c1 in
                             children_[i + 1:]])
                vol_ = sum(degrees[desc[node]])
                cost_ *= np.log2(vol_)

                cost[node] = cost_ + sum([cost[c] for c in children_])

                # Free intermediate computations (otherwise, up to n^2 space for recursive descendants)
                for c in children_:
                    desc[c] = None

            assert node == stack.pop()
    # return 2 * cost[root]
    se = 2 * cost[root]
    volG = np.sum(similarities)
    for degree in degrees:
        se += - degree * np.log2(degree)
    se /= volG
    return se



def se_cost(tree, similarities, degrees=None):
    """ Non-recursive version of DC for binary trees.

    Optimized for speed by reordering similarity matrix for locality
    """
    if degrees is None:
        degrees = np.sum(similarities, axis=-1)
    n = len(list(tree.nodes()))
    root = n - 1
    n_leaves = len(similarities)

    leaves = descendants_traversal(tree)
    volume_desc, n_desc, left_desc = descendants_volume(tree, degrees)
    # print(type(volume_desc[0]), type(n_desc[0]), type(left_desc[0]))
    # print(len(volume_desc), len(n_desc), len(left_desc))

    cost = [0] * n

    similarities = mst.reorder(similarities, np.array(leaves), n_leaves)  # this is the bottleneck; is there a faster way?

    children = [list(tree.neighbors(node)) for node in range(n)]
    stack = [root]
    while len(stack) > 0:
        node = stack[-1]
        if len(children[node]) > 0:
            stack.append(children[node].pop())
        else:
            children_ = list(tree.neighbors(node))

            if len(children_) < 2:
                pass
            elif len(children_) == 2:
                left_c = children_[0]
                right_c = children_[1]

                left_range = [left_desc[left_c], left_desc[left_c] + n_desc[left_c]]
                right_range = [left_desc[right_c], left_desc[right_c] + n_desc[right_c]]
                cost_ = np.add.reduceat(
                    np.add.reduceat(
                        similarities[
                        left_range[0]:left_range[1],
                        right_range[0]:right_range[1]
                        ], [0], axis=1
                    ), [0], axis=0
                )
                cost[node] = cost_[0, 0]

            else:
                assert False, "tree must be binary"
            assert node == stack.pop()
    volG = np.sum(degrees)
    se = 2 * sum(np.array(cost) * np.log2(np.array(volume_desc)))
    for degree in degrees:
        se += - degree * np.log2(degree)
    se /= volG
    return se



def den_purity_recursive(tree, gt_clusters):
    """ The dendrogram purity formulation from the gHHC paper """

    all_classes = np.unique(gt_clusters)

    def _den_purity_(node):
        children = list(tree.neighbors(node))
        if len(children) == 0:
            class_count = {c: 0 for c in all_classes}
            class_count[gt_clusters[node]] = 1
            return 0, class_count, 1, 0.0
        elif len(children) == 1:
            return _den_purity_(children[0])
        # else:
        #     assert len(children) == 2, "Can only compute dendrogram purity on binary trees for now."

        # Recurse
        pair_counts, class_counts, leaf_counts, puritys = zip(*[_den_purity_(c) for c in children])

        leaf_count = sum(leaf_counts)
        class_count = {c: sum([child[c] for child in class_counts]) for c in all_classes}
        # new_pairs = {c: class_counts[0][c]*class_counts[1][c] for c in all_classes} # binary tree case
        new_pairs = {c: (class_count[c]**2 - sum([child[c]**2 for child in class_counts])) // 2 for c in all_classes}
        pair_count = sum(pair_counts) + sum([new_pairs[c] for c in all_classes])
        purity = sum(puritys) + sum([(class_count[c]/leaf_count)*new_pairs[c] for c in all_classes])
        return pair_count, class_count, leaf_count, purity

    leaves, root = tree_utils.get_leaves_root(tree)
    pair_count, class_count, leaf_count, purity = _den_purity_(root)
    return purity / pair_count


# @profile
def den_purity(tree, gt_clusters):
    """ The dendrogram purity formulation from the gHHC paper

    Stack-based analog of den_purity_recursive to avoid Python recursion limits
    """

    n = len(gt_clusters) * 2 - 1

    all_classes = np.unique(gt_clusters)
    _, root = tree_utils.get_leaves_root(tree)

    # print(n, root, leaves)

    children = [list(tree.neighbors(node)) for node in range(n)] # children remaining to process
    stack = [root]
    # Create the computation buffers leaf_count, purity for all nodes
    pair_count = [None] * n  # number of same-class pairs in subtree
    class_count = [None] * n # number of leaves in subtree
    leaf_count = [None] * n # number of leaves (sum of class counts)
    purity = [None] * n # purity of subtree (not normalized by # pairs)
    while len(stack) > 0:
        node = stack[-1]
        if len(children[node]) > 0:
            stack.append(children[node].pop())
        else:
            # Get children computations
            children_ = list(tree.neighbors(node))
            # Base case: node is a leaf
            if len(children_) == 0:
                pair_count[node] = 0
                class_count[node] = {c: 0 for c in all_classes}
                class_count[node][gt_clusters[node]] = 1
                leaf_count[node] = 1
                purity[node] = 0.0

            else:
                pair_counts = [pair_count[c] for c in children_]
                class_counts = [class_count[c] for c in children_]
                leaf_counts = [leaf_count[c] for c in children_]
                puritys = [purity[c] for c in children_]
                # Free children computations
                for c in children[node]:
                    pair_count[c] = class_count[c] = leaf_count[c] = purity[c] = None

                # Compute new info for this node
                leaf_count[node] = sum(leaf_counts)
                class_count[node] = {c: sum([child[c] for child in class_counts]) for c in all_classes}
                new_pairs = {c: (class_count[node][c]**2 - sum([child[c]**2 for child in class_counts])) // 2 for c in all_classes}
                pair_count[node] = sum(pair_counts) + sum([new_pairs[c] for c in all_classes])
                purity[node] = sum(puritys) + sum([(class_count[node][c]/leaf_count[node])*new_pairs[c] for c in all_classes])
                # class_count[node] = class_count

            assert node == stack.pop()

    return purity[root] / pair_count[root]


def dendrogram_purity(t, y):
    y = y.astype(int)
    y_onehot = np.zeros((y.shape[0], y.max()+1))
    y_onehot[np.arange(y.shape[0]), y] = 1
    cluster_dict = {}
    pairs = []
    for i in range(y_onehot.shape[1]):
        indicesi = np.argwhere(y_onehot[:,i] == 1).flatten()
        # print(indicesi)
        pairs_i = list(itertools.permutations(indicesi, 2))
        pairs.append(pairs_i)
        cluster_indices = np.argwhere(y_onehot[:,i] == 1).flatten().tolist()
        for j in indicesi:
            cluster_dict[j] = cluster_indices
    # exit(0)
    purity_list = []
    # for index, pairs_i in enumerate(pairs):
    for index, pairs_i in enumerate(tqdm(pairs)):
        for pair in pairs_i:
            # print(pair)
            i,j = pair
            nodei = t.search_nodes(name=i)
            nodej = t.search_nodes(name=j)
            # print(nodei,nodej)
            # assert len(nodei)==1
            # assert len(nodej)==1
            ancestor = t.get_common_ancestor(nodei[0],nodej[0])
            ancestor_leaves = [int(i.name) for i in ancestor.get_leaves()]
            cluster_indices = cluster_dict[i]
            # print("ancestor_leaves", ancestor_leaves)
            # print("cluster_indices", cluster_indices)
            purity = len(set(ancestor_leaves).intersection(cluster_indices)) / len(ancestor_leaves)
            purity_list.append(purity)
    return np.mean(purity_list)
    # print(np.mean(purity_list))
    # for pair in pairs:
    #     i,j = pair
    #     nodei = t.search_nodes(i)
    #     nodej = t.search_nodes(j)
    #     ancestor = t.get_common_ancestor(nodei, nodej)



def dendrogram_purity_expected(t, y, n_sample=1000):
    n_instance = y.shape[0]
    y = y.astype(int)
    y_onehot = np.zeros((y.shape[0], y.max() + 1))
    y_onehot[np.arange(y.shape[0]), y] = 1
    cluster_dict = {}
    for i in range(y_onehot.shape[1]):
        indicesi = np.argwhere(y_onehot[:, i] == 1).flatten()
        cluster_indices = np.argwhere(y_onehot[:, i] == 1).flatten().tolist()
        for j in indicesi:
            cluster_dict[j] = cluster_indices
    purity_list = []
    leaves_dict = {}
    for leaf in t.get_leaves():
        leaves_dict[int(leaf.name)] = leaf
    for index_sample in tqdm(range(n_sample)):
        nodeID_i = np.random.randint(n_instance)
        # clusterID = y[nodeID_i]
        cluster_indices = cluster_dict[nodeID_i]
        # print(nodeID_i, cluster_indices)
        # assert nodeID_i in cluster_indices
        nodeID_j = np.random.choice(cluster_indices)
        # nodei = t.search_nodes(name=nodeID_i)
        # nodej = t.search_nodes(name=nodeID_j)
        # print(i,j,nodei,nodej)
        nodei = leaves_dict[nodeID_i]
        nodej = leaves_dict[nodeID_j]
        ancestor = t.get_common_ancestor(nodei, nodej)
        ancestor_leaves = [int(i.name) for i in ancestor.get_leaves()]
        purity = len(set(ancestor_leaves).intersection(cluster_indices)) / len(ancestor_leaves)
        purity_list.append(purity)
    return np.mean(purity_list)



