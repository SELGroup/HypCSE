"""Tree traversal util functions."""
import networkx as nx

def descendants_traversal(tree):
    """Get all descendants non-recursively, in traversal order."""
    n = len(list(tree.nodes()))
    root = n - 1

    traversal = []

    children = [list(tree.neighbors(node)) for node in range(n)]  # children remaining to process
    is_leaf = [len(children[node]) == 0 for node in range(n)]
    stack = [root]
    while len(stack) > 0:
        node = stack[-1]
        if len(children[node]) > 0:
            stack.append(children[node].pop())
        else:
            assert node == stack.pop()
            if is_leaf[node]:
                traversal.append(node)

    return traversal[::-1]


def descendants_count(tree):
    """For every node, count its number of descendant leaves, and the number of leaves before it."""
    n = len(list(tree.nodes()))
    root = n - 1

    left = [0] * n
    desc = [0] * n
    leaf_idx = 0

    children = [list(tree.neighbors(node))[::-1] for node in range(n)]  # children remaining to process
    stack = [root]
    while len(stack) > 0:
        node = stack[-1]
        if len(children[node]) > 0:
            stack.append(children[node].pop())
        else:
            children_ = list(tree.neighbors(node))

            if len(children_) == 0:
                desc[node] = 1
                left[node] = leaf_idx
                leaf_idx += 1
            else:
                desc[node] = sum([desc[c] for c in children_])
                left[node] = left[children_[0]]
            assert node == stack.pop()

    return desc, left

def descendants_volume(tree, degrees):
    """For every node, count its volume (degree of descendant leaves), and the number of leaves before it."""
    n = len(list(tree.nodes()))
    root = n - 1

    left = [0] * n
    desc = [0] * n
    volume_desc = [0] * n
    leaf_idx = 0
    leaf_volume = 0

    children = [list(tree.neighbors(node))[::-1] for node in range(n)]
    # print(children)
    # exit(0)
    stack = [root]
    while len(stack) > 0:
        node = stack[-1]
        # print(node)
        if len(children[node]) > 0:
            stack.append(children[node].pop())
        else:
            children_ = list(tree.neighbors(node))

            if len(children_) == 0:
                desc[node] = 1
                volume_desc[node] = degrees[node]
                # left[node] = leaf_volume
                # leaf_volume += degrees[node]
                left[node] = leaf_idx
                leaf_idx += 1
            else:
                desc[node] = sum([desc[c] for c in children_])
                volume_desc[node] = sum(volume_desc[c] for c in children_)
                left[node] = left[children_[0]]
            assert node == stack.pop()
    # print(volume_desc)
    # print(desc)
    # print(left)
    return volume_desc, desc, left

def get_leaves_root(tree):
    leaves = [x for x in tree.nodes() if len(list(tree.neighbors(x))) == 0]
    reversed_tree = nx.reverse_view(tree)
    root = [x for x in reversed_tree.nodes() if len(list(reversed_tree.neighbors(x))) == 0][0]
    return leaves, root