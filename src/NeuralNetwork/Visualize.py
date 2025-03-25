from graphviz import Digraph
from NeuralNetwork.Autograd import Scalar

def trace(child: Scalar) -> tuple[set[Scalar], set[tuple[Scalar, Scalar]]]:
    """
    Connect all nodes from children to all its parent

    Args:
        child (Scalar): the child (lowest) node
    """
    nodes, edges = set(), set()
    def build(v: Scalar):
        if v not in nodes:
            nodes.add(v)
            for parent in v.get_parents():
                edges.add((parent, v))
                build(parent)

    build(child)
    return nodes, edges

def draw_dot(child: Scalar):
    """
    Draw the graph

    Args:
        child (Scalar): the children (latest) Scalar node

    Returns:
        graph: the graph
    """
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})

    nodes, edges = trace(child)
    for n in nodes:
        uid = str(id(n))

        # Create the node to store the value
        dot.node(name = uid, label = "{val: %.5f}" % (n.value, ), shape='record')

        if n.get_operation():
            dot.node(name = uid + n.get_operation(), label=n.get_operation())

            dot.edge(uid + n.get_operation(), uid)

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2.get_operation())

    return dot
