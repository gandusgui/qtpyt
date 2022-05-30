import numpy as np
from collections import deque, namedtuple, defaultdict
from qtpyt.tk_gpaw import get_bfs_indices

cutoff = 1e-4


def get_graph(calc, h, s, cutoff=cutoff):

    num_vertices = len(calc.atoms)
    bfs_ai = get_bfs_indices(calc, range(num_vertices), "append")

    # Vertex
    class vertex:
        """
            :tag       level set index (block index in matrix)
            :neighbors neighborlist vertices (neighborlist atoms)
            :index     vertex in graph (atom index in atoms)
        """

        def __init__(self, tag, neighbors):
            self.tag = tag
            self.neighbors = neighbors
            self.index = 0

    # Graph ::dict of vertices
    graph = defaultdict(lambda: vertex(0, []))

    # Start
    for a0 in range(num_vertices):
        # Find neighborlist for a1 > a0
        for a1 in range(a0, len(calc.atoms)):
            h_ij = h.take(bfs_ai[a0], axis=0).take(bfs_ai[a1], axis=1)
            if np.any(h_ij > cutoff):
                graph[a0].neighbors.append(a1)
                graph[a0].index = a0
                # Do not double count a0
                if a0 != a1:
                    # insert neighbor a0 in a1.neighborlist
                    graph[a1].neighbors.append(a0)

    return graph


def BFS(Q, graph, subset):
    """
        :Q      deque with first vertices
        :graph  collections::defaultdict of ::vertex
        :subset restrict search to subset of vertices
    """

    subset = list(subset)
    # Start
    while Q:
        v = Q.pop()
        adjacentEdges = graph[v].neighbors  # grapht_neighbors(v)[0]
        #
        for w in adjacentEdges:
            # Atom not discovered and in subset
            if (not graph[w].tag) & (w in subset):
                # Assign parent previous level set
                graph[w].tag = (
                    graph[v].tag + 1 if graph[v].tag > 0 else graph[v].tag - 1
                )
                Q.appendleft(w)


def get_tridiagonal_nodes(calc, h, s, pl1, pl2=None, cutoff=cutoff):
    """
        :pl1 number of atoms in left principal layer
        :pl2 number of atoms in right principal layer
    """

    if pl2 is None:
        pl2 = pl1

    # Construct graph
    graph = get_graph(calc, h, s, cutoff)

    # Number of vertices
    num_vertices = len(graph)

    # Pinn first and last l.s.
    v0 = range(pl1)
    vN = range(num_vertices - pl2, num_vertices)
    _set_tags(graph, v0, 1)
    _set_tags(graph, vN, -1)

    # BFS from left for left subset vertices
    Q = deque(range(pl1))
    subset = range(num_vertices // 2)
    BFS(Q, graph, subset)
    # BFS from right for right subset vertices
    Q = deque(range(num_vertices - 1, num_vertices - pl2 - 1, -1))
    subset = range(num_vertices // 2, num_vertices)
    BFS(Q, graph, subset)

    # from IPython import embed
    # embed()

    # Assign last l.s. from right to last l.s. from left
    tags = _get_tags(graph)
    tmin = min(tags)
    tmax = max(tags)
    for a in graph.values():
        if a.tag == tmin:
            a.tag = tmax

    # Rename positive
    tags = _get_tags(graph)
    u = np.unique(tags).tolist()
    for a in graph.values():
        # Start from 0 index
        if a.tag > 0:
            a.tag -= 1
        # Map negative to positive
        else:
            a.tag %= len(u)

    # Get nodes:= cut edges
    tags = _get_tags(graph)
    u = np.unique(tags).tolist()
    nodes = [0]  # N+1 nodes
    for u0 in u:
        indices = _get_indices(graph, u0)
        bfs_i = get_bfs_indices(calc, int(np.max(indices)))
        nodes.append(np.max(bfs_i) + 1)

    return nodes


def split_tridiagonal(mat, nodes):

    N = len(nodes) - 1

    mat_list_ii = [None for _ in range(N)]
    mat_list_ij = [None for _ in range(N - 1)]

    for q in range(N):
        q0 = nodes[q]
        q1 = nodes[q + 1]
        row = mat.take(range(q0, q1), axis=0)
        mat_list_ii[q] = row.take(range(q0, q1), axis=1)
        if q < (N - 1):
            q2 = nodes[q + 2]
            mat_list_ij[q] = row.take(range(q1, q2), axis=1)

    return mat_list_ii, mat_list_ij


def tridiagonalize(calc, h, s, pl1, pl2=None, cutoff=cutoff):

    nodes = get_tridiagonal_nodes(calc, h, s, pl1, pl2, cutoff)

    h_list_ii, h_list_ij = split_tridiagonal(h, nodes)
    s_list_ii, s_list_ij = split_tridiagonal(s, nodes)

    return (h_list_ii, s_list_ii), (h_list_ij, s_list_ij)


################# Helpers ###################


def _set_tags(graph, indices, tag):
    if isinstance(indices, int):
        indices = [indices]
    for i in indices:
        graph[i].tag = tag


def _get_tags(graph):
    return [a.tag for a in graph.values()]


def _get_indices(graph, tag):
    if isinstance(tag, int):
        tag = [tag]
    return sorted([a.index for a in graph.values() if a.tag in tag])
