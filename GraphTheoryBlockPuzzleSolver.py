import random
import math
import pickle
import time
import copy
import itertools
import numpy as np

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 14:02:25 2022

@author: wisse
"""

# -*- coding: utf-8 -*-


def generate_cube_rotations():
    """
    Generate all 24 cube preserving proper rotations matrices in a list.

    Returns
    -------
    cube_rotations : A list of 24 3x3 numpy arrays, which, when acting with
    matrixmultiplication on a vector rotates the vector. They are proper
    rotations, i.e. no reflections, and symmetries of the cube.
    Multiplying a shape by these rotations generates all possible
    orientations of the shape.
    """
    permutations3 = [[0, 1, 2], [0, 2, 1], [
        1, 2, 0], [1, 0, 2], [2, 0, 1], [2, 1, 0]]
    mirrors = [[], [0, 1], [0, 2], [1, 2]]
    cube_rotations = []
    for permutation in permutations3:
        for mirror in mirrors:
            cube_rotation = np.zeros((3, 3))
            for i in range(3):
                if i in mirror:
                    cube_rotation[i, permutation[i]] = -1
                else:
                    cube_rotation[i, permutation[i]] = 1
            cube_rotations.append(cube_rotation)
    return cube_rotations


def rotate_shape(shape, rotation):
    """
    Rotate a shape around its (rounded) midpoint.

    Parameters
    ----------
    shape : A list of 3-dimensional numpy arrays - vectors -  representing
    the locations of the blocks the shape consists of.
    rotation :
        A 3x3 numpy array which rotates vectors via matrix multiplication.

    Returns
    -------
    rotated_shape, the shape rotated in the way specified by the rotation
    around approximately its midpoint
    """
    if len(shape) == 0:
        print('Catastrofic rotate_shape Error')
        return []
    total_sum = np.array([0, 0, 0])
    for location in shape:
        total_sum += location
    average = total_sum/len(shape)
    mid_point = np.rint(average).astype(int)
    rotated_shape = [np.rint(np.matmul(rotation, location - mid_point)
                             + mid_point).astype(int) for location in shape]
    return rotated_shape


def generate_shape_location(shape_config, shape, cube_rotations):
    """
    Transform a shape by a rotation and then a translation.

    Parameters
    ----------
    shape_config : A tuple of length two containing at index 0 a np.array
    of length 3 representing a translation, and at index 1 an index describing
    a rotation
    shape : The np.arrays of length 3 representing the untransformed locations
    of a shape
    cube_rotations : a list of 24 rotations which correspond to the index in
    shape_config[1]

    Returns
    -------
    translated_shape : A transformed shape consisting of np.arrays of length 3.
    It first rotates the shape and  the translates the shape

    """
    rotation = cube_rotations[shape_config[1]]
    translation = shape_config[0]
    rotated_shape = [np.matmul(rotation, location) for location in shape]
    translated_shape = [np.rint(np.add(translation, location)).astype(
        int) for location in rotated_shape]
    return translated_shape


def good_shape_location(shape_location):
    """
    Check whether a shape is contained in the 5x5x5 grid [0,4] x [0,4] x [0,4].

    Parameters
    ----------
    shape_location : A list of 1D np.arrays of length 3, describing a shape

    Returns
    -------
    bool False if the shape is outside the grid.
        True if the shape is contained in the grid.

    """
    for location in shape_location:
        if np.any(location > 4.5):
            return False
        if np.any(location < -0.5):
            return False
    return True


def add_shape(cube, shape_location):
    """
    Add a shape to the cube, keeping track of how many shapes overlap
    a specific point in a 5x5x5 grid.

    Parameters
    ----------
    cube : a 5x5x5 np.array containing positive integer values describing
           how many shapes overlap at that point, excluding the shape
           described in shape_location
    shape_location : A list containing np.arrays of length 3 describing
    locations. As a whole, they describe a shape.

    Returns
    -------
    cube : a 5x5x5 np.array containing positive integer values describing
           how many shapes overlap at that point, including the shape
           described in shape_location
    """
    for location in shape_location:
        value = cube[tuple(location)]
        cube[tuple(location)] = value + 1
    return cube


def generate_base_shapes():
    L_shape = [np.array([i, j, 0]) for i in range(4) for j in range(2)]
    for j in range(2):
        L_shape.append(np.array([3, j, 1]))
    Y_shape = [np.array([i, j, 0]) for i in range(4) for j in range(2)]
    for j in range(2):
        Y_shape.append(np.array([2, j, 1]))
    return L_shape, Y_shape


def generate_cube(cube_size):
    cube = np.zeros((cube_size, cube_size, cube_size))
    locations = [np.array([i, j, k]) for i in range(cube_size)
                 for j in range(cube_size) for k in range(cube_size)]
    return cube, locations


def generate_allowed_shape_locations(shape, shape_configs, cube_rotations):
    possible_shape_locations = [generate_shape_location(shape_config,
                                                        shape, cube_rotations)
                                for shape_config in shape_configs]

    proper_shape_locations = [shape_location for shape_location
                              in possible_shape_locations
                              if good_shape_location(shape_location)]

    proper_shape_configs = [shape_config for shape_config
                            in shape_configs
                            if good_shape_location(generate_shape_location(
                                    shape_config, shape, cube_rotations))]
    return proper_shape_locations, proper_shape_configs


def overlap(shape1, shape2, cube_size):
    cube = np.zeros((cube_size, cube_size, cube_size))
    add_shape(cube, shape1)
    add_shape(cube, shape2)
    if np.any(cube > 1):
        return 1
    return 0


def number_of_combinations():
    ncr = 1
    for i in range(6):
        ncr = ncr * (768 - i)/(i + 1)
        ncr = ncr**2
    return ncr


def generate_graph_EV():
    """
    Generate the graph G = (E,V)
    119 seconds
    """
    cube_size = 5
    L_shape, Y_shape = generate_base_shapes()
    _, locations = generate_cube(cube_size)
    rotation_labels = list(range(24))

    shape_configs = [[location, rotation_label] for location in
                     locations for rotation_label in rotation_labels]

    cube_rotations = generate_cube_rotations()

    L_shapes, _ = generate_allowed_shape_locations(
        L_shape, shape_configs, cube_rotations)
    Y_shapes, _ = generate_allowed_shape_locations(
        Y_shape, shape_configs, cube_rotations)

    shapes = Y_shapes + L_shapes

    E = []
    for i in range(len(shapes)):
        for j in range(i):
            if overlap(shapes[i], shapes[j], cube_size):
                E.append((i, j))

    V = list(range(len(shapes)))
    """
    with open('saved_edges.pkl', 'wb') as f:
        pickle.dump(E, f)
    with open('saved_vertices.pkl', 'wb') as f:
        pickle.dump(V, f)
    with open('saved_shapes.pkl', 'wb') as f:
        pickle.dump(shapes, f)
    """
    return E, V


def generate_neighbour_dictionary(E, V):
    """
    Construct graph dictionary from lists E,V.
    3.2 seconds

    Parameters
    ----------
    E : TYPE
        DESCRIPTION.
    V : TYPE
        DESCRIPTION.

    Returns
    -------
    neighbour : TYPE
        DESCRIPTION.
    """

    neighbour_dict = {}
    for i in V:
        neighbour_dict[i] = []
    for i, j in E:
        neighbour_dict[i] = neighbour_dict[i] + [j]
        neighbour_dict[j] = neighbour_dict[j] + [i]

    with open('neighbour_dict.pkl', 'wb') as f:
        pickle.dump(neighbour_dict, f)

    return neighbour_dict


def generate_graph_array(E, V):
    """
    represent the graph as a square adjacency matrix.

    Parameters
    ----------
    E : List of edges, which are tuples (vertex1,vertex2)
    V : List of vertices

    Returns
    -------
    graph_array : len(V) x len(V) np.array with True on edges.

    """
    graph_array = np.zeros((len(V), len(V)), dtype=bool)
    for i, j in E:
        graph_array[i, j] = 1
        graph_array[j, i] = 1

    with open('graph_array.pkl', 'wb') as f:
        pickle.dump(graph_array, f)

    return graph_array


def find_uncovered_edges(E, C):
    """
    Genereate list of uncovered edge from list of all edges and list of
    covering vertices.

    Could be used to check update_uncovered, but much more computationally
    expensive.

    Parameters
    ----------
    E : List of edges of GraphG, tuples (vertex1, vertex2)
    C : List of covering vertices, vertex, type int.

    Returns
    -------
    uncovered_edges : TYPE
        DESCRIPTION.

    """
    uncovered_edges = []
    for vertex1, vertex2 in E:
        if vertex1 not in C:
            if vertex2 not in C:
                uncovered_edges.append(vertex1, vertex2)
    return uncovered_edges


def update_uncovered(in_C, uncovered_edges, vertex,  neighbour_dict, added):
    # Updates the uncovered_edges set in EDataFrame After
    # Adding (added = True) or removing (added = False) a vertex
    assert in_C[vertex] == added
    for vertex2 in neighbour_dict[vertex]:
        if in_C[vertex2] == 0:
            assert not vertex == vertex2
            sorted_edge = tuple(sorted((vertex, vertex2), reverse=True))

            """
            print('edge', (EDataFrame.at[index,'vertex1'],
                           EDataFrame.at[index,'vertex2']),
                  '\n Covered', EDataFrame.at[index,'Covered'],
                  '\n added', added,
                  '\n vertexHigh,  vertexLow, vertex, vertex2',
                  vertexHigh, vertexLow, vertex, vertex2)
            """
            assert (sorted_edge in uncovered_edges) == added
            if added:
                uncovered_edges.remove(sorted_edge)
            else:
                uncovered_edges.append(sorted_edge)
    return uncovered_edges


def globally_update_dscore(dscores, in_C, weight_array, E):
    """
    Update DScore for all Vertices
    Time: Extremely Long
    The DScore of the vertex is the difference in cost when you flip the dict
    between C and Complement C. The cost is a function of the graph G and C,
    and consists of The sum of all weights of all edges not covered by C.
    When a vertex is not in C, adding it to C covers edges, therefore flipping
    it decreases the cost, and therefore has a positive DScore. The vertex with
    the maximum DScore should be flipped to ensure minimum cost.
    When a vertex is in C it uncovers edges, and hence it has a negative
    DScore. The least negative DScore, i.e. the maximum DScore should be chosen
    in order to get a minimum cost. One way to calculate the DScore is to
    calculate the cost for both scenarios but this requires at least to loop
    over all uncovered edges to calculate the DScore for adding a vertex,
    and all partially covered edges to calculate the DScore for removing a
    vertex. This Sounds Like it will take a long time, in particular for
    removing a vertex. To make matters worse, for the current dataframes used
    we need to loop select all uncovered edges and partially covered edges
    which is O(|E|)
    """
    # First reset the dscore
    dscores = [0]*len(dscores)
    # For each edge, check if the corresponding vertices are in C
    for vertex1, vertex2 in E:
        vertex1InC = in_C[vertex1]
        vertex2InC = in_C[vertex2]
        # If so, change the dscore of the vertices appropriately
        if vertex1InC != vertex2InC:  # x or
            if vertex1InC:
                dscores[vertex1] += - weight_array[vertex1, vertex2]
            if vertex2InC:
                dscores[vertex2] += - weight_array[vertex1, vertex2]
        elif (vertex1InC is False) and (vertex2InC is False):
            dscores[vertex1] += weight_array[vertex1, vertex2]
            dscores[vertex2] += weight_array[vertex1, vertex2]
    return dscores


def update_dscore_local(dscores, in_C, weight_array, neighbour_dict, vertex,
                        added):
    """
    Efficiently update dscores after vertex is added.

    update_dscore_local is tested, using GloballyUpdatedateDScore. But is
    >100 times faster. This code takes up a large portion of computational
    power.

    Parameters
    ----------
    dscores : List of integers
        Contains dscores of all vertices, vertices correspond to labels
    in_C : list of bools
        Vertex corresponding to index is in C if in_C[Vertex] == 1
    weight_array : int array len(V) x len(V)
        Contains the weights of all edges, pairs of vertices (vertex1, vertex2)
        can be used as index, with vertex1 > vertex2. All other entries are 0,
        Also 0 if pair of vertices is not an edge.
    neighbour_dict : Dictionary
        key = vertex, value = list of neighbours of vertex
    vertex : Integer
        An integer corresponding to an index in the graph used for indexing
    added : Bool
        Whether a vertex is added (True) or removed (False).

    Returns
    -------
    dscores : List of integers
        Contains updated dscores of all vertices, vertices correspond to labels

    """

    # The DScore for the vertex changes to negative itself, because
    # flipping the vertex twice changes nothing, and therefore the cost
    # does not change, CostChange = -DScore added - DScore removed = 0
    dscores[vertex] = -dscores[vertex]
    if added:   # If the vertex is added
        for neighbour in neighbour_dict[vertex]:
            if vertex > neighbour: sorted_edge = vertex,neighbour
            else: sorted_edge = neighbour,vertex

            if in_C[neighbour]:
                # In this case, the edge no longer contributes to the DScore
                # decrease if neighbour is removed. If edge vertex, neighbour
                # is removed. Thus, DScore Improves by weight
                dscores[neighbour] += + weight_array[sorted_edge]
            else:
                # In this case, the edge was not covered, but is now covered
                # Therefore it no longer contributes to the DScore of the
                # neighbour. For an edge outside of C, each weight contributes
                # positively to the DScore, hence the DScore is affected
                # negatively If edge vertex, neighbour is removed. Thus, DScore
                # Improves by weight
                dscores[neighbour] += - weight_array[sorted_edge]
    else:       # If the vertex is removed
        for neighbour in neighbour_dict[vertex]:
            if vertex > neighbour: sorted_edge = vertex,neighbour
            else: sorted_edge = neighbour,vertex

            if in_C[neighbour]:
                # In this case, the edge starts to contribute negatively to the
                # DScore decrease if neighbour is removed
                # Thus, DScore decreases by weight

                dscores[neighbour] += - weight_array[sorted_edge]
            else:
                # In this case, the edge was covered, but is now not covered
                # Therefore it now contributes to the DScore of the neighbour
                # For an edge outside of C, each weight contributes positively
                # To the DScore, hence the DScore is affected positively
                # Thus, DScore improves by weight
                dscores[neighbour] += + weight_array[sorted_edge]
    # We may choose to Calculate the DScore after a flip by Looping over the
    # neighbours of the neighbours.
    # If a neighbour is not in C, flipping The vertex flips the edge between
    # uncovered and covered. If the neighbour is in C, the edge is covered
    # regardless of whether the vertex is in C.
    # We can calculate the DScore in this way, instead of updating it
    # based on a initial DScore because it is easier to do, and such a function
    # should probably be written anyway to initialize the DScore
    # If the number of neighbours is very large O(|V|), then the complexity
    # of this function is O(|V|^2), because we need to iterate over neighbours
    # of neighbours. This is rather large.
    # On the other hand, we may use the fact that flipping the DScore change
    return dscores

def weight_update_dscore(uncovered_edges, dscores):
    """
    Update the dscores at the end of the main loop.

    Update is due to change in Weights of uncovered edges.

    Parameters
    ----------
    uncovered_edges : list of edges; tuples (vertex1, vertex2), contained in E.
    dscores : list of ints of length |V|

    Returns
    -------
    dscores : Updated list of ints of length |V|

    """
    # For indices of edges not covered
    for edge in uncovered_edges:
        # The weight increases by 1, since it contributes positively to
        # both vertices, since they are not in C, it affects the DScore by +1
        vertex1, vertex2 = edge
        dscores[vertex1] += 1
        dscores[vertex2] += 1
    return dscores


def choose_added_vertex(edge, confs, dscores, ages):
    """
    Choose vertex from edge to add to covering according to NuMVC algorithm.

    Decision tree:
       Choose vertex with highest DScore with ConfChange = 1
       If Vertices have equally high DScore and both ConfChange = 0:
       Choose Oldest vertex

    Parameters
    ----------
    edge : Luple of int (vertex1, vertex2)
    confs : List of bools of length |V|
    dscores : List of ints of length |V|
    ages : List of ints of length |V|

    Returns
    -------
    vertex : int

    """

    vertex0, vertex1 = edge
    conf_change0, conf_change1 = confs[vertex0], confs[vertex1]

    assert conf_change0 or conf_change1
    if not (conf_change0 and conf_change1):
        if conf_change0:
            vertex = vertex0
        if conf_change1:
            vertex = vertex1
    elif dscores[vertex0] > dscores[vertex1]:
        vertex = vertex0
    elif dscores[vertex1] > dscores[vertex0]:
        vertex = vertex1
    elif ages[vertex0] > ages[vertex1]:
        vertex = vertex0
    else:
        vertex = vertex1
    assert dscores[vertex] >= 0
    return vertex


def select_removed_vertex(in_C, dscores, V, ages):
    """
    select a vertex with the highest dscore from C

    Parameters
    ----------
    in_C : TYPE
        DESCRIPTION.
    dscores : TYPE
        DESCRIPTION.

    Returns
    -------
    vertex : int
        integer with highest dscore

    """
    vertices_in_C = list(itertools.compress(V, in_C))
    C_dscores = list(itertools.compress(dscores, in_C))
    max_dscore = max(C_dscores)
    maxima = [vertex for vertex in vertices_in_C if
              dscores[vertex] == max_dscore]

    ages_maxima = [ages[vertex] for vertex in maxima]
    max_age = max(ages_maxima)
    oldest_maxima = [vertex for vertex in maxima if ages[vertex] == max_age]

    vertex = random.choice(oldest_maxima)
    assert dscores[vertex] <= 0
    assert in_C[vertex] is True
    return vertex


def NuMVC(E, V, neighbour_dict, graph_array, cut_off_time, weight_limit,
          weight_loss_rate):
    """
    Implement the an efficient algorithm to find the Minimum Vertex Cover,
    using two stage vertex interchange and edge weighting with forgetting.

    The Algorithm is as follows:
    (G,cutoff)
    Input: graph G = (V,E), the cutoff time
    Output: vertex cover of G
    2 begin
    3 initialize edge weights and dscores of vertices;
    4 initialize the confChange array as an all-1 array;
    5 construct C greedily until it is a vertex cover; - just C = V
    6 C∗ := C;
    7 while elapsed time < cutoff do
    8   if there is no uncovered edge then
    9   C∗ := C;
    10  remove a vertex with the highest dscore from C;
    11  continue;
    12 choose a vertex u ∈ C with the highest dscore, breaking
        ties in favor of the oldest one;
    13 C := C\{u}, confChange(u) := 0, confChange(z) := 1 for each z ∈ N(u);
    14 choose an uncovered edge e randomly;
    15 choose a vertex v ∈ e such that confChange(v) = 1 with higher dscore,
    breaking ties in favor of the older one; To Break ties
    16 C := C ∪ {v}, confChange(z) := 1 for each z ∈ N(v);
    17 w(e) := w(e) + 1 for each uncovered edge e;
    18 if w ≥ γ then w(e) := ⌊ρ · w(e)⌋ for each edge e;
    19 return C∗;
    20 end

    Parameters
    ----------
    E : List of tuples (vertex1, vertex2), vertices positive integers.
        Contains all edges of graph G
    V : list of vertices, positive integers.
        Contains all vertices of graph G
    neighbour_dict : dictionary
        key: vertices, value: list of all neighbours of vertex in graph G
    graph_array : len(V) x len(V) numpy array of bool.
        graph_array[vertex1, vertex2] True if (vertex1, vertex2) is an edge
    cut_off_time : float or int
        Time the while loop runs before it stops, if it does not find a
        solution
    weight_limit : int
        If average weight > weight limit, weight reduced by factor
    weight_loss_rate : float between 0,1
        determines how much weight is lost at once.

    Returns
    -------
    cover_in_C : list of bools of len(V)
        cover_in_C[vertex] == True if vertex in minimal vertex covering
    in_C : list of bools of len(V)
        cover_in_C[vertex] == True if vertex in C
    dscores : list of ints of len(V)
        contains the dscores of all vertices. dscore is heuristic in order
        to maximise the chance
        of finding a vertex cover, dscore is maximalised
    confs : bool
        If False, may not be added to C.
    ages : int
        ages[vertex] is #loops since last change to vertex.
    uncovered_edges : list tuples (vertex1, vertex2)
        Contains all edges not covered by C
    weight_array : numpy array of integers of size len(V) x len(V)
        weight_array[(vertex1, vertex2)] is edge weight of edge vertex1>vertex2
        If edge not in E, it is 0.
    """

    print('Building state tracking structures')

    # 3 initialize edge weights and dscores of vertices;
    dscores = [int(0)]*len(V)
    ages = [int(0)]*len(V)


    # 4 initialize the confChange array as an all-1 array;
    confs = [bool(1)]*len(V)

    # initialize weight_array
    weight_array = np.zeros((len(V), len(V)), dtype=int)
    for vertex1 in V:
        for vertex2 in V:
            if vertex1 > vertex2:
                weight_array[vertex1, vertex2] = \
                    int(graph_array[vertex1, vertex2])
    average_weight = 1
    uncovered_edges = []

    # 5 construct C greedily until it is a vertex cover; - just C = V
    in_C = [bool(1)]*len(V)

    # 6 C∗ := C;
    cover_in_C = copy.deepcopy(in_C)
    counter_start = 0
    counter_end = 0
    cover_counter = 0
    print('Starting Loop')
    initial_time = time.time()
    # 7 while elapsed time < cutoff and MVC not yet obtained do
    while time.time() - initial_time < cut_off_time and cover_counter < 13:
        counter_start += 1
        """
        for boolean in in_C:
            assert isinstance(boolean, bool)
        for dscore in dscores:
            assert isinstance(dscore,int)
        for conf_change in confs:
            assert isinstance(conf_change,bool)
        for age in ages:
            assert isinstance(age, int)
        """
        # 8   if there is no uncovered edge then
        if not uncovered_edges:
            print(time.time() - initial_time,
                  ' Removing vertex number', cover_counter)
            # If C is empty, we cannot select a vertex, this should not happen.
            if not any(in_C):
                print(time.time() - initial_time)
                assert in_C
                
            # 9   C∗ := C;
            cover_in_C = copy.deepcopy(in_C)
            
            # 10  remove a vertex with the highest dscore from C;
            vertex = select_removed_vertex(in_C, dscores, V, ages)
            in_C[vertex] = False

            # update dscores, uncovered edges, vertex age
            ages[vertex] = int(0)
            dscores = update_dscore_local(dscores, in_C, weight_array,
                                          neighbour_dict, vertex, added=False)
            uncovered_edges = update_uncovered(in_C, uncovered_edges, vertex,
                                               neighbour_dict, added=False)

            cover_counter += 1
            # 11  continue;
            continue

        # 12 choose a vertex u ∈ C with the highest dscore, breaking
        # ties in favor of the oldest one; to be tested and age implemented
        vertex = select_removed_vertex(in_C, dscores, V, ages)

        # 13 C := C\{u}, confChange(u) := 0 and confChange(z) := 1
        # for each z ∈ N(u);
        in_C[vertex] = False
        confs[vertex] = False
        for neighbour in neighbour_dict[vertex]:
            confs[neighbour] = True

        # update dscores, uncovered_edges, vertex age
        uncovered_edges = update_uncovered(in_C, uncovered_edges, vertex,
                                           neighbour_dict, added=False)
        dscores = update_dscore_local(dscores, in_C, weight_array,
                                      neighbour_dict, vertex, added=False)
        ages[vertex] = int(0)

        # 14 choose an uncovered edge e randomly;
        edge = random.choice(uncovered_edges)

        # 15 choose a vertex v ∈ e such that confChange(v) = 1 with higher
        # dscore, breaking ties in favor of the older one;
        vertex = choose_added_vertex(edge, confs, dscores, ages)

        # 16 C := C ∪ {v}, confChange(z) := 1 for each z ∈ N(v);
        in_C[vertex] = True

        for neighbour in neighbour_dict[vertex]:
            confs[neighbour] = True

        # update dscores, uncovered_edges, vertex age
        ages[vertex] = int(0)
        uncovered_edges = update_uncovered(in_C, uncovered_edges, vertex,
                                           neighbour_dict, added=True)
        dscores = update_dscore_local(dscores, in_C, weight_array,
                                      neighbour_dict, vertex, added=True)

        # 17 w(e) := w(e) + 1 for each uncovered edge e;
        for edge in uncovered_edges:
            weight_array[edge] += 1
            average_weight += 1/len(E)

        # update dscores, ages
        dscores = weight_update_dscore(uncovered_edges, dscores)
        for vertex in V:
            ages[vertex] += 1

        # 18 if w ≥ γ then w(e) := ρ · w(e) for each edge e;
        if average_weight > weight_limit:
            for edge in E:
                weight_array[edge] = math.ceil(
                    weight_array[edge]*weight_loss_rate)
                dscores = globally_update_dscore(
                    dscores, in_C, weight_array, E)
                average_weight = np.sum(weight_array)/len(E)
                print('Weights partially forgotten')
        counter_end += 1
    else:
        print('Time is out')

    # print statistics
    print('counter_start', counter_start,
          '\n', 'counter_end', counter_end,
          '\n cover_counter', cover_counter)
    Y_in_cover = cover_in_C[:768]
    L_in_cover = cover_in_C[768:]
    print('Y pieces: ', len(Y_in_cover) - sum(Y_in_cover),
          '\nL pieces: ', len(L_in_cover) - sum(L_in_cover))

    # 19 return C∗; check
    return (cover_in_C, in_C, dscores, confs, ages,
            uncovered_edges, weight_array)


def main():
    """
    Load the data representing the graph and call the main loop. Then dump
    the generated data, the minimal vertex coverings and the edge weights
    in particular to pickle files.

    Returns
    -------
    None.

    """
    weight_limit = 100
    weight_loss_rate = 0.5
    cut_off_time = 3600    # duration of algorithm in seconds


    with open('saved_edges.pkl', 'rb') as f:
        E = pickle.load(f)
    with open('saved_vertices.pkl', 'rb') as f:
        V = pickle.load(f)
    with open('neighbour_dict.pkl', 'rb') as f:
        neighbour_dict = pickle.load(f)
    with open('graph_array.pkl', 'rb') as f:
        graph_array = pickle.load(f)

    cover_in_C, in_C, dscores, confs, ages, uncovered_edges, weight_array = \
        NuMVC(E, V, neighbour_dict, graph_array, cut_off_time, weight_limit,
              weight_loss_rate)
    with open('CoverC.pkl', 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(cover_in_C, f, pickle.HIGHEST_PROTOCOL)
    with open('in_C.pkl', 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(in_C, f, pickle.HIGHEST_PROTOCOL)
    with open('dscores.pkl', 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(dscores, f, pickle.HIGHEST_PROTOCOL)
    with open('confs.pkl', 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(confs, f, pickle.HIGHEST_PROTOCOL)
    with open('ages.pkl', 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(ages, f, pickle.HIGHEST_PROTOCOL)
    with open('uncovered_edges.pkl', 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(uncovered_edges, f, pickle.HIGHEST_PROTOCOL)
    with open('weight_array.pkl', 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(weight_array, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
