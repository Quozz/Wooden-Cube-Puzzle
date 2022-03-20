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


def generate_shape_location(shape_configuration, shape, cube_rotations):
    """
    Transform a shape by a rotation and then a translation.

    Parameters
    ----------
    shape_configuration : A tuple of length two containing at index 0 a np.array
    of length 3 representing a translation, and at index 1 an index describing
    a rotation
    shape : The np.arrays of length 3 representing the untransformed locations
    of a shape
    cube_rotations : a list of 24 rotations which correspond to the index in
    shape_configuration[1]

    Returns
    -------
    translated_shape : A transformed shape consisting of np.arrays of length 3.
    It first rotates the shape and  the translates the shape

    """
    rotation = cube_rotations[shape_configuration[1]]
    translation = shape_configuration[0]
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


def generate_allowed_shape_locations(shape, shape_configurations, cube_rotations):
    possible_shape_locations = [generate_shape_location(shape_configuration,
                                                        shape, cube_rotations)
                                for shape_configuration in shape_configurations]

    proper_shape_locations = [shape_location for shape_location
                              in possible_shape_locations
                              if good_shape_location(shape_location)]

    proper_shape_configurations = [shape_configuration for shape_configuration
                                   in shape_configurations
                                   if good_shape_location(generate_shape_location(
                                       shape_configuration, shape, cube_rotations))]
    return proper_shape_locations, proper_shape_configurations


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

    shape_configurations = [[location, rotation_label] for location in locations
                            for rotation_label in rotation_labels]

    cube_rotations = generate_cube_rotations()

    L_shapes, _ = generate_allowed_shape_locations(
        L_shape, shape_configurations, cube_rotations)
    Y_shapes, _ = generate_allowed_shape_locations(
        Y_shape, shape_configurations, cube_rotations)

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


def update_dscore_local(dscores, in_C, weight_array, neighbour_dict, vertex, added):
    # update_dscore_local and weight_update_dscore are tested, using GloballyUpdatedateDScore.
    # But are >100 times faster. This code still takes up most of the time.
    # The DScore for the vertex changes to negative itself, because
    # flipping the vertex twice changes nothing, and therefore the cost
    # does not change, CostChange = -DScore added - DScore removed = 0
    dscores[vertex] = -dscores[vertex]
    if added:   # If the vertex is added
        for neighbour in neighbour_dict[vertex]:
            sorted_edge = tuple(sorted((vertex, neighbour), reverse=True))
            if in_C[neighbour]:
                # In this case, the edge no longer contributes to the DScore decrease if neighbour is removed
                # If edge vertex, neighbour is removed. Thus, DScore Improves by weight
                dscores[neighbour] += + weight_array[sorted_edge]
            else:
                # In this case, the edge was not covered, but is now covered
                # Therefore it no longer contributes to the DScore of the neighbour
                # For an edge outside of C, each weight contributes positively
                # To the DScore, hence the DScore is affected negatively
                # If edge vertex, neighbour is removed. Thus, DScore Improves
                # by weight
                dscores[neighbour] += - weight_array[sorted_edge]
    else:       # If the vertex is removed
        for neighbour in neighbour_dict[vertex]:
            sorted_edge = tuple(sorted((vertex, neighbour), reverse=True))
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

    reduced_edge = ()
    re_dscores = ()
    re_ages = ()
    for vertex in edge:
        if confs[vertex]:
            reduced_edge = reduced_edge + (vertex,)
            re_dscores = re_dscores + (dscores[vertex],)
            re_ages = re_ages + (ages[vertex],)
    if len(reduced_edge) == 1:
        vertex = reduced_edge[0]
    elif dscores[0] > dscores[1]:
        vertex = reduced_edge[0]
    elif dscores[1] > dscores[0]:
        vertex = reduced_edge[1]
    elif ages[0] > ages[1]:
        vertex = reduced_edge[0]
    else:
        vertex = reduced_edge[1]
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
        ties in favor of the oldest one; to be tested and age implemented
    13 C := C\{u}, confChange(u) := 0 and confChange(z) := 1 for each z ∈ N(u);
    14 choose an uncovered edge e randomly;
    15 choose a vertex v ∈ e such that confChange(v) = 1 with higher dscore,
    breaking ties in favor of the older one; To Break ties
    16 C := C ∪ {v}, confChange(z) := 1 for each z ∈ N(v);  check
    17 w(e) := w(e) + 1 for each uncovered edge e;  check
    18 if w ≥ γ then w(e) := ⌊ρ · w(e)⌋ for each edge e; check
    19 return C∗; check
    20 end check

    Parameters
    ----------
    E : TYPE
        DESCRIPTION.
    V : TYPE
        DESCRIPTION.
    neighbour_dict : TYPE
        DESCRIPTION.
    graph_array : TYPE
        DESCRIPTION.
    cut_off_time : TYPE
        DESCRIPTION.
    weight_limit : TYPE
        DESCRIPTION.
    weight_loss_rate : TYPE
        DESCRIPTION.

    Returns
    -------
    cover_in_C : TYPE
        DESCRIPTION.
    in_C : TYPE
        DESCRIPTION.
    dscores : TYPE
        DESCRIPTION.
    confs : TYPE
        DESCRIPTION.
    ages : TYPE
        DESCRIPTION.
    uncovered_edges : TYPE
        DESCRIPTION.
    weight_array : TYPE
        DESCRIPTION.
    """

    print('Building state tracking structures')

    # 3 initialize edge weights and dscores of vertices;
    dscores = [int(0)]*len(V)
    ages = [int(0)]*len(V)
    weight_array = np.zeros((len(V), len(V)), dtype=int)

    # 4 initialize the confChange array as an all-1 array;
    confs = [bool(1)]*len(V)
    for i in range(len(V)):
        for j in range(len(V)):
            if i > j:
                weight_array[i, j] = int(graph_array[i, j])
    uncovered_edges = []

    # 5 construct C greedily until it is a vertex cover; - just C = V
    in_C = [bool(1)]*len(V)

    # 6 C∗ := C;
    cover_in_C = copy.deepcopy(in_C)
    counter_start = 0
    counter_end = 0
    cover_counter = 0
    print('Starting Loop')
    InitialTime = time.time()
    # 7 while elapsed time < cutoff and MVC not yet obtained do
    while time.time() - InitialTime < cut_off_time and cover_counter < 13:
        counter_start += 1
        """
        for Bool in in_C:
            assert isinstance(Bool, bool)
        for DScore in dscores:
            assert isinstance(DScore,int)
        for ConfChange in confs:
            assert isinstance(ConfChange,bool)
        for age in ages:
            assert isinstance(age, int)
        """
        # 8   if there is no uncovered edge then
        if not uncovered_edges:
            print(time.time() - InitialTime,
                  ' Removing vertex number', cover_counter)
            # If C is empty, we cannot remove a vertex and hence the loop is
            # ended. This could only happen if a catastrophe occured.
            if not any(in_C):
                print(time.time() - InitialTime)
                print('Catastrophe')
                break
            # 9   C∗ := C;

            cover_in_C = copy.deepcopy(in_C)
            # 10  remove a vertex with the highest dscore from C;

            # Find a rendom vertex with max DScore in C
            C_indices = list(itertools.compress(range(len(in_C)), in_C))
            C_dscores = list(itertools.compress(dscores, in_C))
            max_dscore = max(C_dscores)
            sub_index_max = random.choice([index for index in range(len(C_dscores))
                                           if C_dscores[index] == max_dscore])
            vertex = C_indices[sub_index_max]
            assert dscores[vertex] <= 0
            assert in_C[vertex] == True
            in_C[vertex] = False
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
        C_indices = list(itertools.compress(range(len(in_C)), in_C))
        C_dscores = list(itertools.compress(dscores, in_C))
        max_dscore = max(C_dscores)
        sub_index_max = random.choice([index for index in range(len(C_dscores))
                                       if C_dscores[index] == max_dscore])
        vertex = C_indices[sub_index_max]

        # 13 C := C\{u}, confChange(u) := 0 and confChange(z) := 1
        # for each z ∈ N(u);
        assert dscores[vertex] <= 0
        assert in_C[vertex] is True
        in_C[vertex] = False

        confs[vertex] = False
        ages[vertex] = int(0)
        for neighbour in neighbour_dict[vertex]:
            confs[neighbour] = True

        uncovered_edges = update_uncovered(in_C, uncovered_edges, vertex,
                                           neighbour_dict, added=False)

        dscores = update_dscore_local(dscores, in_C, weight_array,
                                      neighbour_dict, vertex, added=False)

        # 14 choose an uncovered edge e randomly;
        edge = random.choice(uncovered_edges)

        # 15 choose a vertex v ∈ e such that confChange(v) = 1 with higher
        # dscore, breaking ties in favor of the older one;
        vertex = choose_added_vertex(edge, confs, dscores, ages)

        # 16 C := C ∪ {v}, confChange(z) := 1 for each z ∈ N(v);
        assert dscores[vertex] >= 0
        in_C[vertex] = True
        ages[vertex] = int(0)
        for neighbour in neighbour_dict[vertex]:
            confs[neighbour] = True

        uncovered_edges = update_uncovered(in_C, uncovered_edges, vertex,
                                           neighbour_dict, added=True)
        dscores = update_dscore_local(dscores, in_C, weight_array, neighbour_dict,
                                      vertex, added=True)
        # 17 w(e) := w(e) + 1 for each uncovered edge e;  check
        for edge in uncovered_edges:
            weight_array[edge] += 1
        dscores = weight_update_dscore(uncovered_edges, dscores)

        for vertex in range(len(ages)):
            ages[vertex] += 1
        average_weight = np.sum(weight_array)/len(E)

        # 18 if w ≥ γ then w(e) := ⌊ρ · w(e)⌋ for each edge e; check
        if average_weight > weight_limit:
            for edge in E:
                weight_array[edge] = math.ceil(
                    weight_array[edge]*weight_loss_rate)
                dscores = globally_update_dscore(
                    dscores, in_C, weight_array, E)
                print('Weights partially forgotten')
        counter_end += 1
    else:
        print('Time is out')

    print('counter_start', counter_start,
          '\n', 'counter_end', counter_end,
          '\n cover_counter', cover_counter)
    # 19 return C∗; check
    return cover_in_C, in_C, dscores, confs, ages, uncovered_edges, weight_array


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
    cut_off_time = 1
    # duration of algorithm in seconds

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
