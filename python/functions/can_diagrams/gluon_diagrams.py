import numpy as np

def combine_paths(*paths):
    """ 
    Combine multiple paths into a single array with the same length by padding with zeros.
    Args:
        paths (list): list of paths to combine
    Returns:
        final_path (np.array): combined path of dimension (len(paths), max_len, 2)
    """
    max_len = max([len(path) for path in paths])
    final_path = np.zeros((len(paths), max_len, 2), dtype=int)
    for i in range(len(paths)):
        if len(paths[i]) < max_len:
            final_path[i] = np.append(paths[i], np.zeros((max_len - len(paths[i]), 2), dtype=int), axis=0)
        else:
            final_path[i] = paths[i]
    return final_path

# First order
points_1st_1 = np.array([[0, 1], [0, 3], [1, 2], [2, 2]])
paths_1st_1g = np.array([[1, 3], [2, 3], [3, 4]]) 
paths_1st_1i = np.array([[0, 0]])

paths_1st_1 = combine_paths(paths_1st_1g, paths_1st_1i)

points_1st_2 = np.array([[0, 2], [1, 2], [2, 1], [2, 3]])
paths_1st_2g = np.array([[1, 2], [2, 3], [2, 4]]) 
paths_1st_2i = np.array([[0, 0]])

paths_1st_2 = combine_paths(paths_1st_2g, paths_1st_2i)

can_points_1st = np.empty((2, max(len(points_1st_1), len(points_1st_2)), 2))
can_points_1st[0] = points_1st_1
can_points_1st[1] = points_1st_2

can_paths_1st = np.empty((2, max(len(paths_1st_1), len(paths_1st_2)), max(len(paths_1st_1[0]), len(paths_1st_2[0])), 2), dtype=int)
can_paths_1st[0] = paths_1st_1
can_paths_1st[1] = paths_1st_2

can_number_1st = np.array([[1], [1], [1], [1]])

#Second order
points_2nd_1_1 = np.array([[0, 1], [0, 3], [0, 5], [1, 4], [2, 3], [4, 1]])
paths_2nd_1_1g = np.array([[1, 5], [2, 4], [3, 4], [5, 6]])
paths_2nd_1_1i = np.array([[4, 5]])
paths_2nd_1_1 = combine_paths(paths_2nd_1_1g, paths_2nd_1_1i)

points_2nd_1_2 = np.array([[0, 1], [0, 3], [0, 5], [1, 2], [2, 3], [4, 1]])
paths_2nd_1_2g = np.array([[1, 4], [2, 4], [3, 5], [5, 6]])
paths_2nd_1_2i = np.array([[4, 5]])
paths_2nd_1_2 = combine_paths(paths_2nd_1_2g, paths_2nd_1_2i)

points_2nd_1_3 = np.array([[0, 1], [0, 3], [0, 5], [1, 4], [2, 3], [4, 1]])
paths_2nd_1_3g = np.array([[1, 4], [2, 5], [3, 4], [5, 6]])
paths_2nd_1_3i = np.array([[4, 5]])
paths_2nd_1_3 = combine_paths(paths_2nd_1_3g, paths_2nd_1_3i)

points_2nd_2_1 = np.array([[0, 1], [0, 3], [1, 2], [2, 2], [3, 1], [3, 3]])
paths_2nd_2_1g = np.array([[1, 3], [2, 3], [4, 5], [4, 6]])
paths_2nd_2_1i = np.array([[3, 4]])
paths_2nd_2_1 = combine_paths(paths_2nd_2_1g, paths_2nd_2_1i)

points_2nd_2_2 = np.array([[0, 1], [0, 3], [1, 1], [2, 3], [3, 1], [3, 3]])
paths_2nd_2_2g = np.array([[1, 3], [2, 4], [3, 5], [4, 6]])
paths_2nd_2_2i = np.array([[3, 4]])
paths_2nd_2_2 = combine_paths(paths_2nd_2_2g, paths_2nd_2_2i)

points_2nd_2_3 = np.array([[0, 1], [0, 3], [1, 3], [2, 1], [3, 1], [3, 3]])
paths_2nd_2_3g = np.array([[1, 4], [2, 3], [3, 6], [4, 5]])
paths_2nd_2_3i = np.array([[3, 4]])
paths_2nd_2_3 = combine_paths(paths_2nd_2_3g, paths_2nd_2_3i)

points_2nd_2_4 = np.array([[0, 1], [0, 3], [1, 1], [2, 3], [3, 1], [3, 3]])
paths_2nd_2_4g = np.array([[1, 4], [2, 3], [3, 5], [4, 6]])
paths_2nd_2_4i = np.array([[3, 4]])
paths_2nd_2_4 = combine_paths(paths_2nd_2_4g, paths_2nd_2_4i)

points_2nd_2_5 = np.array([[0, 1], [0, 3], [1, 3], [2, 1], [3, 1], [3, 3]])
paths_2nd_2_5g = np.array([[1, 3], [2, 4], [3, 6], [4, 5]])
paths_2nd_2_5i = np.array([[3, 4]])
paths_2nd_2_5 = combine_paths(paths_2nd_2_5g, paths_2nd_2_5i)

points_2nd_3_1 = np.array([[0, 1], [2, 3], [3, 4], [4, 1], [4, 3], [4, 5]])
paths_2nd_3_1g = np.array([[1, 2], [2, 4], [3, 5], [3, 6]])
paths_2nd_3_1i = np.array([[2, 3]])
paths_2nd_3_1 = combine_paths(paths_2nd_3_1g, paths_2nd_3_1i)

points_2nd_3_2 = np.array([[0, 1], [2, 3], [3, 2], [4, 1], [4, 3], [4, 5]])
paths_2nd_3_2g = np.array([[1, 2], [3, 4], [3, 5], [2, 6]])
paths_2nd_3_2i = np.array([[2, 3]])
paths_2nd_3_2 = combine_paths(paths_2nd_3_2g, paths_2nd_3_2i)

points_2nd_3_3 = np.array([[0, 1], [2, 3], [3, 4], [4, 1], [4, 3], [4, 5]])
paths_2nd_3_3g = np.array([[1, 2], [3, 4], [2, 5], [3, 6]])
paths_2nd_3_3i = np.array([[2, 3]])
paths_2nd_3_3 = combine_paths(paths_2nd_3_3g, paths_2nd_3_3i)

can_points_2nd = np.empty((11, max(len(points_2nd_1_1), len(points_2nd_1_2), len(points_2nd_1_3)), 2))
can_points_2nd[0] = points_2nd_1_1
can_points_2nd[1] = points_2nd_1_2
can_points_2nd[2] = points_2nd_1_3
can_points_2nd[3] = points_2nd_2_1
can_points_2nd[4] = points_2nd_2_2
can_points_2nd[5] = points_2nd_2_3
can_points_2nd[6] = points_2nd_2_4
can_points_2nd[7] = points_2nd_2_5
can_points_2nd[8] = points_2nd_3_1
can_points_2nd[9] = points_2nd_3_2
can_points_2nd[10] = points_2nd_3_3

can_paths_2nd = np.empty((11, max(len(paths_2nd_1_1), len(paths_2nd_1_2), len(paths_2nd_1_3)), max(len(paths_2nd_1_1[0]), len(paths_2nd_1_2[0]), len(paths_2nd_1_3[0])), 2), dtype=int)
can_paths_2nd[0] = paths_2nd_1_1
can_paths_2nd[1] = paths_2nd_1_2
can_paths_2nd[2] = paths_2nd_1_3
can_paths_2nd[3] = paths_2nd_2_1
can_paths_2nd[4] = paths_2nd_2_2
can_paths_2nd[5] = paths_2nd_2_3
can_paths_2nd[6] = paths_2nd_2_4
can_paths_2nd[7] = paths_2nd_2_5
can_paths_2nd[8] = paths_2nd_3_1
can_paths_2nd[9] = paths_2nd_3_2
can_paths_2nd[10] = paths_2nd_3_3

can_number_2nd = np.array([[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]])

can_points = [can_points_1st, can_points_2nd]
can_paths = [can_paths_1st, can_paths_2nd]
can_count = [can_number_1st, can_number_2nd]

def return_counterterm_diagrams(order):
    """
    Generate counterterm diagrams up to the specified perturbative order,
    using NumPy arrays for better performance and clarity.
    """
    all_points = []
    all_paths = []

    for i in range(2, order + 1):
        points_orders = []
        paths_orders = []

        if i > 3:
            j = i - 4
            if j >= 0 and j < len(all_points):  # safeguard
                points_orders.extend(all_points[j])
                paths_orders.extend(all_paths[j])

        for j in range(1, i):
            k = i - j

            # Create points:
            # Outgoing: [1, 1], ..., [1, j]
            out_pts = np.stack([np.full(j, 1), np.arange(1, j + 1)], axis=1)

            # Counterterm: [2, 1]
            ct_pt = np.array([[2, 1]])

            # Incoming: [3, 1], ..., [3, k]
            in_pts = np.stack([np.full(k, 3), np.arange(1, k + 1)], axis=1)

            points = np.vstack([out_pts, ct_pt, in_pts])
            points_orders.append(points)

            # Create paths:
            # Outgoing: [1, j+1], ..., [j, j+1]
            out_paths = np.stack([np.arange(1, j + 1), np.full(j, j + 1)], axis=1)

            # Counterterm: [j+1, j+1]
            ct_path = np.array([[j + 1, j + 1]])

            # Incoming: [j+1, j+2], ..., [j+1, j+k+1]
            in_paths = np.stack([np.full(k, j + 1), np.arange(j + 2, j + k + 2)], axis=1)

            paths = np.vstack([out_paths, ct_path, in_paths])
            paths_orders.append(paths)

        all_points.append(points_orders)
        all_paths.append(paths_orders)

    return all_points, all_paths
