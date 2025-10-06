import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.special import binom, factorial
from collections import deque

#Import the canoncial diagrams
from functions.can_diagrams.gluon_diagrams import *

#From Eddmik in https://stackoverflow.com/questions/34593824/trim-strip-zeros-of-a-numpy-array
#Note that for numpy 2.2 the function trim_zeros is available for ndarrays.
#I may rewirte this function as i would have done it myself, but for now we will use this one.
"""
def trim_zeros_2D(array, axis=1):
    mask = ~(array==0).all(axis=axis)
    inv_mask = mask[::-1]
    start_idx = np.argmax(mask == True)
    end_idx = len(inv_mask) - np.argmax(inv_mask == True)
    if axis:
        return array[start_idx:end_idx,:]
    else:
        return array[:, start_idx:end_idx]
"""    
#From Chatgpt
def trim_zeros_2D(array: np.ndarray, axis: int = 1) -> np.ndarray:
    """
    Remove any rows (axis=1) or columns (axis=0) that are entirely zero,
    even if they appear between non-zero rows/columns.
    
    Parameters
    ----------
    array : np.ndarray
        2D input array.
    axis : int, optional
        If 1 (default), drop zero-rows; if 0, drop zero-columns.
    
    Returns
    -------
    np.ndarray
        The trimmed array.
    """
    # mask[i] is True iff the i-th row/column has at least one non-zero
    mask = array.any(axis=axis)
    
    if axis:
        # drop rows where mask is False
        return array[mask, :]
    else:
        # drop columns where mask is False
        return array[:, mask]
    
#From chatgpt 
def trim_zeros_3D(array, axis=None):
    if axis is None:
        # Trim along all axes
        mask = ~(array == 0).all(axis=(1, 2))
        trimmed_array = array[mask]

        mask = ~(trimmed_array == 0).all(axis=(0, 2))
        trimmed_array = trimmed_array[:, mask]

        mask = ~(trimmed_array == 0).all(axis=(0, 1))
        trimmed_array = trimmed_array[:, :, mask]
    elif axis == 0:
        # Trim along axis 0
        mask = ~(array == 0).all(axis=(1, 2))
        trimmed_array = array[mask]
    elif axis == 1:
        # Trim along axis 1
        mask = ~(array == 0).all(axis=(0, 2))
        trimmed_array = array[:, mask]
    elif axis == 2:
        # Trim along axis 2
        mask = ~(array == 0).all(axis=(0, 1))
        trimmed_array = array[:, :, mask]
    else:
        raise ValueError("Invalid axis. Axis must be 0, 1, 2, or None.")
    
    return trimmed_array

#From Chatgpt
def find_equal_subarrays(array):
    """"
    Find the positions of duplicate subarrays in a 2D array.
    Args:
        array (np.array): 2D array to search for duplicates
    Returns:
        duplicate_positions (list): list of positions of duplicate
    """
    sorted_subarrays = [np.sort(subarray) for subarray in array]
    unique_subarrays, indices, counts = np.unique(sorted_subarrays, axis=0, return_index=True, return_counts=True)
    duplicate_positions = [np.where((sorted_subarrays == unique_subarrays[i]).all(axis=1))[0] for i in range(len(unique_subarrays)) if counts[i] > 1]
    return duplicate_positions

#From Chatgpt
def prune_points_and_reindex(points: np.ndarray,
                              paths: np.ndarray
                             ) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove all [0,0] rows from `points`, then rebuild `paths` so that:
      - any path entry pointing to a removed point becomes 0,
      - all other entries are remapped down to a compact 1-based range.
    
    Parameters
    ----------
    points : np.ndarray, shape (N,2)
        Your (x,y) coordinates, with placeholder rows exactly equal to [0,0].
    paths : np.ndarray, shape (T,P,2), dtype=int
        Your 1-based index pairs, with [0,0] as placeholders.
    
    Returns
    -------
    new_points : np.ndarray, shape (M,2)
        The pruned points (no [0,0] rows).
    new_paths  : np.ndarray, shape (T,P,2)
        The updated paths, still 1-based with [0,0] placeholders.
    """
    # 1) Mask of rows to keep
    keep = ~(np.all(points == 0, axis=1))
    new_points = points[keep]
    
    # 2) Build old→new 1-based map
    #    new_idx[i] = new 1-based index of old row i, or 0 if dropped
    new_idx = np.zeros(points.shape[0], dtype=int)
    new_idx[keep] = np.arange(1, keep.sum()+1)
    
    # 3) Apply to paths
    #    For each entry u in paths: if u>0, replace with new_idx[u-1], else keep 0
    T, P, _ = paths.shape
    flat = paths.reshape(-1,2)
    # map both columns at once:
    mapped = np.zeros_like(flat)
    for col in (0,1):
        # grab the column, subtract 1 for 0-based indexing into new_idx
        o = flat[:,col]
        # for non-zero entries, look up new index; zeros stay zero
        mapped[:,col] = np.where(o>0, new_idx[o-1], 0)
    new_paths = mapped.reshape(T, P, 2)
    
    return new_points, new_paths

def represent_diagram (points, all_paths, index = False, directory = "", colors = ["tab:blue", "tab:red", "black"], line = ["solid", "solid", "photon"], count = 0):
    """
    Represent a diagram with points and paths.
    Args:
        points (np.array): array of points of dimension (n, 2)
        all_paths (list): list of paths to represent
        index (bool): whether to show the index of each point
        directory (str): directory to save the diagram
        colors (list): list of colors for each path
        line (list): list of line styles for each path
    """
    if (np.all(all_paths == 0)):
        return 
    
    points, all_paths = prune_points_and_reindex(points, all_paths)

    points = trim_zeros_2D(points)
    all_paths = trim_zeros_3D(all_paths, axis=1)
    
    fig=plt.figure(figsize=(5,3)) 
    ax=fig.add_subplot(111)
    ax.axis('off')
    j = 0

    # Note the that here the paths are more similar to the 1 particle case, meaning that is a 2D array
    for paths in all_paths:
        loops = find_equal_subarrays(paths)
        # Following the point made previously len(paths) indicate the number of connections instead of number of types of particles.
        for i in range(len(paths)):
            # In the case that the type of particle is a photon a spetial type of line is used to represent it.
            if (line[j] == "photon"):
                with mpl.rc_context({'path.sketch': (3, 15, 1)}):
                    if np.isin(i, loops):
                        middle_point = (points[paths[i, 0]-1] + points[paths[i, 1]-1]) / 2
                        circle = plt.Circle((middle_point[0], middle_point[1]), np.linalg.norm(points[paths[i, 0]-1]-middle_point), color=colors[j], fill=False)
                        ax.add_patch(circle)
                    elif paths[i, 0] == paths[i, 1] and paths[i ,0] != 0:
                        ax.scatter(points[paths[i, 0]-1, 0], points[paths[i, 0]-1, 1], color = colors[j])
                    else:
                        ax.plot([points[paths[i, 0]-1, 0], points[paths[i, 1]-1, 0]], [points[paths[i, 0]-1, 1], points[paths[i, 1]-1, 1]], color=colors[j])
            else:
                if np.isin(i, loops):
                    middle_point = (points[paths[i, 0]-1] + points[paths[i, 1]-1]) / 2
                    circle = plt.Circle((middle_point[0], middle_point[1]), np.linalg.norm(points[paths[i, 0]-1]-middle_point), color=colors[j], fill=False, linestyle=line[j])
                    ax.add_patch(circle)
                elif paths[i, 0] == paths[i, 1] and paths[i ,0] != 0:
                    ax.scatter(points[paths[i, 0]-1, 0], points[paths[i, 0]-1, 1], color = colors[j], s = 50, zorder = 10)
                else:
                    ax.plot([points[paths[i, 0]-1, 0], points[paths[i, 1]-1, 0]], [points[paths[i, 0]-1, 1], points[paths[i, 1]-1, 1]], color=colors[j], linestyle=line[j])
        j+=1
    ax.axis('equal')
    if index:
        for i in range(len(points)):
            ax.text(points[i, 0], points[i, 1], str(i+1), fontsize=12, color="black", ha="right", va="top")
    if count !=0:
        ax.text(0.5, 0.5, f"N = {count}", fontsize=12, color="black", ha="center", va="center")
    if directory != "":
        plt.savefig(directory, bbox_inches='tight')
        plt.close() #Added to not show in the notebook 

def unique_values(array):
    unique, counts = np.unique(array, return_counts=True)
    unique_values = unique[counts == 1]
    return unique_values

def in_out_paths (paths):
    max_len = max([len(path) for path in paths])
    #len(paths) is the number of type of particles
    in_out_paths = np.zeros((len(paths), 2, max_len), dtype=int)
    unique_vals = unique_values(paths.flatten())
    for i in range(len(paths)):
        inp = 0
        out = 0
        for j in range(max_len):
            if paths[i, j, 0] in unique_vals:
                in_out_paths[i, 1, out] = paths[i, j, 0]
                out += 1
            if paths[i, j, 1] in unique_vals:
                in_out_paths[i, 0, inp] = paths[i, j, 1]
                inp += 1
    in_out_paths = trim_zeros_3D(in_out_paths, axis=2)
    return in_out_paths

def how_connected( max_connections, n_connections, n_1, n_2):
    """
    Generate all possible combinations of connections between two diagrams.
    Args:
        max_connections (int): maximum number of connections between the two diagrams for each type of particle
        n_connections (int): number of connections between the two diagrams taking into account the number of types of particles
        n_1 (int): number of input for each type of particle
        n_2 (int): number of output for each type of particle
    Returns:
        combinations (np.array): array of all possible combinations of connections between the two diagrams
        of dimension (n_connections, max_connections, 2)
    """
    combinations = np.zeros((n_connections, max_connections, 2), dtype=int)
    n = 0 
    while n < n_connections:
        for j in range (n_1):
            for k in range(n_2):
                combinations[n, 0] = np.array([j+1, k+1])
                n += 1
                if n == n_connections:
                    break
            if n == n_connections:
                break
            
    n = n_1*n_2
    if max_connections >1:
        while n < n_connections:
            diff = False
            i = 1
            while i < max_connections:  
                for j in range (n_1):
                    for k in range(n_2):
                        for l in range(i):
                            if (j+1) != combinations[n, l, 0] and (k+1) != combinations[n, l, 1]:
                                diff = True
                            else:
                                diff = False
                        if diff:
                            combinations[n, i] = np.array([j+1, k+1])
                            n +=1   
                        if n == n_connections:
                            return combinations      
                i += 1
    else:
        return combinations

def connection (points1, paths1, points2, paths2, offset = 0):
    in_out_paths1 = in_out_paths(paths1)
    in_out_paths2 = in_out_paths(paths2)

    n_types = len(in_out_paths1)

    #Create the new points array
    points = np.zeros((len(points1) + len(points2), 2))
    points[:len(points1)] = points1
    points[len(points1):] = points2 + np.array([np.max(points1)+1, offset])

    #Displace the paths of the second diagram to rename the points
    for i in range(n_types):
        for j in range(len(in_out_paths2[0])):
            for k in range(len(in_out_paths2[0, 0])):
                if in_out_paths2[i, j, k] != 0:
                    in_out_paths2[i, j, k] += len(points1)

    #n1 and n2 indicate the number of input for each type of particle and output for each type of particle
    n1 = np.zeros(n_types, dtype=int)
    n2 = np.zeros(n_types, dtype=int)
    for i in range(n_types):
        n1[i] = len(np.trim_zeros(in_out_paths1[i, 0]))
        n2[i] = len(np.trim_zeros(in_out_paths2[i, 1]))

    #max_connections indicates the maximum number of connections between the two diagrams for each type of particle
    max_connections = np.zeros(n_types, dtype=int)
    for i in range(n_types):
        max_connections[i] = min(n1[i], n2[i])

    #n_connections indicates the number of connections between the two diagrams taking into account the number of types of particles
    n_connections = np.zeros(n_types, dtype=int)
    for j in range(n_types):
        for i in range(int(max_connections[j])):
            n_connections[j] += int(binom(n1[j], i+1)*binom(n2[j], i+1) * factorial(i+1))
    
    #n_connec indicates the total number of connections between the two diagrams
    n_connec = 0
    for subset in range(1, 1 << n_types):
        product = 1
        for i in range(n_types):
            if subset & (1 << i):
                product *= n_connections[i]
        n_connec += product

    #Use a dummy array to store all possible combinations of connections for each type of particle between the two diagrams
    dummy_combinations = np.zeros((sum(n_connections), n_types,  max(max_connections), 2), dtype=int)
    n = 0
    for i in range(n_types):
        dummy_var = how_connected(max_connections[i], n_connections[i], n1[i], n2[i])
        for j in range(n_connections[i]):
            for k in range(max_connections[i]):
                if(dummy_var[j, k, 0] != 0 and dummy_var[j, k, 1] != 0):
                    dummy_combinations[n, i, k] = dummy_var[j, k]
                else:
                    break
            n+=1

    #From the dummy array, create the array combinations that will store all possible combinations of connections between the two diagrams
    #taking into account mixing different types of particles
    combinations = np.zeros((n_connec, n_types, max(max_connections), 2), dtype=int)

    #The first step is to store a copy of the dummy array in the combinations array, without considering the mixing of different 
    #types of particles since there will be diagrams without mixed particles.
    n = 0
    for i in range(n_types):
        if (n_connections[i] == 0):
            continue
        for j in range(np.sum(n_connections[:i]), n_connections[i]+np.sum(n_connections[:i])):
            for k in range(max_connections[i]):
                if(dummy_combinations[j, i, k, 0] != 0 and dummy_combinations[j, i, k, 1] != 0):
                    combinations[n, i, k] = dummy_combinations[j, i, k]
                else:
                    break
            n+=1        

    #The second step is to store the combinations of connections between the two diagrams taking into account the mixing of 
    #different types of particles. The process could be though as filling a tringular matrix with the combinations of connections
    #between the two diagrams. The first row of the matrix corresponds to the one type case, the second row combining 2 elements of 
    #the first row, this means combining 2 types of particles, and so on.

    #The variable n_start indicates the position in the combinations array where the combinations of connections between 
    #the two diagrams taking into account the mixing of different types of particles start.
    n_start = n

    n_prime = 0
    for i in range(n_types-1):
        leng = 0
        for l in range(i+1, n_types):   
            leng += n_connections[i]*n_connections[l]
        for n in range(n_start, leng+n_start):
            combinations[n, i] = dummy_combinations[n_prime, i]
            for j in range(i+1, n_types):
                combinations[n, j] = dummy_combinations[n-n_start+n_connections[i], j]
        n_prime += 1    
    #Create the paths array that will store the connections between the two diagrams.
    paths = np.zeros((n_connec, n_types, len(paths1[0]) + len(paths2[0]) + max(max_connections), 2), dtype=int)
    paths[:n_connec,:n_types,:len(paths1[0])] = paths1
    for i in range(n_connec):
        for j in range(n_types):
            for k in range(max_connections[j]):
                if (combinations[i,j, k, 0] != 0 and combinations[i,j, k, 1] != 0):
                    paths[i,j, len(paths1[0])+k] = np.array([in_out_paths1[j,0, combinations[i, j, k, 0]-1], in_out_paths2[j,1, combinations[i, j, k, 1]-1]])
            if (np.count_nonzero(paths2[j]) != 0):
                for k in range(len(paths2[j])):
                    if (paths2[j, k, 0] != 0 and paths2[j, k, 1] != 0):
                        paths[i,j, len(paths1[0])+max(max_connections)+k] = paths2[j, k] + np.array([len(points1), len(points1)])

    return points, paths

def decrement_number_in_array(array, number):
    array[array == number] -= 1
    return array

def simplify_diagram_it (points, paths):
    """
    Function that will be iterated to simplify the diagram by removing the points and paths that are not needed.
    """
    pos = np.zeros((2, 3), dtype=int)
    for i in range(1, np.max(paths)+1):
        count = 0
        for j in range(len(paths)):
            for k in range(len(paths[j])):
                for l in range(2):
                    if paths[j, k, l] == i:
                        count += 1
                        if count == 1:
                            pos[0] = np.array([j, k, l])
                        elif count == 2:
                            pos[1] = np.array([j, k, l])
                        else:
                            break
        if count == 2 and pos[0, 0] == pos[1, 0]:
            j = pos[0, 0] # type of particle
            points = np.delete(points, i-1, axis=0)
            if pos[0, 2] == 0:
                if pos[1, 2] == 0:
                    prov = np.array([paths[j, pos[0, 1], 1], paths[j, pos[1, 1], 1]])
                elif pos[1, 2] == 1:
                    prov = np.array([paths[j, pos[0, 1], 1], paths[j, pos[1, 1], 0]])  
            elif pos[0, 2] == 1:
                if pos[1, 2] == 0:
                    prov = np.array([paths[j, pos[0, 1], 0], paths[j, pos[1, 1], 1]])                  
                elif pos[1, 1] == 1:
                    prov = np.array([paths[j, pos[0, 1], 0], paths[j, pos[1, 1], 0]])
            paths[j, pos[1, 1]] = np.array([0, 0])
            paths[j, pos[0, 1]] = prov
                    
            for k in range(i, np.max(paths)+1):
                paths = decrement_number_in_array(paths, k)

    return points, trim_zeros_3D(paths, axis=1)

def simplify_diagram (points, paths):
    """
    Simplify the diagram by removing the points and paths that are not needed, by iterating the function simplify_diagram_it, until the 
    number of points and paths does not change anymore.
    """
    new_points, new_paths = simplify_diagram_it(points, paths)
    new_new_points, new_new_paths = simplify_diagram_it(new_points, new_paths)
    while len(new_points) != len(new_new_points) or len(new_paths) != len(new_new_paths):
        new_points, new_paths = new_new_points, new_new_paths
        new_new_points, new_new_paths = simplify_diagram_it(new_points, new_paths)
    return new_new_points, new_new_paths

def return_counterterm_diagrams (order):
    """
    Return all the counterterm diagrams up till a given order.
    """
    #initialize the list of points and paths where the diagrams will be stored, and returned at the end.
    all_points = []
    all_paths = []
    #iterate through the order in perturbation (variable i)
    for i in range(2, order + 1):
        #initialize the list of points and paths for this order
        points_orders = []
        paths_orders = []
        #at each order, compute the counterterm diagrams
        #where j is the number of outgoing particles

        if (i > 3):
            if (i%2 == 0):
                j = i-4
                for l in range(len(all_points[j])):
                    points_orders.append(all_points[j][l])
                for l in range(len(all_paths[j])):
                    paths_orders.append(all_paths[j][l])

            if (i%2 == 1):
                j = i-4
                for l in range(len(all_points[j])):
                    points_orders.append(all_points[j][l])
                for l in range(len(all_paths[j])):
                    paths_orders.append(all_paths[j][l])

        for j in range(1, i):
            """
            Starting with the points for the diagrams.
            """
            #initialize a dummy array variable for each diagram
            points = []
            #k is the number of incoming particles
            k = i - j
            #create the outgoing points for the diagram
            for l in range(1, j+1):
                points.append([1, l])
            #create the point where the counterterm is located
            points.append([2, 1])
            #create the incoming points for the diagram
            for l in range(1, k+1):
                points.append([3, l])
            #append the points for this diagram
            points_orders.append(points)

            """
            Following a similar procedure for the paths.
            """
            #initialize a dummy array variable for each path
            paths = []
            for l in range(1, j+1):
                #create the outgoing paths for the diagram
                paths.append([l, j+1])
            paths.append([j+1, j+1])
            for l in range(1, k+1):
                #create the incoming paths for the diagram
                paths.append([j+1, j+l+1])
            #append the paths for this diagram
            paths_orders.append(paths)
            
        #append the points for this order
        all_points.append(points_orders)

        #append the paths for this order
        all_paths.append(paths_orders)
    
    return all_points, all_paths

def combine_diagrams_order (points, paths, count, typeofproc, max_order, offset = 0):
    """
    Combine all the diagrams that would contribute to a give order, and type of interaction. If the order that is calculating is 
    the max_order intended, then it will try to skip products that definitely do not contribute.
    """

    #Similar to the one type of particle case, we need to first approximate the number of new diagrams that will produce.
    curr_order = len(points)
    n_types = len(paths[0][0])
    max_points = np.zeros((n_types, 2), dtype=int)

    n1 = np.zeros(n_types, dtype=int)
    n2 = np.zeros(n_types, dtype=int)
    for i in range(len(paths[0])): #in number of first order diagrams
        for j in range(n_types):
            n1[j] = len(np.trim_zeros(in_out_paths(paths[0][i])[j, 0]))
            if (n1[j] > max_points[j, 0]):
                max_points[j, 0] = n1[j]
            n2[j] = len(np.trim_zeros(in_out_paths(paths[0][i])[j, 1]))
            if (n2[j] > max_points[j, 1]):
                max_points[j, 1] = n2[j]            
    for i in range(len(paths[-1])):
        for j in range(n_types):
            n1[j] = len(np.trim_zeros(in_out_paths(paths[-1][i])[j, 0]))
            if (n1[j] > max_points[j, 0]):
                max_points[j, 0] = n1[j]
            n2[j] = len(np.trim_zeros(in_out_paths(paths[-1][i])[j, 1]))
            if (n2[j] > max_points[j, 1]):
                max_points[j, 1] = n2[j]

    max_connections = np.zeros(n_types, dtype=int)
    for i in range(n_types):
        max_connections[i] = min(max_points[i, 0], max_points[i, 1])

    n_connec = 0
    n_connections = np.zeros(n_types, dtype=int)
    for i in range(len(paths[0])):
        for j in range(n_types):
            for k in range(int(max_connections[j])):
                n_connections[j] += int(binom(n1[j], i+1)*binom(n2[j], k+1) * factorial(k+1))
        #n_connec indicates the total number of connections between the two diagrams
        for subset in range(1, 1 << n_types):
            product = 1
            for i in range(n_types):
                if subset & (1 << k):
                    product *= n_connections[k]
            n_connec += product
    n = 0
    f = 20

    new_points = np.zeros((4*n_types*len(paths[0])*len(paths[-1])*n_connec*(curr_order+1), len(points[0][0]) + len(points[-1][0]), 2))
    new_paths = np.zeros((4*n_types*len(paths[0])*len(paths[-1])*n_connec*(curr_order+1), n_types, len(paths[0][0]) + len(paths[-1][0])+np.max(max_connections)+5, 2), dtype=int)
    new_count = np.zeros((4*n_types*len(paths[0])*len(paths[-1])*n_connec*(curr_order+1), 1), dtype=int)

    counter_points, counter_paths = return_counterterm_diagrams(max_order+1)

    for i in range(len(paths[0])):
        for j in range(len(paths[-1])):
            dummy_points, dummy_paths = connection(trim_zeros_2D(points[-1][j]), trim_zeros_3D(paths[-1][j], axis = 1),trim_zeros_2D(points[0][i]), trim_zeros_3D(paths[0][i], axis=1), offset=offset)
            for k in range(len(dummy_paths)):
                if(curr_order+1  == max_order): 
                    in_out_path = in_out_paths(dummy_paths[k])
                    if (len(np.trim_zeros(in_out_path[0, 0])) != typeofproc[0][1] or len(np.trim_zeros(in_out_path[0, 1]))!= typeofproc[0][0]):
                        continue
                simp_points, simp_paths = simplify_diagram(dummy_points, trim_zeros_3D(dummy_paths[k], axis=1))
                for l in range(len(simp_points)):
                    new_points[n, l] = simp_points[l]
                for l in range(n_types):
                    for m in range(len(simp_paths[0])):
                        new_paths[n, l, m] = simp_paths[l, m]
                new_count[n] = count[0][i] * count[-1][j]
                n += 1
            if (curr_order-1 < len(can_paths)):
                dummy_points, dummy_paths = connection(trim_zeros_2D(points[0][i]), trim_zeros_3D(paths[0][i], axis=1),trim_zeros_2D(points[-1][j]), trim_zeros_3D(paths[-1][j], axis = 1), offset=offset)
                for k in range(len(dummy_paths)):
                    if(curr_order+1  == max_order): 
                        in_out_path = in_out_paths(dummy_paths[k])
                        if (len(np.trim_zeros(in_out_path[0, 0])) != typeofproc[0][1] or len(np.trim_zeros(in_out_path[0, 1]))!= typeofproc[0][0]):
                            continue
                    simp_points, simp_paths = simplify_diagram(dummy_points, trim_zeros_3D(dummy_paths[k], axis=1))
                    for l in range(len(simp_points)):
                        new_points[n, l] = simp_points[l]
                    for l in range(n_types):
                        for m in range(len(simp_paths[0])):
                            new_paths[n, l, m] = simp_paths[l, m]
                    new_count[n] = count[0][i] * count[-1][j]
                    n += 1
    for i in range(1, 2):
        for j in range(i-1, len(paths)):
            if (i+1 + j) == curr_order:
                for k in range(len(can_paths[i])):
                    for l in range(len(paths[j])):
                        dummy_points, dummy_paths = connection(trim_zeros_2D(can_points[i][k]), trim_zeros_3D(can_paths[i][k], axis=1), trim_zeros_2D(points[j][l]), trim_zeros_3D(paths[j][l], axis = 1), offset=offset)
                        for m in range(len(dummy_paths)):
                            if(curr_order+1  == max_order): 
                                in_out_path = in_out_paths(dummy_paths[m])
                                if (len(np.trim_zeros(in_out_path[0, 0])) != typeofproc[0][0] or len(np.trim_zeros(in_out_path[0, 1]))!= typeofproc[0][1]):
                                    continue
                            simp_points, simp_paths = simplify_diagram(dummy_points, trim_zeros_3D(dummy_paths[m], axis=1))
                            for o in range(len(simp_points)):
                                new_points[n, o] = simp_points[o]
                            for o in range(n_types):
                                for p in range(len(simp_paths[0])):
                                    new_paths[n, o, p] = simp_paths[o, p]
                            new_count[n] = can_count[i][k] * count[j][l]        
                            n += 1
                for k in range(len(can_paths[i])):
                    for l in range(len(paths[j])):
                        dummy_points, dummy_paths = connection(trim_zeros_2D(points[j][l]), trim_zeros_3D(paths[j][l], axis=1), trim_zeros_2D(can_points[i][k]), trim_zeros_3D(can_paths[i][k], axis = 1), offset=offset)
                        for m in range(len(dummy_paths)):
                            if(curr_order+1  == max_order): 
                                in_out_path = in_out_paths(dummy_paths[m])
                                if (len(np.trim_zeros(in_out_path[0, 0])) != typeofproc[0][0] or len(np.trim_zeros(in_out_path[0, 1]))!= typeofproc[0][1]):
                                    continue
                            simp_points, simp_paths = simplify_diagram(dummy_points, trim_zeros_3D(dummy_paths[m], axis=1))
                            for o in range(len(simp_points)):
                                new_points[n, o] = simp_points[o]
                            for o in range(n_types):
                                for p in range(len(simp_paths[0])):
                                    new_paths[n, o, p] = simp_paths[o, p]
                            new_count[n] = can_count[i][k] * count[j][l] 
                            n += 1

    for i in range(len(counter_points)):
        for j in range(i, len(paths)):
            if (i+2+ j +1) == curr_order:
                for k in range(len(counter_points[i])):
                    for l in range(len(paths[j])):
                        dummy_points, dummy_paths = connection(trim_zeros_2D(np.array(counter_points[i][k])), trim_zeros_3D(np.array([counter_paths[i][k],np.zeros_like(counter_paths[i][k])]), axis=1), trim_zeros_2D(points[j][l]), trim_zeros_3D(paths[j][l], axis = 1), offset=offset)
                        for m in range(len(dummy_paths)):
                            if(curr_order+1  == max_order): 
                                in_out_path = in_out_paths(dummy_paths[m])
                                if (len(np.trim_zeros(in_out_path[0, 0])) != typeofproc[0][0] or len(np.trim_zeros(in_out_path[0, 1]))!= typeofproc[0][1]):
                                    continue
                            simp_points, simp_paths = simplify_diagram(dummy_points, trim_zeros_3D(dummy_paths[m], axis=1))
                            for o in range(len(simp_points)):
                                new_points[n, o] = simp_points[o]
                            for o in range(n_types):
                                for p in range(len(simp_paths[0])):
                                    new_paths[n, o, p] = simp_paths[o, p]
                            new_count[n] = can_count[i][k] * count[j][l]        
                            n += 1
                for k in range(len(counter_points[i])):
                    for l in range(len(paths[j])):
                        dummy_points, dummy_paths = connection(trim_zeros_2D(points[j][l]), trim_zeros_3D(paths[j][l], axis=1), trim_zeros_2D(np.array(counter_points[i][k])), trim_zeros_3D(np.array([counter_paths[i][k],np.zeros_like(counter_paths[i][k])]), axis=1), offset=offset)
                        for m in range(len(dummy_paths)):
                            if(curr_order+1  == max_order): 
                                in_out_path = in_out_paths(dummy_paths[m])
                                if (len(np.trim_zeros(in_out_path[0, 0])) != typeofproc[0][0] or len(np.trim_zeros(in_out_path[0, 1]))!= typeofproc[0][1]):
                                    continue
                            simp_points, simp_paths = simplify_diagram(dummy_points, trim_zeros_3D(dummy_paths[m], axis=1))
                            for o in range(len(simp_points)):
                                new_points[n, o] = simp_points[o]
                            for o in range(n_types):
                                for p in range(len(simp_paths[0])):
                                    new_paths[n, o, p] = simp_paths[o, p]
                            new_count[n] = can_count[i][k] * count[j][l] 
                            n += 1 

                
    #In the case of the gluon diagrams, there is only it second order diagrams, so this will only be used for curr_order = 1, but it should be general, for the cases where 
    #there are higher order canonical diagrams.
    if curr_order < len(can_paths):
        for i in range(len(can_points[curr_order])):
            for j in range(len(can_points[curr_order][i])):
                new_points[n, j] = can_points[curr_order][i][j]
            for j in range(n_types):
                for k in range(len(can_paths[curr_order][i][j])):
                    new_paths[n, j, k] = can_paths[curr_order][i][j][k]
            new_count[n] = can_count[curr_order][i][0]
            n += 1
    
    if curr_order < len(counter_points):
        for i in range(len(counter_points[curr_order-1])):
            for j in range(len(counter_points[curr_order-1][i])):
                new_points[n, j] = counter_points[curr_order-1][i][j]
            for j in range(n_types):
                for k in range(len(counter_paths[curr_order-1][i])):
                    if j == 0:
                        new_paths[n, j, k] = counter_paths[curr_order-1][i][k]
                    else:
                        new_paths[n, j, k] = np.zeros(2, dtype=int)
            new_count[n] = 0
            n += 1
    
    return new_points, new_paths, new_count

#From Chatgpt
def diagram_signature(paths: np.ndarray) -> tuple:
    """
    paths: (K, M, 2)
    returns: a tuple of length K, where each element is
             a flattened tuple of that row's points sorted lexicographically.
    """
    sig = []
    for row in paths:
        # sort points by (x, then y)
        idx = np.lexsort((row[:,1], row[:,0]))
        sorted_row = row[idx]
        sig.append(tuple(sorted_row.ravel()))
    return tuple(sig)

#From Chatgpt
def group_diagrams(points: np.ndarray,
                   paths: np.ndarray,
                   numbers: np.ndarray):
    """
    points:  (N, P, 2)
    paths:   (N, K, M, 2)
    numbers: (N, 1)
    """
    # 1) filter out the “all-zero” diagrams in one vectorized mask
    nonzero_mask = ~((paths == 0).all(axis=(1,2,3)))
    pts  = points [nonzero_mask]
    pths = paths  [nonzero_mask]
    nums = numbers[nonzero_mask, 0]

    # 2) single-pass grouping via a dict
    sig2group = {}
    grouped_pts  = []
    grouped_pths = []
    counts       = []

    for idx, (pt, pth, num) in enumerate(zip(pts, pths, nums)):
        sig = diagram_signature(pth)
        if sig in sig2group:
            gi = sig2group[sig]
            counts[gi] += num
        else:
            gi = len(grouped_pts)
            sig2group[sig] = gi
            grouped_pts .append(pt)
            grouped_pths.append(pth)
            counts    .append(num)

    # 3) stack into arrays
    G = len(grouped_pts)
    grouped_points = np.stack(grouped_pts, axis=0)   # (G, P, 2)
    grouped_paths  = np.stack(grouped_pths, axis=0)  # (G, K, M, 2)
    counts         = np.array(counts)                # (G,)

    return grouped_points, grouped_paths, counts

#From Chatgpt
def find_shortest_undirected_path(paths: np.ndarray, start: int, end: int):
    """
    Given:
      - paths: array of shape (T, P, 2), listing edges [u, v]
      - start: starting node
      - end:   target node
    Returns:
      - (num_hops, path_nodes)
        * num_hops: minimum number of edges
        * path_nodes: list of nodes [start, ..., end]
    Raises:
      ValueError if no route exists.
    """
    # 1) Build adjacency lists, preserving your input order
    adj = {}  # node -> list of neighbors in the order encountered
    T, P, _ = paths.shape
    for t in range(T):
        for p in range(P):
            u, v = int(paths[t, p, 0]), int(paths[t, p, 1])
            # skip placeholders or self‐loops
            if u == v or (u == 0 and v == 0):
                continue
            adj.setdefault(u, []).append(v)
            adj.setdefault(v, []).append(u)

    # 2) BFS to find shortest path
    visited = {start}
    queue = deque([(start, [start])])  # (current_node, path_so_far)
    while queue:
        node, path = queue.popleft()
        # explore neighbors in **exact** order they were added
        for nbr in adj.get(node, []):
            if nbr in visited:
                continue
            new_path = path + [nbr]
            if nbr == end:
                return len(new_path) - 1, new_path
            visited.add(nbr)
            queue.append((nbr, new_path))

    raise ValueError(f"No path found from {start} to {end}")

#From Chatgpt
def is_ordered_route(route):
    """
    Return True if `route` is strictly monotonic (either all increasing or all decreasing).
    Examples:
      [1, 3, 4, 5] → True (increasing)
      [6, 5, 3, 2] → True (decreasing)
      [1, 2, 2, 3] → False (not strict)
      [1, 3, 2]    → False (changes direction)
    """
    if len(route) < 2:
        return True
    # check strictly increasing
    inc = all(route[i] < route[i+1] for i in range(len(route)-1))
    # check strictly decreasing
    dec = all(route[i] > route[i+1] for i in range(len(route)-1))
    return inc or dec

#From Chatgpt
def find_same_height_outside(points: np.ndarray, route_ids: list[int]) -> list[int]:
    """
    Given:
      - points:  (N,2) array of (x,y) coords, zero-based indexing
      - route_ids: list of 1-based point IDs (from your path), all sharing the same y
    Returns:
      - list of 1-based IDs of all other points whose y == that common y
    """
    # convert to zero-based indices
    idxs = [rid - 1 for rid in route_ids]
    y_route = points[idxs, 1]
    if not np.allclose(y_route, y_route[0]):
        raise ValueError(f"Route points do not share one y: {y_route}")
    y0 = y_route[0]

    # mask all points at that y, then exclude route indices
    same_y = np.isclose(points[:, 1], y0)
    mask = same_y.copy()
    mask[idxs] = False

    outside_idxs = np.nonzero(mask)[0]
    # convert back to 1-based IDs
    return (outside_idxs + 1).tolist()

#From Chatgpt
def equalize_x_spacing(points: np.ndarray, spacing: float = 1.0) -> np.ndarray:
    """
    Return a new (N×2) array where the x-coords have been remapped so that
      - each unique original x is assigned to 0, spacing, 2*spacing, … in ascending order,
      - any two points that had the same x stay tied,
      - the y-coords are left unchanged.
    
    Parameters
    ----------
    points : np.ndarray of shape (N,2)
        Your original (x,y) coordinates.
    spacing : float, default=1.0
        The distance between consecutive unique x-positions.
    
    Returns
    -------
    new_pts : np.ndarray of shape (N,2)
        The transformed points.
    """
    # 1) find the sorted unique x-values
    orig_x = points[:,0]
    uniq = np.unique(orig_x)
    
    # 2) build a map: original x → new x
    #    e.g. uniq = [1.2, 3.4, 7.9]  →  {1.2:0, 3.4:1*spacing, 7.9:2*spacing}
    mapping = {x: i * spacing for i, x in enumerate(uniq)}
    
    # 3) apply it
    new_x = np.vectorize(mapping.get)(orig_x)
    new_pts = points.copy()
    new_pts[:,0] = new_x
    return new_pts

def reposition_diagram (points, in_out_path, paths, typeofproc, print_=False, spacing_=1.0):
    """
    Tries to avoid superposition using a different strategy depending on the type of interaction, e.g.
    - for [2, 2] interactions, it will try to find the 2 longest paths that connect each in-particle with each out-particle, 
    straightening both paths and keeping them at different height, while all the rest of paths connecting them will be inbetween.
    - for [2, 1] interactions, it will try to find which of the 2 particle from one side have the longest path to the other particle,
    making that one straight and at a certain height, making all other points out of said height.
    -etc.
    """
    minx_point = np.min(points[:, 0])
    maxx_point = np.max(points[:, 0])

    for i in range(len(in_out_path)):
        for j in range(len(in_out_path[i, 0])):
            if in_out_path[i, 0][j] == 0:
                continue
            if points[in_out_path[i, 0][j]-1, 0] == maxx_point:
                continue
            else:
                points[in_out_path[i, 0][j]-1, 1] = maxx_point - points[in_out_path[i, 0][j]-1, 0]
                points[in_out_path[i, 0][j]-1, 0] = maxx_point
        for j in range(len(in_out_path[i, 1])):
            if in_out_path[i, 1][j] == 0:
                continue
            if points[in_out_path[i, 1][j]-1, 0] == minx_point:
                continue
            else:
                points[in_out_path[i, 1][j]-1, 1] = - minx_point + points[in_out_path[i, 1][j]-1, 0]
                points[in_out_path[i, 1][j]-1, 0] = minx_point

    if typeofproc == [2, 2]:
        maxy_point = 5
        path1 = find_shortest_undirected_path(paths, in_out_path[0, 0][0], in_out_path[0, 1][0])
        path2 = find_shortest_undirected_path(paths, in_out_path[0, 0][1], in_out_path[0, 1][1])
        path3 = find_shortest_undirected_path(paths, in_out_path[0, 0][0], in_out_path[0, 1][1])
        path4 = find_shortest_undirected_path(paths, in_out_path[0, 0][1], in_out_path[0, 1][0])
        if path1[0] < path3[0]:
            if(print_):
                print(path2[1])
            for i in path1[1]:
                points[i-1, 1] = maxy_point
            if is_ordered_route(path1[1]) == False:
                for i in range(1, len(path1[1])-1):
                    points[path1[1][i]-1, 1] = (maxy_point-1)/2
            for i in path2[1]:
                points[i-1, 1] = 1
            for i in find_same_height_outside (points, path2[1]):
                points[i-1, 1] = (maxy_point-1)/2
        
        else:
            if(print_):
                print(path4[1])
            for i in path3[1]:
                points[i-1, 1] = maxy_point
            if is_ordered_route(path3[1]) == False:
                for i in range(1, len(path3[1])-1):
                    points[path3[1][i]-1, 1] = (maxy_point-1)/2
            for i in path4[1]:
                points[i-1, 1] = 1
            for i in find_same_height_outside (points, path4[1]):
                points[i-1, 1] = (maxy_point-1)/2
    elif typeofproc == [2, 1]:
        maxy_point = 5
        path1 = find_shortest_undirected_path(paths, in_out_path[0, 0][0], in_out_path[0, 1][0])
        path2 = find_shortest_undirected_path(paths, in_out_path[0, 0][0], in_out_path[0, 1][1])
        if path1[0] < path2[0]:
            for i in path1[1]:
                points[i-1, 1] = maxy_point
            if is_ordered_route(path1[1]) == False:
                for i in range(1, len(path1[1])-1):
                    points[path1[1][i]-1, 1] = (maxy_point-1)/2
            for i in path2[1]:
                points[i-1, 1] = 1
            for i in find_same_height_outside (points, path2[1]):
                points[i-1, 1] = (maxy_point-1)/2
        else:
            for i in path2[1]:
                points[i-1, 1] = maxy_point
            if is_ordered_route(path2[1]) == False:
                for i in range(1, len(path2[1])-1):
                    points[path2[1][i]-1, 1] = (maxy_point-1)/2
            for i in path1[1]:
                points[i-1, 1] = 1
            for i in find_same_height_outside (points, path1[1]):
                points[i-1, 1] = (maxy_point-1)/2
    elif typeofproc == [1, 2]:
        maxy_point = 3
        path1 = find_shortest_undirected_path(paths, in_out_path[0, 0][0], in_out_path[0, 1][0])
        path2 = find_shortest_undirected_path(paths, in_out_path[0, 0][1], in_out_path[0, 1][0])
        if path1[0] < path2[0]:
            for i in path1[1]:
                points[i-1, 1] = maxy_point
            if is_ordered_route(path1[1]) == False:
                for i in range(1, len(path1[1])-1):
                    points[path1[1][i]-1, 1] = (maxy_point-1)/2
            for i in path2[1]:
                points[i-1, 1] = 1
            for i in find_same_height_outside (points, path2[1]):
                points[i-1, 1] = (maxy_point-1)/2
        else:
            for i in path2[1]:
                points[i-1, 1] = maxy_point
            if is_ordered_route(path2[1]) == False:
                for i in range(1, len(path2[1])-1):
                    points[path2[1][i]-1, 1] = (maxy_point-1)/2
            for i in path1[1]:
                points[i-1, 1] = 1
            for i in find_same_height_outside (points, path1[1]):
                points[i-1, 1] = (maxy_point-1)/2
    elif typeofproc == [1, 1]:
        maxy_point = 3
        path1 = find_shortest_undirected_path(paths, in_out_path[0, 0][0], in_out_path[0, 1][0])
        if path1[0] < 2:
            for i in path1[1]:
                points[i-1, 1] = maxy_point
            if is_ordered_route(path1[1]) == False:
                for i in range(1, len(path1[1])-1):
                    points[path1[1][i]-1, 1] = maxy_point
        else:
            for i in path1[1]:
                points[i-1, 1] = 1
            for i in find_same_height_outside (points, path1[1]):
                points[i-1, 1] = maxy_point
        
    points = equalize_x_spacing(points, spacing=spacing_)

    return points

"""
3 particle loops are specially problematic when it comes to superpositions, using the method above 
we can not avoid, since making the "longest" path straight (2 of the paths), forces the 3rd path to be also
in the same direction.

So we need to detect the 3-particle loops and treat them separately.
"""

def detect_3p_loops(points, paths):
    """
    
    """
    loop = np.zeros((len(paths), int(len(trim_zeros_2D(paths[0]))/3), 3), dtype=int)
    n = 0
    for i in range(len(paths)):
        if np.count_nonzero(paths[i]) == 0:
            continue
        path = trim_zeros_2D(paths[i])
        if (len(path) < 3):
            continue
        index = np.arange(len(path))
        j = 0
        for j in range(len(path)):
            if j in index:
                start = path[j, 0]
                end = path[j, 1]
                next = -1
                for k in range(len(path)):
                    if(k != j and k in index):
                        if path[k, 0] == start:
                            next = path[k, 1]
                        elif path[k, 1] == start:
                            next = path[k, 0]
                        if next == -1:
                            continue
                        for l in range(len(path)):
                            next2 = -1
                            if (l != j and l!= k and l in index):
                                if path[l, 0] == next:
                                    next2 = path[l, 1]
                                elif path[l, 1] == next:
                                    next2 = path[l, 0]
                                if next2 == next:
                                    continue
                                if next2 == end:
                                    loop[i, n, 0] = j+1
                                    loop[i, n, 1] = k+1
                                    loop[i, n, 2] = l+1
                                    index = np.delete(index, np.where(index == j))
                                    index = np.delete(index, np.where(index == k))
                                    index = np.delete(index, np.where(index == l))

    return trim_zeros_3D(loop, axis=0)

def detect_superposition(points, paths):
    loop = detect_3p_loops(points, paths)
    if np.count_nonzero(loop) == 0:
        return points, paths  
    else:
        for i in range(len(loop)):
            if np.count_nonzero(loop[i]) == 0:
                continue
            for j in range(len(loop[i])):
                if np.count_nonzero(loop[i, j]) == 0:
                    continue
                else:
                    height = points[paths[i,loop[i, j, 0]-1, 0]-1, 1]
                    sorted_array = sorted([paths[i,loop[i, j, 0]-1, 0], paths[i,loop[i, j, 1]-1, 0], paths[i,loop[i, j, 2]-1, 0], paths[i,loop[i, j, 0]-1, 1], paths[i,loop[i, j, 1]-1, 1], paths[i,loop[i, j, 2]-1, 1]])
                    same_height = True
                    for k in sorted_array:
                        if points[k-1, 1] != height:
                            same_height = False
                            break
                    if same_height:
                        middle = sorted_array[2]-1
                        points[middle, 1] = height+1
                    else:
                        continue
        return points, paths
 
def represent_order(points, paths, count_, typeofproc, index_ = True,  lines_ = ["solid", "dotted"], colors_ = ["black", "black"], directory_ = "", docount = True, spacing = 1.0):
    """
    Represent all the diagrams calculated of a certain order, for a specific type of interaction.
    """
    n=1
    for i in range(len(points)):
        in_out_paths_ = in_out_paths(paths[i])
        inp = 0
        out = 0
        for j in range(len(paths[0])):
            inp += len(np.trim_zeros(in_out_paths_[j, 0]))
            out += len(np.trim_zeros(in_out_paths_[j, 1]))
        if typeofproc == []:
            points[i], paths[i] = detect_superposition(points[i], paths[i])
            points[i] = reposition_diagram(points[i], in_out_paths_, paths[i], [inp, out], spacing_=spacing)
            if docount:    
                if directory_ != "":
                    represent_diagram(points[i], paths[i], index=index_, line=lines_, colors=colors_, count=count_[i], directory=directory_+str(n))
                else:
                    represent_diagram(points[i], paths[i], index=index_, line=lines_, colors=colors_, count=count_[i])
            else:
                if directory_ != "":
                    represent_diagram(points[i], paths[i], index=index_, line=lines_, colors=colors_, count=0, directory=directory_+str(n))
                else:
                    represent_diagram(points[i], paths[i], index=index_, line=lines_, colors=colors_, count=0)
            n += 1
        elif inp == typeofproc[0][1] and out == typeofproc[0][0]:
            points[i], paths[i] = detect_superposition(points[i], paths[i])
            points[i] = reposition_diagram(points[i], in_out_paths_, paths[i], typeofproc[0], spacing_=spacing)
            if docount:
                if directory_ != "":
                    represent_diagram(points[i], paths[i], index=index_, line=lines_, colors=colors_, count=count_[i], directory=directory_+str(n))
                else:
                    represent_diagram(points[i], paths[i], index=index_, line=lines_, colors=colors_, count=count_[i])
            else:
                if directory_ != "":
                    represent_diagram(points[i], paths[i], index=index_, line=lines_, colors=colors_, count=0, directory=directory_+str(n))
                else:
                    represent_diagram(points[i], paths[i], index=index_, line=lines_, colors=colors_, count=0)
            n += 1


def next_order (points, paths, count, typeofproc, max_order, print_reduction = False):
    """
    Group 2 functions needed to calculate the diagrams for the next order.
    """
    next_points, next_paths, next_count = combine_diagrams_order(points, paths, count, typeofproc, max_order, offset = 0)
    if print_reduction:
        print("Number of diagrams in order", print(len(points)), ":", len(next_points))
    return group_diagrams(next_points, next_paths, next_count)