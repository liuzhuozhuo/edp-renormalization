import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.special import binom, factorial

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

def represent_diagram (points, all_paths, index = False, directory = "", colors = ["tab:blue", "tab:red", "black"], line = ["solid", "solid", "dashed"]):
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

    fig=plt.figure(figsize=(5,3)) 
    ax=fig.add_subplot(111)
    ax.axis('off')
    j = 0
    for paths in all_paths:
        loops = find_equal_subarrays(paths)
        for i in range(len(paths)):
            if (line[j] == "dashed"):
                with mpl.rc_context({'path.sketch': (3, 15, 1)}):
                    if np.isin(i, loops):
                        middle_point = (points[paths[i, 0]-1] + points[paths[i, 1]-1]) / 2
                        circle = plt.Circle((middle_point[0], middle_point[1]), np.linalg.norm(points[paths[i, 0]-1]-middle_point), color=colors[j], fill=False)
                        ax.add_patch(circle)
                    else:
                        ax.plot([points[paths[i, 0]-1, 0], points[paths[i, 1]-1, 0]], [points[paths[i, 0]-1, 1], points[paths[i, 1]-1, 1]], color=colors[j])
            else:
                if np.isin(i, loops):
                    middle_point = (points[paths[i, 0]-1] + points[paths[i, 1]-1]) / 2
                    circle = plt.Circle((middle_point[0], middle_point[1]), np.linalg.norm(points[paths[i, 0]-1]-middle_point), color=colors[j], fill=False, linestyle=line[j])
                    ax.add_patch(circle)
                else:
                    ax.plot([points[paths[i, 0]-1, 0], points[paths[i, 1]-1, 0]], [points[paths[i, 0]-1, 1], points[paths[i, 1]-1, 1]], color=colors[j], linestyle=line[j])
        j+=1
    if index:
        for i in range(len(points)):
            ax.text(points[i, 0], points[i, 1], str(i+1), fontsize=12, color="black", ha="right", va="top")
    if directory != "":
        plt.savefig(directory, bbox_inches='tight')
        plt.close()
#From Chatgpt
def unique_values(array):
    unique, counts = np.unique(array, return_counts=True)
    unique_values = unique[counts == 1]
    return unique_values

#From Eddmik in https://stackoverflow.com/questions/34593824/trim-strip-zeros-of-a-numpy-array
#Note that for numpy 2.2 the function trim_zeros is available for ndarrays.
#I may rewirte this function as i would have done it myself, but for now we will use this one.
def trim_zeros_2D(array, axis=1):
    mask = ~(array==0).all(axis=axis)
    inv_mask = mask[::-1]
    start_idx = np.argmax(mask == True)
    end_idx = len(inv_mask) - np.argmax(inv_mask == True)
    if axis:
        return array[start_idx:end_idx,:]
    else:
        return array[:, start_idx:end_idx]
    
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
    paths = np.zeros((n_connec, n_types, len(paths1) + len(paths2) + max(max_connections), 2), dtype=int)
    paths[:n_connec,:n_types,:len(paths1)] = paths1
    for i in range(n_connec):
        for j in range(n_types):
            for k in range(max_connections[j]):
                if (combinations[i,j, k, 0] != 0 and combinations[i,j, k, 1] != 0):
                    paths[i,j, len(paths1)+k] = np.array([in_out_paths1[j,0, combinations[i, j, k, 0]-1], in_out_paths2[j,1, combinations[i, j, k, 1]-1]])
            if (np.count_nonzero(paths2[j]) != 0):
                for k in range(len(paths2[j])):
                    if (paths2[j, k, 0] != 0 and paths2[j, k, 1] != 0):
                        paths[i,j, len(paths1)+max(max_connections)+k] = paths2[j, k] + np.array([len(points1), len(points1)])

    return points, paths

def decrement_number_in_array(array, number):
    array[array == number] -= 1
    return array

def decrement_number_in_array_2D(array, number):
    """
    Decrement a number in a 2D array.
    Args:
        array (np.array): 2D array to decrement the number
        number (int): number to decrement
    Returns:
        array (np.array): decremented array
    """
    for i in range(len(array)):
        for j in range(len(array[i])):
            for k in range(2):
                if array[i, j, k] == number:
                    array[i, j, k] -= 1
    return array

def simplify_diagram_it (points, paths):
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
    new_points, new_paths = simplify_diagram_it(points, paths)
    new_new_points, new_new_paths = simplify_diagram_it(new_points, new_paths)
    while len(new_points) != len(new_new_points) or len(new_paths) != len(new_new_paths):
        new_points, new_paths = new_new_points, new_new_paths
        new_new_points, new_new_paths = simplify_diagram_it(new_points, new_paths)
    return new_new_points, new_new_paths

