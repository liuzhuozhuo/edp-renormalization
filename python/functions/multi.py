import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.special import binom, factorial

def combine_paths(*paths):
    max_len = max([len(path) for path in paths])
    final_path = np.zeros((len(paths), max_len, 2), dtype=int)
    for i in range(len(paths)):
        if len(paths[i]) < max_len:
            final_path[i] = np.append(paths[i], np.zeros((max_len - len(paths[i]), 2), dtype=int), axis=0)
        else:
            final_path[i] = paths[i]
    return final_path

def find_equal_subarrays(array):
    sorted_subarrays = [np.sort(subarray) for subarray in array]
    unique_subarrays, indices, counts = np.unique(sorted_subarrays, axis=0, return_index=True, return_counts=True)
    duplicate_positions = [np.where((sorted_subarrays == unique_subarrays[i]).all(axis=1))[0] for i in range(len(unique_subarrays)) if counts[i] > 1]
    return duplicate_positions

def represent_diagram (points, all_paths, index = False, directory = "", colors = ["tab:blue", "tab:red", "black"], line = ["solid", "solid", "dashed"]):
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

def unique_values(array):
    unique, counts = np.unique(array, return_counts=True)
    unique_values = unique[counts == 1]
    return unique_values

def trim_zeros_2D(array, axis=1):
    mask = ~(array==0).all(axis=axis)
    inv_mask = mask[::-1]
    start_idx = np.argmax(mask == True)
    end_idx = len(inv_mask) - np.argmax(inv_mask == True)
    if axis:
        return array[start_idx:end_idx,:]
    else:
        return array[:, start_idx:end_idx]
    
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

    combinations = np.zeros((n_connec, n_types, max(max_connections), 2), dtype=int)
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

    paths = np.zeros((n_connec, n_types, len(paths1) + len(paths2) + max(max_connections), 2), dtype=int)
    paths[:n_connec,:n_types,:len(paths1)] = paths1
    for i in range(n_connec):
        for j in range(n_types):
            for k in range(max_connections[j]):
                if (combinations[i,j, k, 0] != 0 and combinations[i,j, k, 1] != 0):
                    paths[i,j, len(paths1)+k] = np.array([in_out_paths1[j,0, combinations[i, j, k, 0]-1], in_out_paths2[j,1, combinations[i, j, k, 1]-1]])
            if (np.count_nonzero(paths2[j]) != 0):
                paths[i,j, len(paths1)+max(max_connections):] = paths2[j] + np.array([len(points1), len(points1)])

    return points, paths
