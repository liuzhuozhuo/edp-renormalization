#Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import binom, factorial

def find_equal_subarrays(array):
    sorted_subarrays = [np.sort(subarray) for subarray in array]
    unique_subarrays, indices, counts = np.unique(sorted_subarrays, axis=0, return_index=True, return_counts=True)
    duplicate_positions = [np.where((sorted_subarrays == unique_subarrays[i]).all(axis=1))[0] for i in range(len(unique_subarrays)) if counts[i] > 1]
    return duplicate_positions

#Function to represent a diagram
def represent_diagram (points, paths, index = False, directory = "", colors = "tab:blue"):
    fig=plt.figure(figsize=(5,3)) 
    ax=fig.add_subplot(111)
    ax.axis('off')
    loops = find_equal_subarrays(paths)
    for i in range(len(paths)):
        if np.isin(i, loops):
            middle_point = (points[paths[i, 0]-1] + points[paths[i, 1]-1]) / 2
            circle = plt.Circle((middle_point[0], middle_point[1]), np.linalg.norm(points[paths[i, 0]-1]-middle_point), color=colors, fill=False)
            ax.add_patch(circle)
        else:
            ax.plot([points[paths[i, 0]-1, 0], points[paths[i, 1]-1, 0]], [points[paths[i, 0]-1, 1], points[paths[i, 1]-1, 1]], color=colors)
    if index:
        for i in range(len(points)):
            ax.text(points[i, 0], points[i, 1], str(i+1), fontsize=12, color="black", ha="right", va="top")
    if directory != "":
        plt.savefig(directory, bbox_inches='tight')
        plt.close()

def in_out_paths (paths):
    in_out_paths = np.zeros((2, len(paths)), dtype=int)
    inp = 0
    out = 0
    #iterate over all points (from 1 to the maximum point number)
    for i in range(1, np.max(paths)+1):
        count = 0
        input = False
        #iterate over all paths to find the points that appear only once.
        #if the point appears in the first column of a path, it is an output.
        #if the point appears in the second column of a path, it is an input.
        for j in range(len(paths)):
            for k in range(2):
                if paths[j, k] == i:
                    count += 1
                    if k == 1:
                        input = True
                    if count > 1:
                        break
        if count == 1:
            if input:
                in_out_paths[0,inp] = i
                inp += 1
            else:
                in_out_paths[1,out] = i
                out += 1
    in_out_paths = trim_zeros_2D(in_out_paths, axis=0)
    in_out_paths = trim_zeros_2D(in_out_paths, axis=1)
    return in_out_paths

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
    
#From Chatgpt
def trim_zeros_3D(array):
    # Create a mask to identify non-zero elements along each dimension
    mask = ~(array == 0).all(axis=(1, 2))
    trimmed_array = array[mask]
    
    mask = ~(trimmed_array == 0).all(axis=(0, 2))
    trimmed_array = trimmed_array[:, mask]
    
    mask = ~(trimmed_array == 0).all(axis=(0, 1))
    trimmed_array = trimmed_array[:, :, mask]
    
    return trimmed_array

def connection(points1, paths1, points2, paths2):
    in_out_paths1 = in_out_paths(paths1)
    in_out_paths2 = in_out_paths(paths2)

    #Create the new points array
    points = np.zeros((len(points1) + len(points2), 2))
    points[:len(points1)] = points1
    points[len(points1):] = points2 + np.array([np.max(points1)+1,0])

    for i in range(len(in_out_paths2)):
        for j in range(len(in_out_paths2[0])):
            if in_out_paths2[i, j] != 0:
                in_out_paths2[i, j] += len(points1)

    #Calculate the number of connections
    n_1 = len(np.trim_zeros(in_out_paths1[0]))
    n_2 = len(np.trim_zeros(in_out_paths2[1]))
    max_connections = min(n_1, n_2)

    n_connections = 0
    for i in range (max_connections):
        n_connections += int(binom(n_1, i+1)*binom(n_2, i+1) * factorial(i+1))

    combinations = np.zeros((n_connections, max_connections, 2), dtype=int)

    combinations = how_connected(combinations, max_connections, n_connections, n_1, n_2)
    combinations = trim_zeros_3D(combinations)
    n  = len(combinations)
    n_paths = len(combinations[0])
    paths = np.zeros((n, len(paths1) + len(paths2) + n_paths, 2), dtype=int)
    paths[:n, :len(paths1)] = paths1

    for i in range(n):
        for j in range(n_paths):
            if (combinations[i, j, 0] != 0 and combinations[i, j, 1] != 0):
                paths[i, len(paths1)+j] = np.array([in_out_paths1[0, combinations[i, j, 0]-1], in_out_paths2[1, combinations[i, j, 1]-1]])
        paths[i, len(paths1)+n_paths:] = paths2 + np.array([len(points1), len(points1)])

    return points, paths

def how_connected(combinations, max_connections, n_connections, n_1, n_2):
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

def decrement_number_in_array(array, number):
    array[array == number] -= 1
    return array

def simplify_diagram_it(points, paths):
    pos = np.zeros((2, 2), dtype=int)
    for i in range(1, np.max(paths)+1):
        count = 0
        for j in range(len(paths)):
            for k in range(2):
                if paths[j, k] == i:
                    count += 1
                    if count == 1:
                        pos[0] = np.array([j, k])
                    elif count == 2:
                        pos[1] = np.array([j, k])
                    else:
                        break
        if count == 2:
            points = np.delete(points, i-1, axis=0)
            if pos[0, 1] == 0:
                if pos[1, 1] == 0:
                    prov = np.array([paths[pos[0, 0], 1], paths[pos[1, 0], 1]])
                elif pos[1, 1] == 1:
                    prov = np.array([paths[pos[0, 0], 1], paths[pos[1, 0], 0]])                   
            elif pos[0, 1] == 1:
                if pos[1, 1] == 0:
                    prov = np.array([paths[pos[0, 0], 0], paths[pos[1, 0], 1]])                  
                elif pos[1, 1] == 1:
                    prov = np.array([paths[pos[0, 0], 0], paths[pos[1, 0], 0]])
            paths = np.delete(paths, (pos[0, 0], pos[1, 0]), axis=0)
            paths = np.append(paths, [prov], axis=0)    
            for j in range (i, np.max(paths)+1):
                paths = decrement_number_in_array(paths, j)  
    for i in range(int(np.max(points, axis=0)[0])+1):
        count = 0
        for j in range(len(points)):
            if points[j, 0] == i:
                count +=1
                break
        if count == 0:
            for j in range(len(points)):
                if points[j, 0] > i:
                    points[j, 0] -= 1
        count = 0
    for i in range(int(np.max(points, axis=0)[1])+1):
        count = 0
        for j in range(len(points)):
            if points[j, 1] == i:
                count +=1
                break
        if count == 0:
            for j in range(len(points)):
                if points[j, 1] > i:
                    points[j, 1] -= 1
     
    for i in range(0, int(np.min(points, axis=0)[1])-1, -1):
        count = 0
        for j in range(len(points)):
            if points[j, 1] == i:
                count +=1
                break
        if count == 0:
            for j in range(len(points)):
                if points[j, 1] < i:
                    points[j, 1] += 1
    return points, paths

def simplify_diagram (points, paths):
    new_points, new_paths = simplify_diagram_it(points, paths)
    new_new_points, new_new_paths = simplify_diagram_it(new_points, new_paths)
    while len(new_points) != len(new_new_points) or len(new_paths) != len(new_new_paths):
        new_points, new_paths = new_new_points, new_new_paths
        new_new_points, new_new_paths = simplify_diagram_it(new_points, new_paths)
    return new_new_points, new_new_paths

