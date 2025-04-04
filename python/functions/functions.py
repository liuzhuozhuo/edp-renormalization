import matplotlib.pyplot as plt
import numpy as np
from scipy.special import binom, factorial

#Canonical diagram functions
points_1st_1 = np.array([[0, 1], [0, 3], [1, 2], [2, 2]])
paths_1st_1 = np.array([[1, 3], [2, 3], [3, 4]]) 

points_1st_2 = np.array([[0, 2], [1, 2], [2, 1], [2, 3]])
paths_1st_2 = np.array([[1, 2], [2, 3], [2, 4]]) 

can_points_1 = np.empty((2, max(len(points_1st_1), len(points_1st_2)), 2))
can_points_1[0] = points_1st_1
can_points_1[1] = points_1st_2

can_paths_1 = np.empty((2, max(len(paths_1st_1), len(paths_1st_2)), 2), dtype=int)
can_paths_1[0] = paths_1st_1
can_paths_1[1] = paths_1st_2


def trim_zeros_2D(array):
    # Identify rows and columns that contain only zeros
    non_zero_rows = ~np.all(array == 0, axis=1)
    non_zero_cols = ~np.all(array == 0, axis=0)

    # Filter the array to include only rows and columns with non-zero values
    trimmed_array = array[non_zero_rows][:, non_zero_cols]
    return trimmed_array
    
def trim_zeros_3D(array):
    # Create a mask to identify non-zero elements along each dimension
    mask = ~(array == 0).all(axis=(1, 2))
    trimmed_array = array[mask]
    
    mask = ~(trimmed_array == 0).all(axis=(0, 2))
    trimmed_array = trimmed_array[:, mask]
    
    mask = ~(trimmed_array == 0).all(axis=(0, 1))
    trimmed_array = trimmed_array[:, :, mask]
    
    return trimmed_array

def find_equal_subarrays(array):
    sorted_subarrays = [np.sort(subarray) for subarray in array]
    unique_subarrays, indices, counts = np.unique(sorted_subarrays, axis=0, return_index=True, return_counts=True)
    duplicate_positions = [np.where((sorted_subarrays == unique_subarrays[i]).all(axis=1))[0] for i in range(len(unique_subarrays)) if counts[i] > 1]
    return duplicate_positions

def represent_diagram (points, paths, index = False, directory = "", colors = "tab:blue", number = 0):
    fig=plt.figure(figsize=(5,3)) 
    ax=fig.add_subplot(111)
    ax.axis('off')
    paths = trim_zeros_2D(paths)
    points = trim_zeros_2D(points)

    loops = find_equal_subarrays(paths)
    
    for i in range(len(paths)):
        if np.isin(i, loops):
            middle_point = (points[paths[i, 0]-1] + points[paths[i, 1]-1]) / 2
            circle = plt.Circle((middle_point[0], middle_point[1]), np.linalg.norm(points[paths[i, 0]-1]-middle_point), color=colors, fill=False)
            ax.add_patch(circle)
        else:
            ax.plot([points[paths[i, 0]-1, 0], points[paths[i, 1]-1, 0]], [points[paths[i, 0]-1, 1], points[paths[i, 1]-1, 1]], color=colors)
    if index:
        for i in range(np.max(paths)):
            ax.text(points[i, 0], points[i, 1], str(i+1), fontsize=12, color="black", ha="right", va="top")
    if number !=0:
        ax.text(0.5, 0.5, f"N = {number}", fontsize=12, color="black", ha="center", va="center")
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
    return in_out_paths

def connection(points1, paths1, points2, paths2, offset = 0):

    in_out_paths1 = in_out_paths(paths1)
    in_out_paths2 = in_out_paths(paths2)

    #Create the new points array
    points = np.zeros((len(points1) + len(points2), 2))
    points[:len(points1)] = points1
    points[len(points1):] = points2 + np.array([np.max(points1)+1, offset])

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

    combinations = how_connected(max_connections, n_connections, n_1, n_2)
    combinations = trim_zeros_3D(combinations)
    n  = len(combinations)
    n_paths = len(combinations[0])
    paths = np.zeros((n, len(paths1) + len(paths2) + n_paths, 2), dtype=int)
    paths[:n, :len(paths1)] = paths1

    for i in range(n):
        for j in range(n_paths):
            if (combinations[i, j, 0] != 0 and combinations[i, j, 1] != 0):
                paths[i, len(paths1)+j] = np.array([in_out_paths1[0, combinations[i, j, 0]-1], in_out_paths2[1, combinations[i, j, 1]-1]])
        if (np.count_nonzero(paths2) != 0):
            paths[i, len(paths1)+n_paths:] = paths2 + np.array([len(points1), len(points1)])

    return points, paths

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
    for i in range(1, int(np.max(points, axis=0)[1])+1):
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
    # i = 0
    # while i < len(points):
    #     if points[i, 0] == 0 and points[i, 1] == 0:
    #         points = np.delete(points, i, axis=0)
    #         for j in range (i, np.max(paths)+1):
    #             paths = decrement_number_in_array(paths, j)  
    #     i += 1
    
    end = len(points)

    i=1
    while i <= end:
        count = 0
        for j in range(len(paths)):
            for k in range(2):
                if paths[j, k] == i:
                    count += 1
                    if count > 1:
                        break
        if count == 0:
            end -= 1
            points = np.delete(points, i-1, axis=0)
            for j in range (i, np.max(paths)+1):
                paths = decrement_number_in_array(paths, j)  
            i-=1
        i += 1
    
    return points, paths

def simplify_diagram (points, paths):
    new_points, new_paths = simplify_diagram_it(points, paths)
    new_new_points, new_new_paths = simplify_diagram_it(new_points, new_paths)
    while len(new_points) != len(new_new_points) or len(new_paths) != len(new_new_paths):
        new_points, new_paths = new_new_points, new_new_paths
        new_new_points, new_new_paths = simplify_diagram_it(new_points, new_paths)
    
    return new_new_points, new_new_paths

def combine_diagrams_order (points, paths, offset = 0):
    curr_max_order = len(points)
    
    max_points = np.zeros((2), dtype=int)
    for i in range(len(paths[0])):
        n1_1 = len(np.trim_zeros(in_out_paths(paths[0][i])[0]))
        if (n1_1 > max_points[0]):
            max_points[0] = n1_1
        n2_1 = len(np.trim_zeros(in_out_paths(paths[0][i])[1]))
        if (n2_1 > max_points[1]):
            max_points[1] = n2_1

    for i in range(len(paths[-1])):
        n1_2 = len(np.trim_zeros(in_out_paths(paths[-1][i])[0]))
        if (n1_2 > max_points[0]):
            max_points[0] = n1_2
        n2_2 = len(np.trim_zeros(in_out_paths(paths[-1][i])[1]))
        if (n2_2 > max_points[1]):
            max_points[1] = n2_2

    max_connections = np.min(max_points)

    n_connections = 0
    for i in range (max_connections):
        n_connections += int(binom(max_points[0], i+1)*binom(max_points[1], i+1) * factorial(i+1))
    
    n = 0
    if len(points) == 1:
        new_points = np.zeros((2*n_connections-1, len(points[0][0]) + len(points[-1][0]), 2))
        new_paths = np.zeros((2*n_connections-1, len(paths[0][0]) + len(paths[-1][0])+max_connections, 2), dtype=int)
        for i in range(len(paths[0])): #Iterate over the len of the paths of order 1
            for j in range(len(paths[-1])): #Iterate over the len of the paths of the highest order
                dummy_points, dummy_paths = connection(points[0][i], trim_zeros_2D(paths[0][i]), points[-1][j], trim_zeros_2D(paths[-1][j]), offset=offset)
                for k in range(len(dummy_paths)):
                    simp_points, simp_paths = simplify_diagram(dummy_points, trim_zeros_2D(dummy_paths[k]))
                    for l in range(len(simp_points)):
                        new_points[n, l] = simp_points[l]
                    for l in range(len(simp_paths)):
                        new_paths[n, l] = simp_paths[l]
                    n += 1
    else:
        new_points = np.zeros((2*len(paths[0])*len(paths[-1])*n_connections, len(points[0][0]) + len(points[-1][0]), 2))
        new_paths = np.zeros((2*len(paths[0])*len(paths[-1])*n_connections, len(paths[0][0]) + len(paths[-1][0])+max_connections, 2), dtype=int)
        for i in range(len(paths[0])): #Iterate over the len of the paths of order 1
            for j in range(len(paths[-1])): #Iterate over the len of the paths of the highest order
                dummy_points, dummy_paths = connection(points[0][i], trim_zeros_2D(paths[0][i]), points[-1][j], trim_zeros_2D(paths[-1][j]), offset=offset)
                for k in range(len(dummy_paths)):
                    simp_points, simp_paths = simplify_diagram(dummy_points, dummy_paths[k])
                    for l in range(len(simp_points)):
                        new_points[n, l] = simp_points[l]
                    for l in range(len(simp_paths)):
                        new_paths[n, l] = simp_paths[l]
                    n += 1
                dummy_points, dummy_paths = connection(points[-1][j], trim_zeros_2D(paths[-1][j]),points[0][i], trim_zeros_2D(paths[0][i]), offset=offset)
                for k in range(len(dummy_paths)):
                    simp_points, simp_paths = simplify_diagram(dummy_points, dummy_paths[k])
                    for l in range(len(simp_points)):
                        new_points[n, l] = simp_points[l]
                    for l in range(len(simp_paths)):
                        new_paths[n, l] = simp_paths[l]
                    n += 1
    return new_points, trim_zeros_3D(new_paths)

def all_components_in_other(array1, array2):
    for row1 in array1:
        found = False
        for row2 in array2:
            if np.array_equal(np.sort(row1), np.sort(row2)):
                found = True
                break
        if not found:
            return False
    return True

def group_diagrams (points, paths):
    group_paths = np.zeros((1, len(paths[0]), 2), dtype=int)
    group_points = np.zeros((1, len(points[0]), 2))
    group_points[0] = points[0]
    group_paths[0] = paths[0]
    count = np.zeros((1), dtype=int)
    count[0] = 1
    for i in range(1, len(paths)):
        cont = False
        for j in range(len(group_paths)):
            if all_components_in_other(paths[i], group_paths[j]):
                count[j] += 1
                cont = False
                break
            else:
                cont = True

        if cont:
            group_paths = np.append(group_paths, [paths[i]], axis=0)
            group_points = np.append(group_points, [points[i]], axis=0)
            count = np.append(count, [1])
    return group_points, group_paths, count