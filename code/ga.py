import numpy as np
import random
from itertools import permutations
from collections import Counter
import matplotlib.pyplot as plt
from tabulate import tabulate

# ------ parameters  --------------------------------------------------------------------------#
# globals
numbers = None
index = 0
N = None
given_constraints = None
population_size = 200
old_population_percent = 0.3
new_population_percent = 0.4
random_population_percent = 0.3
constraint_weight = 5
mutation_percent = 0.2
maximum_generation = None
black_list = set()
max_score_list_reg, av_score_list_reg = [], []
max_score_list_darwin, av_score_list_darwin = [], []
max_score_list_lamark, av_score_list_lamark = [], []
accuracy_reg, accuracy_darwin, accuracy_lamark = 0, 0, 0

# dynamic parameters
config_list, initial_matrix, greater_constraints, max_fitness = None, None, None, None
population, fitness_scores, generation = None, None, None
# ----------------------------------------------------------------------------------------------#

# returns the next index - grows each time +1
def next_index():
    global index
    index += 1
    return index

# returns a list of all configuration according to the config.txt file given
def read_config_list(filename):
    config = open(filename, "r")
    config_list = []
    for line in config:
      stripped_line = line.strip()
      item = stripped_line
      if ' ' in stripped_line:
          item = stripped_line.split(' ')
      config_list.append(item)
    config.close()
    return config_list

# creates a matrix according to the list of configuration made by config.txt
def build_matrix(config_list):
    global N
    N = int(config_list[index])
    matrix = np.zeros((N, N)) # empty
    given_digits = int(config_list[next_index()])
    for i in range(given_digits):
        curr = next_index()
        x, y, digit = int(config_list[curr][0]) - 1, int(config_list[curr][1]) - 1, int(config_list[curr][2])
        matrix[x,y] = digit # filled
        black_list.add((x,y))
    return matrix

# builds greater_constraints = a dict with all the constarins accccording to config_list
def build_greater_than_dict(config_list):
    global given_constraints
    given_constraints = int(config_list[next_index()])
    greater_constraints = {}
    for i in range(given_constraints): # go through constraints
        curr = next_index()
        x1, y1 = int(config_list[curr][0]) - 1, int(config_list[curr][1]) - 1
        x2, y2 = int(config_list[curr][2]) - 1, int(config_list[curr][3]) - 1
        if (x1, y1) not in greater_constraints.keys():
            greater_constraints[(x1, y1)] = [] # init key
        greater_constraints[(x1, y1)].append((x2, y2)) # add value
    return greater_constraints

# builds population dict = from a matrix copy inserts randon population
# generates a poulation into empty matrix cells and dict
def build_population(matrix):
    population = {}
    for index in range(population_size): #100
        matrix_copy = matrix.copy()
        for i in range(N):
            for j in range(N):
                if matrix_copy[i,j] == 0:
                  matrix_copy[i,j] = numbers[random.randrange(0, len(numbers))]  # insert to matrix
        population[index] = matrix_copy # add to dict
    return population

# receives a matrix and fills row values as permutation and returns population dict
def build_population_rows(matrix):
    population = {}
    list_row = numbers.copy() # val not in constrains
    while len(population) < population_size: # make matrixs
        matrix_copy = matrix.copy()
        # go through each row
        for row in range(N):
            list_row = numbers.copy() # val not in constrains
            no_cons_list = [] # index of cells with NO sonstrains (need to fill)
         
            # go through each each cell in row
            for column in range(N):
                if matrix_copy[row,column] == 0: # empty no constrain
                    no_cons_list.append(column)
                else: # not 0 has constarain - remove from list_row
                    list_row.remove(matrix_copy[row,column])      
            # look at full row
            l = list(permutations(list_row)) # using only vaues that arent in constrains in current row
            chosen_row = random.choice(l) # get random perm
            # fill row with values that arent in constrains in indexes that arend in constarins
            i = 0
            for column in no_cons_list: # index
                matrix_copy[row,column] = chosen_row[i] # insert to matrix
                i =+ 1
        population[str(matrix_copy)] = matrix_copy # add to dict
    return population

# ------- fitness functions ---------------------------------------------------------------------#
# calc matrix fitness score according to num of errors in row and column 
# the higher the score the better
def fitness(matrix):
    numbers_set = set(numbers)
    errors = 0
    for i in range(N):
        errors += N - len(set(matrix[i]))   # row
        errors += N - len(set(matrix[:,i])) # column 
    for cell in greater_constraints.keys():
        for related in greater_constraints[cell]:
            if matrix[cell] <= matrix[related]:
                errors += constraint_weight     
    return max_fitness - errors # scrore     

# receives population dict of all positions and calcalates corrent population's score
def build_fitness_scores(population):
# ------------------------------------------------------------------------------------------------#
    fitness_scores = {}
    for key in population.keys():
        fitness_scores[key] = fitness(population[key])
    return fitness_scores

# chooses best two population matrix with highest scores - matrix with higher scores will have more chance to be chosen
def choose_parents(population, fitness_scores):
    raffle_box = [] # list
    for index in population.keys():
        raffle_box.extend([index for i in range(fitness_scores[index])]) # matrix with higher scores will have more chance to be chosen
    index1, index2 = random.sample(raffle_box, 2) # choose 2
    return population[index1], population[index2]

# ------- cross_over ----------------------------------------------------------------------------#
# create new matrix according to 2 parent matrix
def cross_over(matrix1, matrix2):
    max_crossed = matrix1.copy()
    max_fit = fitness(max_crossed)
    # checks horizontal and vertical cross and according to highest fit chooses best crossover
    N_half = int(N/2)
    chosen = []
    for index in range(N):
        # 1) horizontal
        horizontal_cross = matrix1.copy()
        horizontal_cross[index:] = matrix2[index:] # row
        horizontal_fit = fitness(horizontal_cross)
        if horizontal_fit > max_fit:
            chosen = "horizontal"
            max_crossed = horizontal_cross
            max_fit = horizontal_fit
            
        # half left horizontal
        horizontal_half_left = matrix1.copy()
        horizontal_half_left[index:N_half] = matrix2[index:N_half] # left row
        horizontal_half_left_fit = fitness(horizontal_half_left)
        if horizontal_half_left_fit > max_fit:
            chosen = "half left horizontal"
            max_crossed = horizontal_half_left
            max_fit = horizontal_half_left_fit
            
         # half right horizontal
        horizontal_half_right = matrix1.copy()
        horizontal_half_right[N_half:index] = matrix2[N_half:index] # right row
        horizontal_half_right_fit = fitness(horizontal_half_right)
        if horizontal_half_right_fit > max_fit:
            chosen = "half right horizontal"
            max_crossed = horizontal_half_right
            max_fit = horizontal_half_right_fit
            
        # half vertical
        vertical_cross = matrix1.copy()
        vertical_cross[:, index:] = matrix2[:, index:] # column
        vertical_fit = fitness(vertical_cross)
        if vertical_fit > max_fit:
            chosen = "half vertical"
            max_crossed = vertical_cross
            max_fit = vertical_fit
            
        # half top vertical
        vertical_cross_top = matrix1.copy()
        vertical_cross_top[:, index:N_half] = matrix2[:, index:N_half] # top column
        vertical_top_fit = fitness(vertical_cross_top)
        if vertical_top_fit > max_fit:
            chosen = "half top vertical"
            max_crossed = vertical_cross_top
            max_fit = vertical_top_fit
             
        # half lower vertical
        vertical_cross_lower = matrix1.copy()
        vertical_cross_lower[:, N_half:index] = matrix2[:, N_half:index] # lower column
        vertical_lower_fit = fitness(vertical_cross_lower)
        if vertical_lower_fit > max_fit:
            chosen = "half lower vertical"
            max_crossed = vertical_cross_lower
            max_fit = vertical_lower_fit
            
        
    #print("chosen: ", chosen)
    return max_crossed # where the slices' union is the smallest

# bool checks if random is smaller then the probability given - used in create_new_population()
def raffle(probability):
    return random.random() < probability

# ------- mutation ------------------------------------------------------------------------------#
# create mutation in one cell of the matrix given - inputs a random value
def mutation(matrix):
    success = False
    while not success:
        i = random.randrange(0, N)
        j = random.randrange(0, N)
        if (i,j) not in black_list:
            success = True
    new_number = random.choice(numbers)
    matrix[i,j] = new_number
    return matrix

# creates generations that keep improving 
def create_new_population(population, fitness_scores):
    # cross over
    matrix1, matrix2 = choose_parents(population, fitness_scores) # parents
    matrix = cross_over(matrix1, matrix2) # child
    
    # mutate
    if raffle(mutation_percent):
        matrix = mutation(matrix)
    return matrix

# generates a new matrix randomly
def create_random_population():
    random_matrix = initial_matrix.copy()
    for i in range(N):
        for j in range(N):
            if random_matrix[i,j] == 0:
              random_matrix[i,j] = numbers[random.randrange(0, len(numbers))]  # insert to matrix
    return random_matrix

# returns a list chosen_population that hold 200*0.3 best scored matrixs
def choose_best(population, fitness_scores):
    # sort fitness_scores
    sorted_fitness_scores_values = sorted(fitness_scores.items(), key=lambda x: x[1], reverse=True)
    # cut 200*0.3 highest = best of population
    highest = int(population_size * old_population_percent)
    highest_mats = sorted_fitness_scores_values[:highest]
    chosen_population = []
    for key,value in highest_mats:
        chosen_population.append(population[key])   # add
    return chosen_population

# ------- create_new_generation ---------------------------------------------------------------------#
# genareate a new generation -  
def create_new_generation(chosen_population, lamarck = False):
    new_amount = int(population_size * new_population_percent)
    amount_random = int(population_size * random_population_percent)
    new_generation = {}
    new_scores = {}
    # takes best of current populations and keeps them alive for next gen
    for matrix in chosen_population:
        if lamarck:
            matrix = optimization(matrix)
        key = str(matrix)
        new_generation[key] = matrix
        new_scores[key] = fitness(matrix)
    # creates new matrix according to crossovers between parent matrixs
    counter = 0
    while counter < new_amount:
        matrix = create_new_population(population, fitness_scores)
        if lamarck:
            matrix = optimization(matrix)
        key = str(matrix)
        if key not in new_generation:
            counter += 1
            new_generation[key] = matrix
            new_scores[key] = fitness(matrix)
    # rand new population
    counter = 0
    while counter < amount_random:
        matrix = create_random_population()
        if lamarck:
            matrix = optimization(matrix)
        key = str(matrix)
        if key not in new_generation:
            counter += 1
            new_generation[key] = matrix
            new_scores[key] = fitness(matrix)
    return new_generation, new_scores

# ------- optimization ------------------------------------------------------------------------------#
# oprimizes a matrix by finding the cells where the constrains aren't settled
# and randomly choose one of the problimatic cells and to fix to the right value
def optimization(original_matrix):
    matrix = original_matrix.copy()
    for i in range(N):
        row = matrix[i] # looking for duplicates in row i
        row_set = set(row) # unique items in the row
        if len(row_set) < N: # which means we have missing digits and duplicates
            missing = list(set(numbers).difference(row_set)) # list of missing numbers in the row
            repeating = list(set([i for i in list(row) if list(row).count(i)>1])) # list of duplicated numbers in the row
            chosen_missing = missing[random.randrange(0, len(missing))] # randomly choose one of each
            chosen_repeating = repeating[random.randrange(0, len(repeating))]
            
            repeating_index = [] # getting all the places where the value is equal to the chosen duplicated number
            for j in range(N):
                if row[j] == chosen_repeating: # appending each index that satisfies this condition
                    repeating_index.append(j)

            chosen_j = repeating_index[random.randrange(0, len(repeating_index))] # choosing randomly one index 
            if (i, chosen_j) not in black_list: # making sure we're not changing a given digits (part of the constraints)
                matrix[i, chosen_j] = chosen_missing
            
        col = matrix[:, i] # looking for duplicates in column i, logically identical as with row
        col_set = set(col)
        if len(col_set) < N:
            missing = list(set(numbers).difference(col_set))
            repeating = list(set([i for i in list(col) if list(col).count(i)>1]))
            chosen_missing = missing[random.randrange(0, len(missing))]
            chosen_repeating = repeating[random.randrange(0, len(repeating))]
            
            repeating_index = []
            for j in range(N):
                if col[j] == chosen_repeating:
                    repeating_index.append(j)

            chosen_j = repeating_index[random.randrange(0, len(repeating_index))]
            if (chosen_j, i) not in black_list:
                matrix[chosen_j, i] = chosen_missing
        
    if fitness(original_matrix) < fitness(matrix): # we return this matrix only if its really optimized
        return matrix
    
    return original_matrix # otherwise stay with the original matrix

# used each time we create a changed and better solution matrix - initialized vars needed
def initialize(config_file = "config.txt"):
    global index, config_list, initial_matrix, greater_constraints, max_fitness, numbers, max_score_list_reg, \
        av_score_list_reg, max_score_list_darwin, av_score_list_darwin, max_score_list_lamark, av_score_list_lamark, \
        accuracy_reg, accuracy_darwin, accuracy_lamark, maximum_generation
    max_score_list_reg, av_score_list_reg = [], []
    max_score_list_darwin, av_score_list_darwin = [], []
    max_score_list_lamark, av_score_list_lamark = [], []
    index = 0
    config_list = read_config_list(config_file) # board config txt file
    initial_matrix = build_matrix(config_list) # matrix of all configuration - empty
    greater_constraints = build_greater_than_dict(config_list) # dict with all constarins
    max_fitness = N * N * 2 + len(greater_constraints) * constraint_weight
    numbers = list(np.arange(1,N + 1))
    accuracy_reg, accuracy_darwin, accuracy_lamark = 0, 0, 0
    maximum_generation = N ** 3

# used to restart varubales between the execution of diffrent genetic algo
# keep algo independent one from another
def restart():
    global population, fitness_scores, generation
    generation = 1
    population = build_population_rows(initial_matrix) # init poulation matrix and dict - filled #TODO
    max_fitness = N * N * 2 + len(greater_constraints) * constraint_weight
    fitness_scores = build_fitness_scores(population) # dict

# get average of a list
def average_list_val(lst):
    return sum(lst) / len(lst)

# ------- our genetic algorithms ----------------------------------------------------------------#
# the naive regular genetic algo - no optimization made
def regular_ga():
    restart()
    global population, fitness_scores, generation, accuracy_reg
    while generation != maximum_generation:
        best_old_population = choose_best(population, fitness_scores)
        population, fitness_scores = create_new_generation(best_old_population)
        # print(generation, ":", max(fitness_scores.values()))
        
        max_score_list_reg.append(max(fitness_scores.values())) # max
        av_score_list_reg.append(average_list_val(fitness_scores.values())) # mean
        generation = generation + 1
        
        for key in fitness_scores.keys():
            fit = fitness_scores[key]
            if fit == max_fitness:
                accuracy_reg = 1
                success_message(population[key])
                return
    
    accuracy_reg = round(max(fitness_scores.values()) / max_fitness, 2)
    approx_message()
                
# optimize all matrixs, fitness score calc after opt.
# next gen made BEFORE optimization                
def darwin_ga():
    restart()
    global population, fitness_scores, generation, accuracy_darwin
    while generation != maximum_generation:
        best_old_population = choose_best(population, fitness_scores)
        population, fitness_scores = create_new_generation(best_old_population)
        # print(generation, ":", max(fitness_scores.values()))
        
        generation = generation + 1
        
        for key, matrix in population.items():
            optimized = optimization(matrix)
            fit = fitness(optimized)
            if fit == max_fitness:
                accuracy_darwin = 1
                max_score_list_darwin.append(fit)
                av_score_list_darwin.append(average_list_val(fitness_scores.values()))
                success_message(optimized)
                return
        
        max_score_list_darwin.append(max(fitness_scores.values())) # max
        av_score_list_darwin.append(average_list_val(fitness_scores.values())) # mean
            
    accuracy_darwin = round(max(fitness_scores.values()) / max_fitness, 2)
    approx_message()

# optimize all matrixs, fitness score calc after opt. 
# next gen made AFTER optimization
def lamarck_ga():
    restart()
    global population, fitness_scores, generation, accuracy_lamark
    while generation != maximum_generation:
        best_old_population = choose_best(population, fitness_scores)
        population, fitness_scores = create_new_generation(best_old_population, True)
        # print(generation, ":", max(fitness_scores.values()))
        
        max_score_list_lamark.append(max(fitness_scores.values())) # max
        av_score_list_lamark.append(average_list_val(fitness_scores.values())) # mean
        
        generation = generation + 1
                    
        for key in fitness_scores.keys():
            fit = fitness_scores[key]
            if fit == max_fitness:
                accuracy_lamark = 1
                success_message(population[key])
                return
    
    accuracy_lamark = round(max(fitness_scores.values()) / max_fitness, 2)
    approx_message()
# ----------------------------------------------------------------------------------------------#

def success_message(matrix):
    print("An optimal solution found after", generation, "generations")
    print_solution(matrix)

def approx_message():
    matrix_id, fitness = max(fitness_scores.items(), key=lambda x: x[1])
    matrix = population[matrix_id]
    print("An approximate solution found after", generation, "generations, with accuracy of", str(round(fitness/max_fitness, 2)))
    print_solution(matrix)
            
# printing a represantation matrix with signs according to the given constraints
def print_solution(matrix):
    rep_mat = np.full((2*N-1, 2*N-1), " ")
    for x1 ,y1 in greater_constraints.keys():
        for x2, y2 in greater_constraints[(x1 ,y1)]:
            if x1 == x2:
                if y1 < y2:
                    rep_mat[x1*2, y1*2 + 1] = ">"
                else:
                    rep_mat[x1*2, y1*2 - 1] = "<"
            elif y1 == y2:
                if x1 < x2:
                    rep_mat[x1*2+1, y1*2] = "V"
                else:
                    rep_mat[x1*2-1, y1*2] = "A"
                
    for i in range(N):
        for j in range(N):
            rep_mat[i*2,j*2] = matrix[i,j]
            
    print(tabulate(rep_mat, tablefmt='fancy_grid'))

# -------- plot --------------------------------------------------------------------------------#
# creates a comparision plot for a config txt file with all three genetic algorithms
def plot():
    board_size = str(N) + "*" + str(N)
    plt.plot(max_score_list_reg, label = "Regular MAX score", color = 'b')
    plt.plot(av_score_list_reg, label = "Regular AVERAGE score", color = 'c', linestyle ='--')
    plt.plot(max_score_list_darwin, label = "Darwin MAX score", color ='g')
    plt.plot(av_score_list_darwin, label = "Darwin AVERAGE score", color ='y', linestyle ='--')
    plt.plot(max_score_list_lamark, label = "Lamark MAX score", color ='r')
    plt.plot(av_score_list_lamark, label = "Lamark AVERAGE score", color ='m', linestyle ='--')
    plt.xlabel('generation')
    plt.ylabel('Fitness score')
    level = ["EASY", "TRICKY"]
    plt.title("Comparision of algorithms - Board: " + str(board_size) + " Level: " 
           + str(level[1]) + '\n' + "Accuracy: Regular: " + str(accuracy_reg) +
           " Darwin: " + str(accuracy_darwin) + " Lamark: " + str(accuracy_lamark))
    plt.legend()    
    plt.show()
