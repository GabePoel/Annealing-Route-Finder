from student_utils import *
import heapq
import time
import random
import numpy as np
import networkx as nx

class Solution:
    # allows for easier analysis of solutions
    def __init__(self, graph, path, dropoffs):
        self.setup_parameters()
        self.graph = graph
        self.path = path
        self.dropoffs = dropoffs
        self.hash_string = self.generate_hash()
        self.hash = self.__hash__()
        self.cost = self.find_cost()
        self.explored = False
        self.tags = {}

    def __lt__(self, other):
        return self.cost < other.cost

    def __le__(self, other):
        return self.cost <= other.cost

    def __eq__(self, other):
        return self.hash == other

    def __ne__(self, other):
        return self.hash != other

    def __gt__(self, other):
        return self.cost > other.cost

    def __ge__(self, other):
        return self.cost >= other.cost

    def __hash__(self):
        return self.hash_string.__hash__()

    def __str__(self):
        return_string = '('
        if self.print_path:
            return_string += ' path: ' + str(self.path) + ' '
        if self.print_cost:
            return_string += ' cost: ' + str(round(self.cost, 2)) + ' '
        if self.print_dropoffs:
            return_string += ' dropoffs: ' + str(self.dropoffs) + ' '
        return_string += ')'
        return return_string

    def setup_parameters(self):
        self.print_path = True
        self.print_cost = True
        self.print_dropoffs = True

    def find_cost(self):
        cost_given = cost_of_solution(self.graph, self.path, self.dropoffs)
        if cost_given[0] == 'infinite':
            return np.inf
        else:
            return cost_given[0]

    def tag(self, key, entry):
        self.tags.update({key:entry})

    def get_tag(self, key):
        # explored tag
        # home tag
        return self.tags[key]

    def generate_hash(self):
        dropoff_locations = list(self.dropoffs.keys())
        dropoff_locations.sort()
        hash_string = ''
        for location in self.path:
            hash_string += int_to_chr(location)
        hash_string += chr(600)
        for location in dropoff_locations:
            hash_string += int_to_chr(location)
            homes_dropped_off = self.dropoffs[location]
            homes_dropped_off.sort()
            for home in homes_dropped_off:
                hash_string += int_to_chr(home)
            hash_string += chr(600)
        return hash_string

class Solver:
    def __init__(self, list_of_locations, list_of_homes, starting_car_location, adjacency_matrix, params=['10']):
        self.known_solutions = {}
        self.best_solutions = []
        self.location_names = list_of_locations
        self.home_names = list_of_homes
        self.num_locations = len(self.location_names)
        self.num_homes = len(self.home_names)
        self.start_name = starting_car_location
        self.adjacency_matrix = adjacency_matrix
        self.start = self.location_names.index(starting_car_location)
        self.graph, adj_message = adjacency_matrix_to_graph(adjacency_matrix)
        self.locations = []
        self.homes = []
        self.num_explore_calls = 0
        self.num_known_solutions = 0
        for name in self.location_names:
            self.locations.append(self.location_names.index(name))
        for name in self.home_names:
            self.homes.append(self.location_names.index(name))
        self.apply_parameters(params)

    def apply_parameters(self, params):
        solution_depth = 1500 / self.num_locations
        self.initial_depth = solution_depth
        self.density = nx.density(self.graph)
        # print('density ' + str(self.density))
        self.final_depth = 2
        self.energy_level = 0
        self.initial_temperature = 36 * self.num_locations
        self.temperature = self.initial_temperature
        self.cooling_rate = 0
        self.update_explore_depth()

    def update_explore_depth(self):
        self.energy_level = self.num_known_solutions / self.density
        self.explore_depth = int(max([self.initial_depth * np.exp(- self.energy_level / self.temperature), self.final_depth]))
        # print('---')
        # print('current energy level: ' + str(self.energy_level))
        # print('current temperature: ' + str(self.temperature))
        # print('current explore depth: ' + str(self.explore_depth))
        self.temperature = max(1, self.temperature - self.temperature * self.cooling_rate)

    def solve(self):
        start_time = time.clock()
        self.generate_min_solution()
        self.generate_mid_solution()
        self.generate_max_solution()
        good_solutions = self.get_best_solutions(self.explore_depth)
        while not self.all_explored(good_solutions):
            self.explore_all(good_solutions)
            self.update_explore_depth()
            good_solutions = self.get_best_solutions(self.explore_depth)
            # total_cost = sum(s.cost for s in good_solutions)
            # mean_cost = total_cost / self.solution_depth
            # print(mean_cost)
        best_solution = self.get_best_solution()
        end_time = time.clock()
        total_time = round(end_time - start_time, 2)
        print('~~~ Best Solution ~~~')
        print(best_solution)
        print('~~~ Statistics ~~~')
        print('Number of Explored Solutions: ' + str(len(self.best_solutions)))
        print('Number of Explore Calls: ' + str(self.num_explore_calls))
        print('Processing Time: ' + str(total_time) + ' seconds')
        return best_solution.path, best_solution.dropoffs

    def generate_min_solution(self):
        # generates all solutions with students dropped off at one location
        for location in self.locations:
            if location == self.start:
                path = [location, location]
            else:
                path = nx.shortest_path(self.graph, source=self.start, target=location)
                path += nx.shortest_path(self.graph, source=location, target=self.start)[1:]
            all_locations = self.locations.copy()
            dropoffs = {location:all_locations}
            solution = Solution(self.graph, path, dropoffs)
            self.add_solution(solution)

    def generate_max_solution(self):
        # greedily generates solution where students are dropped off at all their houses
        all_distances = dict(nx.all_pairs_dijkstra_path_length(self.graph))
        greedy_path = [self.start]
        current_location = self.start
        homes_remaining = self.homes.copy()
        homes_remaining.sort(key=lambda h: all_distances[current_location][h])
        greedy_dropoffs = {}
        while len(homes_remaining) > 0:
            closest_home = homes_remaining.pop(0)
            if closest_home != self.start:
                path_extension = nx.shortest_path(self.graph, current_location, closest_home)[1:]
                greedy_path += path_extension
            current_location = closest_home
            greedy_dropoffs.update({closest_home:[closest_home]})
        if current_location != self.start:
            path_extension = nx.shortest_path(self.graph, current_location, self.start)[1:]
            greedy_path += path_extension
        candidate_solution = Solution(self.graph, greedy_path, greedy_dropoffs)
        self.add_solution(candidate_solution)

    def generate_mid_solution(self):
        # generates solutions in steps of five
        for i in range(1,self.num_homes):
            if i % 5 == 0:
                self.generate_sized_solution(i)

    def generate_sized_solution(self, size):
        # greedily generates solution of given size where students are dropped off at all their houses
        all_distances = dict(nx.all_pairs_dijkstra_path_length(self.graph))
        greedy_path = [self.start]
        current_location = self.start
        homes_remaining = random.sample(self.homes, size)
        homes_remaining.sort(key=lambda h: all_distances[current_location][h])
        while len(homes_remaining) > 0:
            closest_home = homes_remaining.pop(0)
            if closest_home != self.start:
                path_extension = nx.shortest_path(self.graph, current_location, closest_home)[1:]
                greedy_path += path_extension
            current_location = closest_home
        if current_location != self.start:
            path_extension = nx.shortest_path(self.graph, current_location, self.start)[1:]
            greedy_path += path_extension
        dropoffs = self.find_best_dropoffs(greedy_path)
        candidate_solution = Solution(self.graph, greedy_path, dropoffs)
        self.add_solution(candidate_solution)

    def add_solution(self, solution):
        if not solution.hash in self.known_solutions:
            # print(solution)
            self.num_known_solutions += 1
            self.known_solutions.update({solution.hash:solution})
            heapq.heappush(self.best_solutions, solution)

    def get_best_solutions(self, n_best):
        return heapq.nsmallest(n_best, self.best_solutions)

    def get_best_solution(self):
        return heapq.heappop(self.best_solutions)

    def all_explored(self, solutions):
        all_solutions_explored = True
        for solution in solutions:
            if not solution.explored:
                all_solutions_explored = False
        return all_solutions_explored
    
    def find_best_dropoffs(self, path):
        dropoffs = {}
        max_num_locations = min(len(set(path)), 15)
        homes_to_optimize = self.homes.copy()
        for home in homes_to_optimize:
            other_homes = self.homes.copy()
            other_homes.remove(home)
            location_distances = {}
            for location in random.sample(path, max_num_locations):
                location_distances.update({location:nx.shortest_path_length(self.graph, location, home)})
            best_location = min(location_distances.keys(), key=(lambda l: location_distances[l]))
            add_to_dropoffs(dropoffs, best_location, home)
        return dropoffs

    def find_best_path_addition(self, path, location):
        index_distances = {}
        for i in range(1, len(path) - 1):
            previous_location = path[i - 1]
            next_location = path[i]
            distance_to = nx.shortest_path_length(self.graph, previous_location, location)
            distance_from = nx.shortest_path_length(self.graph, location, next_location)
            total_added_distance = distance_to + distance_from
            index_distances.update({i:total_added_distance})
        if len(index_distances.keys()) > 0:
            target_index = min(index_distances.keys(), key=(lambda k: index_distances[k]))
            path_before = path[:target_index]
            path_after = path[target_index:]
            path_to = nx.shortest_path(self.graph, source=path[target_index - 1], target=location)[1:]
            path_from = nx.shortest_path(self.graph, source=location, target=path[target_index])[1:-1]
            new_path = path_before + path_to + path_from + path_after
        else:
            new_path = path
        return new_path

    def previous_stop(self, solution, index):
        # returns index of last visited stop location before the given index
        stop = self.start
        iterate = True
        while iterate:
            index -= 1
            if solution.path[index] in self.homes:
                stop = solution.path[index]
                iterate = False
            if index <= 0:
                iterate = False
        return solution.path.index(stop)

    def next_stop(self, solution, index):
        # returns index of next visited stop location after the given index
        stop = self.start
        iterate = True
        while iterate:
            index += 1
            if solution.path[index] in self.homes:
                stop = solution.path[index]
                iterate = False
            if index >= len(solution.path) - 1:
                iterate = False
        return solution.path.index(stop)

    def explore_up(self, solution):
        # adds solutions with one additional stop/detour to solution space
        candidate_locations = set(self.locations) - set(solution.path)
        if len(candidate_locations) > 0:
            for location in candidate_locations:
                candidate_path = self.find_best_path_addition(solution.path, location)
                candidate_dropoffs = self.find_best_dropoffs(solution.path)
                candidate_solution = Solution(self.graph, candidate_path, candidate_dropoffs)
                self.add_solution(candidate_solution)

    def explore_down(self, solution):
        # adds solutions with one less stop/detour to solution space
        for location in solution.path:
            if location in solution.dropoffs.keys():
                index = solution.path.index(location)
                previous_node_index = self.previous_stop(solution, index)
                previous_node = solution.path[previous_node_index]
                next_node_index = self.next_stop(solution, index)
                next_node = solution.path[next_node_index]
                shortcut = nx.shortest_path(self.graph, previous_node, next_node)
                path_start = solution.path[:previous_node_index]
                path_end = solution.path[next_node_index - 1:]
                candidate_path = path_start + shortcut + path_end
                candidate_dropoffs = self.find_best_dropoffs(candidate_path)
                candidate_solution = Solution(self.graph, candidate_path, candidate_dropoffs)
                self.add_solution(candidate_solution)

    def explore(self, solution):
        self.num_explore_calls += 1
        # explores the given solution
        if solution.explored:
            pass
        else:
            # print(solution)
            solution.explored = True
            self.explore_up(solution)
            self.explore_down(solution)

    def explore_all(self, solutions):
        # explores all solutions in given set or list
        for solution in solutions.copy():
            self.explore(solution)

def add_to_dropoffs(dropoffs, location, home):
    # adds a dropoff to a given dropoffs dictionary
    if not location in dropoffs.keys():
        dropoffs.update({location:[]})
    dropoffs[location].append(home)

def int_to_chr(i):
    i += 33
    if i > 126:
        i += 65
    return chr(i)

def chr_to_int(c):
    i = ord(c)
    if i > 192:
        i -= 65
    i -= 33
    return i

def solve(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix, params=['10']):
    solver = Solver(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix, params)
    return solver.solve()
