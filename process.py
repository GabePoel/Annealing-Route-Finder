import heapq
import time
import random
import numpy as np
import networkx as nx
import os
from solution import Solution

class Solver:
    def __init__(self, list_of_locations, list_of_homes, starting_car_location, adjacency_matrix, input_file):
        self.run = True
        self.file = input_file
        self.stop_reason = 'Still running.'
        self.setup_input_information(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix)
        self.setup_monitoring()
        self.setup_display()
        self.setup_data()
        self.setup_solving_paramaters()

    def setup_input_information(self, list_of_locations, list_of_homes, starting_car_location, adjacency_matrix):
        self.location_names = list_of_locations
        self.home_names = list_of_homes
        self.location_count = len(list_of_locations)
        self.home_count = len(list_of_homes)
        self.start_name = starting_car_location
        self.adjacency_matrix = adjacency_matrix
        self.start = list_of_locations.index(starting_car_location)
        self.setup_locations()
        self.setup_homes()
        self.setup_graph()

    def setup_locations(self):
        self.locations = set([])
        for name in self.location_names:
            self.locations.add(self.location_names.index(name))

    def setup_homes(self):
        self.homes = set([])
        for name in self.home_names:
            self.homes.add(self.location_names.index(name))

    def setup_monitoring(self):
        self.calculated_solutions_count = 0
        self.explored_solutions_count = 0

    def setup_graph(self):
        adjacency_matrix_formatted = [[0 if entry == 'x' else entry for entry in row] for row in self.adjacency_matrix]
        for i in range(len(adjacency_matrix_formatted)):
            adjacency_matrix_formatted[i][i] = 0
        self.graph = nx.convert_matrix.from_numpy_matrix(np.matrix(adjacency_matrix_formatted))
        self.graph_paths = dict(nx.all_pairs_shortest_path(self.graph))
        self.graph_lengths = dict(nx.all_pairs_shortest_path_length(self.graph))
        self.density = nx.density(self.graph)

    def setup_display(self):
        self.explore_updates = False
        self.explore_conditions = True
        self.solution_updates = False
        self.final_stats = True

    def setup_data(self):
        self.known_solutions = {}
        self.best_solutions = []

    def setup_solving_paramaters(self):
        if self.location_count < 51:
            self.size = 's'
        elif self.location_count < 101:
            self.size = 'm'
        else:
            self.size = 'l'
        temperatures = {'s':1800, 'm':3000, 'l':6000}
        max_runtimes = {'s':500, 'm':2000, 'l':5000}
        initial_explore_depths = {'s':30, 'm':30, 'l':30}
        final_explore_depths = {'s':3, 'm':4, 'l':6}
        seeding_intervals = {'s':25, 'm':25, 'l':30}
        seeding_minimums = {'s':25, 'm':50, 'l':50}
        random_up_search_radii = {'s':10, 'm':25, 'l':10}
        random_down_search_radii = {'s':10, 'm':25, 'l':10}
        random_across_search_radii = {'s':25, 'm':25, 'l':25}
        random_cycle_improvement_radii = {'s':10, 'm':25, 'l':15}
        random_dropoff_home_radii = {'s':50, 'm':10, 'l':10}
        random_dropoff_location_radii = {'s':50, 'm':10, 'l':10}
        self.temperature = temperatures[self.size]
        self.max_runtime = max_runtimes[self.size]
        self.initial_explore_depth = initial_explore_depths[self.size]
        self.final_explore_depth = final_explore_depths[self.size]
        self.seeding_interval = seeding_intervals[self.size]
        self.seeding_minimum = seeding_minimums[self.size]
        self.random_up_search_radius = random_up_search_radii[self.size]
        self.random_down_search_radius = random_down_search_radii[self.size]
        self.random_across_search_radius = random_across_search_radii[self.size]
        self.random_cycle_improvement_radius = random_cycle_improvement_radii[self.size]
        self.random_dropoff_home_radius = random_dropoff_home_radii[self.size]
        self.random_dropoff_location_radius = random_dropoff_location_radii[self.size]

    def solve_for_input(self):
        self.start_time = time.perf_counter()
        self.time_check = self.start_time
        self.generate_seeding_solutions()
        self.initial_cost = self.get_best_solution().cost
        while self.run:
            self.search()
        best_solution = self.get_best_solution()
        self.total_time = time.perf_counter() - self.start_time
        self.final_cost = self.get_best_solution().cost
        if self.final_stats:
            self.print_stats()
        return best_solution.full_path, best_solution.dropoffs

    def generate_seeding_solutions(self):
        self.generate_min_solution()
        self.generate_mid_solution()
        self.generate_max_solution()

    def generate_min_solution(self):
        candidate_locations = random.sample(list(self.locations), min(self.seeding_minimum, self.location_count))
        for location in candidate_locations:
            if location == self.start:
                new_core_path = [self.start, self.start]
            else:
                new_core_path = [self.start, location, self.start]
            new_solution = Solution(self, new_core_path)
            self.add_solution(new_solution)

    def generate_mid_solution(self):
        solution_sizes = random.sample(list(range(1, self.home_count - 1)), min(self.home_count - 2, self.seeding_interval))
        for size in solution_sizes:
            stops = set(random.sample(list(self.locations - {self.start}), min(size, len(self.locations) - 1)))
            new_core_path = self.make_path(stops)
            new_solution = Solution(self, new_core_path)
            self.add_solution(new_solution)

    def generate_max_solution(self):
        new_core_path = self.make_path(set(self.locations))
        new_solution = Solution(self, new_core_path)
        self.add_solution(new_solution)

    def print_stats(self):
        print('')
        print(' ~~~ Best Solution ~~~ ')
        print(self.get_best_solution())
        print('')
        print(' ~~~ Statistics ~~~ ')
        print('Number of Calculated Solutions: ' + str(self.calculated_solutions_count))
        print('Number of Explore Calls: ' + str(self.explored_solutions_count))
        print('Processing Time: ' + str(self.total_time) + ' seconds')
        print('Ending Reason: ' + self.stop_reason)
        print('')
        print_file_location = os.path.join(os.getcwd(), 'output_results.txt')
        print_file = open(print_file_location, 'a+')
        print_file.write('\n')
        print_file.write('\nData for ' + str(self.file) + ':')
        print_file.write('\n - Input Size: ' + str(self.size))
        print_file.write('\n - Temperature: ' + str(self.temperature))
        print_file.write('\n - Max Runtime: ' + str(self.max_runtime))
        print_file.write('\n - Initial Explore Depth: ' + str(self.initial_explore_depth))
        print_file.write('\n - Final Explore Depth: ' + str(self.final_explore_depth))
        print_file.write('\n - Seeding interval: ' + str(self.seeding_interval))
        print_file.write('\n - Seeding Minimum: ' + str(self.seeding_minimum))
        print_file.write('\n - Explore Up Radius: ' + str(self.random_up_search_radius))
        print_file.write('\n - Explore Down Radius: ' + str(self.random_down_search_radius))
        print_file.write('\n - Explore Across Radius: ' + str(self.random_across_search_radius))
        print_file.write('\n - Cycle Improvement Radius: ' + str(self.random_cycle_improvement_radius))
        print_file.write('\n - Droppoff Home Radius: ' + str(self.random_dropoff_home_radius))
        print_file.write('\n - Dropoff Location Radius ' + str(self.random_dropoff_location_radius))
        print_file.write('\nResults for ' + str(self.file))
        print_file.write('\n - Best Solution: ' + str(self.get_best_solution()))
        print_file.write('\n - Number of Calculated Solutions: ' + str(self.calculated_solutions_count))
        print_file.write('\n - Number of Explore Calls: ' + str(self.explored_solutions_count))
        print_file.write('\n - Processing Time: ' + str(self.total_time) + ' seconds')
        print_file.write('\n - Ending Reason: ' + self.stop_reason)
        print_file.write('\n - Initial Cost: ' + str(self.initial_cost))
        print_file.write('\n - Final Cost: ' + str(self.final_cost))
        print_file.write('\n - Cost Improvement ' + str(self.initial_cost - self.final_cost))
        print_file.write('\n')

    def search(self):
        self.update_search_conditions()
        candidate_solutions = self.get_best_solutions(self.explore_depth)
        if all([s.explored for s in candidate_solutions]):
            self.run = False
            self.stop_reason = 'Exploration limit reached.'
        self.explore_all(candidate_solutions)

    def update_search_conditions(self):
        self.check_time()
        self.energy_level = self.calculated_solutions_count / self.density
        self.explore_depth = int(((self.initial_explore_depth - self.final_explore_depth) * np.exp( - self.energy_level / self.temperature)) + self.final_explore_depth)
        if self.explore_conditions:
            print('Current Explore Conditions:')
            print(' - Energy Level: ' + str(self.energy_level))
            print(' - Explore Depth: ' + str(self.explore_depth))
            print(' - Current Best: ' + str(self.get_best_solution()))
            print(' - Previous Explore Time: ' + str(round(time.perf_counter() - self.time_check, 2)))
            self.time_check = time.perf_counter()

    def check_time(self):
        if time.perf_counter() - self.start_time > self.max_runtime and self.run:
            self.run = False
            self.stop_reason = 'Time limit reached.'

    def explore(self, solution):
        if not solution.explored:
            solution.explored = True
            self.explored_solutions_count += 1
            if self.explore_updates:
                print('explore up on ' + str(solution))
            self.explore_up(solution)
            if self.explore_updates:
                print('explore across on ' + str(solution))
            self.explore_across(solution)
            if self.explore_updates:
                print('explore down on ' + str(solution))
            self.explore_down(solution)

    def explore_all(self, solutions):
        i = 0
        self.check_time()
        while i < len(solutions) and self.run:
            self.explore(solutions[i])
            i += 1

    def explore_up(self, solution):
        # increase |D| by 1
        current_stops = set(solution.dropoffs.keys())
        available_stops = self.locations - current_stops
        candidate_stops = random.sample(list(available_stops), min(self.random_up_search_radius, len(available_stops)))
        i = 0
        self.check_time()
        while i < len(candidate_stops) and self.run:
            new_stop = candidate_stops[i]
            new_core_path = self.add_to_path(solution.core_path, new_stop)
            new_solution = Solution(self, new_core_path)
            self.add_solution(new_solution)
            self.check_time()
            i += 1

    def explore_down(self, solution):
        # decrease |D| by 1
        current_stops = set(solution.dropoffs.keys()) - {self.start}
        candidate_stops = random.sample(list(current_stops), min(self.random_down_search_radius, len(current_stops)))
        i = 0
        self.check_time()
        while i < len(candidate_stops) and self.run:
            new_core_path = self.remove_from_path(solution.core_path, candidate_stops[i])
            new_solution = Solution(self, new_core_path)
            self.add_solution(new_solution)
            self.check_time()
            i += 1

    def explore_across(self, solution):
        # exchange an element of D for an element of L without changing |D|
        current_stops = set(solution.dropoffs.keys())
        available_new_stops = self.locations - current_stops
        available_old_stops = current_stops - {self.start}
        candidate_new_stops = random.sample(list(available_new_stops), min(self.random_across_search_radius, len(available_new_stops)))
        candidate_old_stops = random.sample(list(available_old_stops), min(self.random_across_search_radius, len(available_old_stops)))
        i = 0
        self.check_time()
        while i < len(candidate_old_stops) and self.run:
            j = 0
            while j < len(candidate_new_stops) and self.run:
                new_core_path = [candidate_new_stops[j] if x==candidate_old_stops[i] else x for x in solution.core_path]
                new_solution = Solution(self, new_core_path)
                self.add_solution(new_solution)
                self.check_time()
                j += 1
            i += 1

    def tweak_path(self, solution):
        # echange the ordering that elements of D are visited by P without changing |D|
        swap_indices = random.sample(list(range(0, len(solution.core_path))), min(self.random_cycle_improvement_radius, len(solution.core_path)))
        i = 0
        self.check_time()
        while i < len(swap_indices) and self.run:
            j = 0
            while j < len(solution.core_path) - swap_indices[i] and self.run:
                start_index = swap_indices[i]
                end_index = j + start_index
                path_start = solution.core_path[1:start_index]
                path_end = solution.core_path[end_index - 1:]
                path_middle = solution.core_path[start_index:end_index - 1].reverse()
                new_core_path = path_start + path_middle + path_end
                new_dropoffs = solution.dropoffs
                new_solution = Solution(self, new_core_path, new_dropoffs)
                self.add_solution(new_solution)
                self.check_time()
                j += 1
            i += 1

    def add_to_path(self, core_path, location):
        extra_distance = {}
        for i in range(1, len(core_path)):
            pre_stop = core_path[i - 1]
            post_stop = core_path[i]
            pre_distance = self.graph_lengths[pre_stop][location]
            post_distance = self.graph_lengths[location][post_stop]
            extra_distance.update({i:pre_distance + post_distance})
        insertion_index = min(extra_distance.keys(), key=lambda i: extra_distance[i])
        new_core_path = core_path[:insertion_index] + [location] + core_path[insertion_index:]
        return new_core_path

    def remove_from_path(self, core_path, location):
        if location == self.start:
            new_core_path = core_path.copy()
        elif location in core_path:
            new_core_path = core_path.copy()
            new_core_path.remove(location)
        else:
            new_core_path = core_path.copy()
        return new_core_path

    def make_path(self, stops):
        # make P using greedy heuristics given D
        core_path = [self.start]
        stops_remaining = stops - {self.start}
        while len(stops_remaining) > 0:
            next_stop = min(stops_remaining, key=lambda d: self.graph_lengths[core_path[-1]][d])
            stops_remaining -= {next_stop}
            core_path.append(next_stop)
        core_path.append(self.start)
        return core_path

    def add_solution(self, solution):
        if self.solution_updates:
            print(solution)
        self.calculated_solutions_count += 1
        self.known_solutions.update({solution.hash:solution})
        heapq.heappush(self.best_solutions, solution)

    def get_best_solutions(self, n_best):
        return heapq.nsmallest(n_best, self.best_solutions)

    def get_best_solution(self):
        return self.get_best_solutions(1)[0]

def solve(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix, input_file):
    solver = Solver(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix, input_file)
    return solver.solve_for_input()