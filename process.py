from student_utils import *
import numpy as np
import networkx as nx

class World_Data:
    # where all the input stuff is stored
    def __init__(self, list_of_locations, list_of_homes, starting_car_location, adjacency_matrix):
        self.location_names = list_of_locations
        self.home_names = list_of_homes
        self.start_name = starting_car_location
        self.adjacency_matrix = adjacency_matrix
        self.start = self.location_names.index(starting_car_location)
        self.graph, adj_message = adjacency_matrix_to_graph(adjacency_matrix)
        self.locations = []
        self.homes = []
        for name in self.location_names:
            self.locations.append(self.location_names.index(name))
        for name in self.home_names:
            self.homes.append(self.location_names.index(name))

class Solution:
    # allows for easier analysis of solutions
    def __init__(self, graph, path, dropoffs):
        self.graph = graph
        self.path = path
        self.dropoffs = dropoffs
        self.cost = self.find_cost()
        self.tags = {}

    def find_cost(self):
        cost_given = cost_of_solution(self.graph, self.path, self.dropoffs)
        if cost_given[0] == 'infinite':
            return np.inf
        else:
            return cost_given[0]

    def print_solution(self):
        to_print = False
        if to_print:
            print(str(self.path) + ' ~ ' + str(self.cost))

    def tag(self, key, entry):
        self.tags.update({key:entry})

    def get_tag(self, key):
        return self.tags[key]

def solve(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix, params=[]):
    # the core solving function
    world = World_Data(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix)
    solutions = []
    cluster_solution = best_min_dropoff(world)
    heuristic_solution = best_max_dropoff(world)
    top_down_search_solution = shorter_path_solution(world, heuristic_solution, 0)
    bottom_up_search_solution = longer_path_solution(world, cluster_solution, 1)
    solutions.append(cluster_solution)
    solutions.append(heuristic_solution)
    solutions.append(top_down_search_solution)
    solutions.append(bottom_up_search_solution)
    solution = best_solution(solutions)
    return solution.path, solution.dropoffs

def best_min_dropoff(world):
    # drops all students off at most central location
    results = []
    for location in world.locations:
        if location == world.start:
            path = [location, location]
        else:
            path = nx.shortest_path(world.graph, source=world.start, target=location)
            path += nx.shortest_path(world.graph, source=location, target=world.start)[1:]
        all_locations = world.locations.copy()
        dropoffs = {location:all_locations}
        result = Solution(world.graph, path, dropoffs)
        result.print_solution()
        results.append(result)
    return min(results, key=lambda r: r.cost)

def best_max_dropoff(world):
    # always goes to the closest next house
    homes_remaining = world.homes.copy()
    all_distances = dict(nx.all_pairs_dijkstra_path_length(world.graph))
    specific_path = [world.start]
    current_location = world.start
    dropoffs = {}
    while len(homes_remaining) > 0:
        closest_home = min(homes_remaining, key=lambda h: all_distances[current_location][h])
        if closest_home != world.start:
            path_extension = nx.shortest_path(world.graph, source=current_location, target=closest_home)[1:]
            specific_path += path_extension
        homes_remaining.remove(closest_home)
        current_location = closest_home
        dropoffs.update({closest_home:[closest_home]})
    if current_location != world.start:
        path_extension = nx.shortest_path(world.graph, source=current_location, target=world.start)[1:]
        specific_path += path_extension
    solution = Solution(world.graph, specific_path, dropoffs)
    solution.print_solution()
    return solution

def shorter_path_solution(world, solution, recurse_limit):
    # removes dropoffs from solution until cost is optimized
    # increasing recurse_limit by 1 doubles the runtime!
    candidate_solutions = [solution]
    if recurse_limit > 0:
        for location in solution.path:
            if location in solution.dropoffs.keys():
                index = solution.path.index(location)
                previous_node_index = previous_stop(world, solution, index)
                next_node_index = next_stop(world, solution, index)
                previous_node = solution.path[previous_node_index]
                next_node = solution.path[next_node_index]
                shortcut = nx.shortest_path(world.graph, source=previous_node, target=next_node)
                path_start = solution.path[:previous_node_index]
                path_end = solution.path[next_node_index - 1:]
                new_path = path_start + shortcut + path_end
                new_dropoffs = find_best_dropoffs(world, new_path)
                new_solution = Solution(world.graph, new_path, new_dropoffs)
                new_solution.print_solution()
                if not same_solutions(solution, new_solution):
                    if new_solution.cost < solution.cost:
                        candidate_solutions.append(world, new_solution, recurse_limit)
                    else:
                        new_solution = shorter_path_solution(world, new_solution, recurse_limit - 1)
                candidate_solutions.append(new_solution)
    return best_solution(candidate_solutions)

def longer_path_solution(world, solution, recurse_limit):
    # adds dropoffs to solution until cost is optimized
    # increasing recurse_limit by 1 doubles the runtime!
    candidate_solutions = [solution]
    if recurse_limit > 0:
        candidate_locations = world.locations.copy()
        current_locations = solution.path.copy()
        if len(candidate_locations) > 0:
            for location in current_locations:
                if location in candidate_locations:
                    candidate_locations.remove(location)
            for location in candidate_locations:
                candidate_path = find_best_path_addition(world, solution.path, location)
                candidate_dropoffs = find_best_dropoffs(world, candidate_path)
                candidate_solution = Solution(world.graph, candidate_path, candidate_dropoffs)
                candidate_solutions.append(candidate_solution)
                candidate_solution.print_solution()
        old_solutions = candidate_solutions.copy()
        for old_solution in old_solutions:
            if old_solution.cost < solution.cost:
                candidate_solutions.append(longer_path_solution(world, old_solution, recurse_limit))
            else:
                candidate_solutions.append(longer_path_solution(world, old_solution, recurse_limit - 1))
    return best_solution(candidate_solutions)

def find_best_dropoffs(world, path):
    # returns optimal dropoffs for a given path
    dropoffs = {}
    for home in world.homes:
        other_homes = world.homes.copy()
        other_homes.remove(home)
        individual_solutions = []
        for location in path:
            if location == world.start:
                possible_dropoffs = {world.start:world.homes.copy()}
            else:
                possible_dropoffs = {world.start:other_homes}
                possible_dropoffs.update({location:[home]})
            possible_solution = Solution(world.graph, path, possible_dropoffs)
            possible_solution.tag('DROPOFF_LOCATION', location)
            individual_solutions.append(possible_solution)
        best_dropoff = best_solution(individual_solutions).get_tag('DROPOFF_LOCATION')
        add_to_dropoffs(dropoffs, best_dropoff, home)
    return dropoffs

def find_best_path_addition(world, path, location):
    addition_indices = {}
    for i in range(1, len(path) - 1):
        previous_location = path[i - 1]
        next_location = path[i]
        distance_to = nx.shortest_path_length(world.graph, source=previous_location, target=location)
        distance_from = nx.shortest_path_length(world.graph, source=location, target=next_location)
        added_distance = distance_to + distance_from
        addition_indices.update({i:added_distance})
    if len(addition_indices.keys()) > 0:
        target_index = min(addition_indices.keys(), key=(lambda k: addition_indices[k]))
        path_before = path[:target_index]
        path_after = path[target_index:]
        path_to = nx.shortest_path(world.graph, source=path[target_index - 1], target=location)[1:]
        path_from = nx.shortest_path(world.graph, source=location, target=path[target_index])[1:-1]
        new_path = path_before + path_to + path_from + path_after
    else:
        new_path = path
    return new_path

def add_to_dropoffs(dropoffs, location, home):
    # adds a dropoff to a given dropoffs dictionary
    if not location in dropoffs.keys():
        dropoffs.update({location:[]})
    dropoffs[location].append(home)

def same_solutions(solution_1, solution_2):
    path_match = solution_1.path == solution_2.path
    dropoffs_match = solution_1.dropoffs == solution_2.dropoffs
    return path_match and dropoffs_match

def previous_stop(world, solution, index):
    # returns index of last visited stop location before the given index
    stop = world.start
    iterate = True
    while iterate:
        index -= 1
        if solution.path[index] in world.homes:
            stop = solution.path[index]
            iterate = False
        if index <= 0:
            iterate = False
    return solution.path.index(stop)

def next_stop(world, solution, index):
    # returns index of next visited stop location after the given index
    stop = world.start
    iterate = True
    while iterate:
        index += 1
        if solution.path[index] in world.homes:
            stop = solution.path[index]
            iterate = False
        if index >= len(solution.path) - 1:
            iterate = False
    return solution.path.index(stop)

def best_solution(solutions):
    # returns solution with smallest cost from list of solutions
    return min(solutions, key=lambda s: s.cost)