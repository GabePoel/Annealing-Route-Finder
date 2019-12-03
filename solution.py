from student_utils import *
import networkx as nx
import random

class Solution:
    # allows for easier analysis of solutions
    def __init__(self, solver, core_path, dropoffs=None):
        self.setup_parameters()
        self.solver = solver
        self.graph = solver.graph
        self.core_path = core_path
        self.full_path = self.generate_full_path()
        if dropoffs == None:
            self.dropoffs = self.find_dropoffs()
        else:
            self.dropoffs = dropoffs
        self.hash_string = self.generate_hash()
        self.hash = self.__hash__()
        self.cost = self.find_cost()
        self.explored = False

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
            return_string += ' path: ' + str(self.full_path) + ' '
        if self.print_cost:
            return_string += ' cost: ' + str(round(self.cost, 2)) + ' '
        if self.print_dropoffs:
            return_string += ' dropoffs: ' + str(self.dropoffs) + ' '
        return_string += ')'
        return return_string

    def setup_parameters(self):
        self.print_path = True
        self.print_cost = True
        self.print_dropoffs = False

    def find_cost(self):
        cost_given = cost_of_solution(self.graph, self.full_path, self.dropoffs)
        if cost_given[0] == 'infinite':
            return np.inf
        else:
            return cost_given[0]

    def generate_full_path(self):
        core_path = self.core_path.copy()
        full_path = [core_path.pop(0)]
        while len(core_path) > 0:
            full_path += self.solver.graph_paths[full_path[-1]][core_path.pop(0)][1:]
        if len(full_path) < 2:
            full_path = [self.solver.start, self.solver.start]
        return full_path

    def generate_hash(self):
        dropoff_locations = list(self.dropoffs.keys())
        dropoff_locations.sort()
        hash_string = ''
        for location in self.full_path:
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

    def find_dropoffs(self):
        dropoffs = {}
        known_dropoffs = self.solver.homes.intersection(set(self.core_path))
        available_dropoffs = self.solver.homes - known_dropoffs
        candidate_dropoffs = random.sample(list(available_dropoffs), min(self.solver.random_dropoff_home_radius, len(available_dropoffs)))
        available_locations = set(self.core_path)
        candidate_locations = random.sample(list(available_locations), min(self.solver.random_dropoff_location_radius, len(available_locations)))
        for home in known_dropoffs:
            add_to_dropoffs(dropoffs, home, home)
        for home in candidate_dropoffs:
            location_distances = {}
            for location in candidate_locations:
                location_distances.update({location:self.solver.graph_lengths[location][home]})
            best_location = min(location_distances.keys(), key=(lambda l: location_distances[l]))
            add_to_dropoffs(dropoffs, best_location, home)
        for home in available_dropoffs - set(candidate_dropoffs):
            location = random.choice(list(available_locations))
            add_to_dropoffs(dropoffs, location, home)
        new_core_path = self.full_path.copy()
        for location in new_core_path:
            if location not in set(dropoffs.keys()).union({self.solver.start}):
                new_core_path.remove(location)
        self.core_path = new_core_path
        return dropoffs

def add_to_dropoffs(dropoffs, location, home):
    # adds a stop to a given dropoff dictionary
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