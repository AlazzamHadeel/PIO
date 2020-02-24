import random as rand
from math import exp
from problem import R, get_number_of_inputs
from problem import L
from problem import U
from problem import calc_fitness
from problem import get_attr
import numpy

AW = .46
BW = .46
CW = .08


def comparator(pigeon):
    return pigeon.fitness()


def cos_sim(gx, x):
    na = numpy.linalg.norm(gx)
    nb = numpy.linalg.norm(x)
    if na == 0 and nb == 0:
        return 1
    if na == 0:
        return 0
    if nb == 0:
        return 0
    d = numpy.dot(gx, x)
    return d / (na * nb)


class Pigeon:

    def __init__(self, random=False):
        if random:
            self.__x = [rand.uniform(L, U) for _ in range(0, get_number_of_inputs())]
            self.__v = [rand.uniform(0, 1) for _ in range(0, get_number_of_inputs())]
        else:
            self.__x = [.0] * get_number_of_inputs()
            self.__v = [.0] * get_number_of_inputs()
        self.__fitness = None
        self.tpr = .0
        self.fpr = .0

    def update_velocity_and_path(self, pg, t):
        self.__v = [(vi * exp(-R * t) + rand.uniform(0, 1) * (pg.__x[i] - self.__x[i])) for i, vi in
                    enumerate(self.__v)]
        self.__x = [xi + self.__v[i] for i, xi in enumerate(self.__x)]
        self.__fitness = None
        return self

    def mutate(self, prop):
        self.__x = [self.__x[i] if prop <= rand.uniform(0, 1) else self.__x[i] + R * rand.uniform(-1, 1)
                    for i in range(0, get_number_of_inputs())]
        self.__fitness = None
        return self

    @staticmethod
    def desirable_destination_center(pop, np):
        pop.sort(key=comparator)
        n = len(pop[0].__x)
        xc = [.0] * n
        xf = [.0] * n
        f = 0
        for i in range(0, np):
            fi = pop[i].fitness()
            f += fi
            for j in range(0, n):
                xf[j] += pop[i].__x[j] * fi
        for c in range(0, n):
            xc[c] = xf[c] / (np * f)
        return xc

    def update_path(self, xc):
        self.__x = [(xi + rand.uniform(0, 1) * (xc[i] - xi)) for i, xi in enumerate(self.__x)]
        self.__fitness = None

    def fitness(self):
        if not self.__fitness:
            self.tpr, self.fpr, n = calc_fitness(self.__x)
            if n == 0:
                return float("inf")
            a = 1.0 / (self.tpr * 100)
            b = self.fpr * 100
            self.__fitness = n + a + b
        return self.__fitness

    def __get_id(self):
        return "-".join(map(str, self.__x))

    def __str__(self):
        return "[" + ", ".join(format(x, "6.6f") for x in self.__x) + "] " + str(self.fitness())

    def __hash__(self):
        return hash(self.__get_id())

    def __eq__(self, other):
        return isinstance(other, Pigeon) and self.__get_id() == other.__get_id()

    def attr(self):
        return get_attr(self.__x)

    def x(self):
        return self.__x;


class CosinePigeon:

    def __init__(self, random=False):
        if random:
            self.__x = [rand.getrandbits(1) for _ in range(0, get_number_of_inputs())]
        else:
            self.__x = [0] * get_number_of_inputs()
        self.__fitness = None
        self.tpr = .0
        self.fpr = .0

    def update_velocity_and_path(self, pg, t):
        v = cos_sim(pg.__x, self.__x)
        self.__x = [self.__x[i] if v > rand.uniform(0, 1) else pg.__x[i] for i in range(0, get_number_of_inputs())]
        self.__fitness = None
        return self

    def mutate(self, prop):
        self.__x = [self.__x[i] if prop <= rand.uniform(0, 1) else 1 - self.__x[i]
                    for i in range(0, get_number_of_inputs())]
        self.__fitness = None
        return self

    @staticmethod
    def desirable_destination_center(pop, np):
        pop.sort(key=comparator)
        n = len(pop[0].__x)
        xc = [.0] * n
        for i in range(0, np):
            for j in range(0, n):
                xc[j] += pop[i].__x[j]
        for j in range(0, n):
            xc[j] = xc[j] / n
        return xc

    # @staticmethod
    # def desirable_destination_center(pop, np):
    #     pop.sort(key=comparator)
    #     n = len(pop[0].__x)
    #     xc = [.0] * n
    #     xf = [.0] * n
    #     f = .0
    #     for i in range(0, np):
    #         fi = pop[i].fitness()
    #         f += fi
    #         for j in range(0, n):
    #             xf[j] += pop[i].__x[j] * fi
    #     for c in range(0, n):
    #         xc[c] = xf[c] / (np * f)
    #     avg = sum(xc) / len(xc)
    #     return [1 if xc[i] >= avg else 0 for i in range(0, n)]

    # def update_path(self, xc):
    #     v = cos_sim(xc, self.__x)
    #     self.__x = [self.__x[i] if v > rand.uniform(0, 1) else xc[i] for i in range(0, get_number_of_inputs())]
    #     self.__fitness = None

    def update_path(self, xc):
        self.__x = [self.__x[i] if xc[i] > rand.uniform(0, 1) else (1 - self.__x[i]) for i in
                    range(0, get_number_of_inputs())]
        self.__fitness = None

    def fitness(self):
        if not self.__fitness:
            self.tpr, self.fpr, n = calc_fitness(self.__x)
            if n == 0:
                return float("inf")
            a = AW * (1.0 / self.tpr)
            b = BW * self.fpr
            c = CW * (n / get_number_of_inputs())
            self.__fitness = a + b + c
        return self.__fitness

    def __get_id(self):
        return "-".join(map(str, self.__x))

    def __str__(self):
        return "[" + ", ".join(format(x, "6.6f") for x in self.__x) + "] " + str(self.fitness())

    def __hash__(self):
        return hash(self.__get_id())

    def __eq__(self, other):
        return isinstance(other, CosinePigeon) and self.__get_id() == other.__get_id()

    def attr(self):
        return get_attr(self.__x)

    def x(self):
        return self.__x;


class SigmoidalPigeon:

    def __init__(self, random=False):
        if random:
            self.__x = [rand.getrandbits(1) for _ in range(0, get_number_of_inputs())]
            self.__v = [rand.uniform(0, 1) for _ in range(0, get_number_of_inputs())]
        else:
            self.__x = [0] * get_number_of_inputs()
            self.__v = [.0] * get_number_of_inputs()
        self.__fitness = None
        self.tpr = .0
        self.fpr = .0

    def update_velocity_and_path(self, pg, t):
        self.__v = [(vi * exp(-R * t) + rand.uniform(0, 1) * (pg.__x[i] - self.__x[i])) for i, vi in
                    enumerate(self.__v)]
        for i in range(0, get_number_of_inputs()):
            s = 1.0 / (1.0 + exp(-self.__v[i]/2))
            self.__x[i] = 1 if s > rand.uniform(0, 1) else 0
        self.__fitness = None
        return self

    @staticmethod
    def desirable_destination_center(pop, np):
        pop.sort(key=comparator)
        n = len(pop[0].__x)
        xc = [.0] * n
        for i in range(0, np):
            for j in range(0, n):
                xc[j] += pop[i].__x[j]
        for j in range(0, n):
            xc[j] = xc[j] / n
        return xc

    def update_path(self, xc):
        self.__x = [self.__x[i] if xc[i] > rand.uniform(0, 1) else (1 - self.__x[i]) for i in
                    range(0, get_number_of_inputs())]
        self.__fitness = None

    def fitness(self):
        if not self.__fitness:
            self.tpr, self.fpr, n = calc_fitness(self.__x)
            if n == 0:
                return float("inf")
            a = AW * (1.0 / self.tpr)
            b = BW * self.fpr
            c = CW * (n / get_number_of_inputs())
            self.__fitness = a + b + c
        return self.__fitness

    def mutate(self, prop):
        self.__x = [self.__x[i] if prop <= rand.uniform(0, 1) else 1 - self.__x[i]
                    for i in range(0, get_number_of_inputs())]
        self.__fitness = None
        return self

    def __get_id(self):
        return "-".join(map(str, self.__x))

    def __str__(self):
        return "[" + ", ".join(format(x, "6.6f") for x in self.__x) + "] " + str(self.fitness())

    def __hash__(self):
        return hash(self.__get_id())

    def __eq__(self, other):
        return isinstance(other, Pigeon) and self.__get_id() == other.__get_id()

    def attr(self):
        return get_attr(self.__x)

    def x(self):
        return self.__x;