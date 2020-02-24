# Replace each SigmoidalPigeon with CosinePigeon in this file to use the cosine similarity version of PIO
from pigons import CosinePigeon
from pigons import SigmoidalPigeon
from problem import np, acc__f_score
from problem import init
from problem import number_of_iterations
import copy


def f(value):
    return "{0:.3f}".format(value)


def find_best(pop):
    pg = None
    for p in pop:
        if (not pg) or (pg.fitness() > p.fitness()):
            pg = p
    return pg


def main():
    file = open('fitness.txt', 'w')

    pop = set()

    for i in range(0, np):
        pop.add(SigmoidalPigeon(True))

    pg = find_best(pop)
    gb = copy.deepcopy(pg)

    file.write('global\tbest\r\n')
    file.write(str(gb.fitness()) + '\t' + str(pg.fitness()) + '\r\n')

    for t in range(0, number_of_iterations):
        n_pop = set()
        for p in pop:
            p.update_velocity_and_path(pg, t)
            while p in n_pop:
                p.mutate(0.2)
            n_pop.add(p)

        while len(n_pop) < np:
            n_pop.add(SigmoidalPigeon(True))
            print('Error')

        pop = n_pop
        pg = find_best(pop)
        if pg.fitness() < gb.fitness():
            gb = copy.deepcopy(pg)
        attr = gb.attr()
        print(t, " [tpr = ",f(gb.tpr), ", fpr = ", f(gb.fpr), "] a = ", attr, " ("+str(len(attr))+"), ", "[tpr = ", f(pg.tpr), ", fpr = ", f(pg.fpr),
              "] a = ", len(pg.attr()))

        file.write(str(gb.fitness()) + '\t' + str(pg.fitness()) + '\r\n')

    print("--------------------------------")

    pop = list(pop)

    nnp = np // 2
    while nnp > 2:
        xc = SigmoidalPigeon.desirable_destination_center(pop, nnp)
        for i in range(0, nnp):
            pop[i].update_path(xc)
        pg = find_best(pop)
        if pg.fitness() < gb.fitness():
            gb = copy.deepcopy(pg)
        attr = gb.attr()
        print(" [tpr = ", f(gb.tpr), ", fpr = ", f(gb.fpr), "] a = ", attr, " (" + str(len(attr)) + "), ", "[tpr = ", f(pg.tpr),
              ", fpr = ", f(pg.fpr),
              "] a = ", len(pg.attr()))
        nnp = nnp // 2
        file.write(str(gb.fitness()) + '\t' + str(pg.fitness()) + '\r\n')

    acc, f_score = acc__f_score(gb.x())
    print('acc = '+str(acc) + '\tf_score=' + str(f_score))
    file.close()

    #print(gb)

init()
main()
