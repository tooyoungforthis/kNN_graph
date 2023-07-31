# Задание
# Написать скрипт, который вычисляет Knn(K)
# для не направленной, не взвешенной сети, представленной списком ребер
# При взаимодействии с юзером использовать правильную терминологию


from graph_with_clusters_generator import generate_clustered_graph, clustering_graph, draw_graph
from knn import Graph_KNearestNeighbor
from random import sample

"""Рекомендуется брать граф более, чем с 60 вершинами и менее 100
    Со средним кол-вом вершин в кластере 25-30
    Для правильного разбиение на кластеры
    Кол-во соседей должно равняться примерно sqrt(Кол-во вершин) 
    В нашем случае будем брать 60 вершин и 3 кластера
    При увеличении кол-ва кластеров необходимо изменить метод draw_graph"""


def main():
    N = 60
    n_clusters = 3

    Graph = generate_clustered_graph(N, n_clusters)
    clusters = clustering_graph(Graph, n_clusters)

    model = Graph_KNearestNeighbor()
    model.fit(clusters)

    number_of_neighbours = int(pow(len(Graph.nodes()), 0.5)) + 1
    X = sample(list(Graph.nodes()), number_of_neighbours)

    print(X)
    # number_of_neighbours = 0
    # while not 0 < number_of_neighbours <= len(X):
    #     number_of_neighbours = int(
    #         input(f"Введите ко-во соседей от 1 до {len(X)}: "))
    #     if not 0 < number_of_neighbours <= len(X):
    #         number_of_neighbours == 0

    X_pred = model.predict(X, number_of_neighbours)
    print(X_pred)

    draw_graph(Graph, clusters)

    new_node_index = len(Graph.nodes())
    new_nodes = [[new_node_index, value] for value in X]
    Graph.add_edges_from(new_nodes)
    clusters[new_node_index] = X_pred

    draw_graph(Graph, clusters)


if __name__ == "__main__":
    main()
