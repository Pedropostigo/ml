from sklearn.cluster import KMeans as sk_KMeans

class KMeans(sk_KMeans):

    def __init__(self, n_clusters = 8, init = 'k-means++', n_init = 10, max_iter = 300, tol = 0.0001, 
                verbose = 0, random_state = None, copy_x = True, algorithm = 'auto'):
                
                # calls the original method of the KMeans
                super(KMeans, self).__init__(n_clusters = n_clusters, init = init, n_init = n_init, 
                                            max_iter = max_iter, tol = tol, verbose = verbose, 
                                            random_state = random_state, copy_x = copy_x, 
                                            algorithm = algorithm)


    def select_clusters(self, X, beginning_clusters = 2, end_clusters = 10):
        params = self.get_params()
        n_clusters = []
        inertia = []

        for i in range(beginning_clusters, end_clusters + 1):
            params['n_clusters'] = i
            k_means_model = sk_KMeans(**params)
            k_means_model.fit(X = X)

            n_clusters.append(i)
            inertia.append(k_means_model.inertia_)

        return n_clusters, inertia

if __name__ == '__main__':
    import pandas as pd
    data = pd.DataFrame({'x1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 19],
                        'x2': [3, 1, 7, 4, 9, 5, 4, 6, 10, 13],
                        'x3': [7, 3, 4, 1, 9, 4, 5, 3, 8, 2],
                        'y': [2, 3, 4, 3, 6, 1, 2, 3, 4, 5]})
    print(data.head())

    k_means = KMeans()
    n_clusters, inertia = k_means.select_clusters(X = data[['x1', 'x2', 'x3']])

    print(inertia)