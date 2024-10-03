
def grid_search(X_train: np.array, y_train: np.array, random=False) -> Dict:
    X_train = StandardScaler().fit_transform(MinMaxScaler().fit_transform(X_train))
    models = {
        'Linear': LinearRegression(),
        'Ridge': Ridge(),
        'Elastic': ElasticNet(),
        'SVR': SVR(),
        'Tree': DecisionTreeRegressor(),
        # 'Gradient Boosting': GradientBoostingRegressor(n_estimators=4),
        # 'Bagging': BaggingRegressor(n_jobs=4)
    }
    params = {
        'Linear': {
        },
        'Ridge': {
            'alpha': [0.1, 0.5, 1.0],
            'solver': ['auto', 'sparse_cg', 'sag']
        },
        'Elastic': {
            'alpha': [0.1, 0.5, 1.0],
            'l1_ratio': [0.25, 0.5, 0.75],
            'tol': [1e-6, 1e-4, 1e-3],
            'selection': ['cyclic', 'random']
        },
        'SVR': {
            'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
            'degree': [2, 3],
            'gamma': ['scale', 'auto', 0.1, 0.25, 0.2, 0.3]
        },
        'Tree': {
            'max_depth': [6, 12, 16]
        },
        'Gradient Boosting': {
            'loss': ['absolute_error'],
            'learning_rate': [0.01, 1, 2],
            'n_estimators': [30, 50, 100],
            'criterion': ['friedman_mse'],
            'max_depth': [3, 5, 10],
            'n_iter_no_change': [5],
        },
        'Bagging': {
            'base_estimator': [Ridge(alpha=0.1, solver='sparse_cg', positive=False, ), LinearRegression()],
            '_n_estimators': [50, 100],
            '_max_features': [0.1, 0.5, 1.]
        }
    }
    best_params = {}

    for name, model in models.items():
        if random:
            random_search = RandomizedSearchCV(model, params[name], scoring='r2', n_iter=100, cv=5, random_state=42,
                                               n_jobs=-1)
            random_search.fit(X_train, y_train)
            best_params[name] = (random_search.best_params_, random_search.best_score_)
            print(f'{name}: {random_search.best_params_} {random_search.best_score_}')
        else:
            grid = GridSearchCV(model, params[name], cv=5, n_jobs=4, scoring='r2',
                                verbose=True).fit(X_train, y_train)
            best_params[name] = (grid.best_params_, grid.best_score_)
    return best_params


def feature_selection(model, X_train, y_train, sfs_figure, sfs_df_filename, k_features, forward):
    sfs = SFS(model,
              k_features=k_features,
              forward=forward,
              floating=True,
              scoring='neg_mean_squared_error',
              cv=4,
              n_jobs=-1
              )
    sfs = sfs.fit(X_train, y_train)
    sfs_df = pd.DataFrame.from_dict(sfs.get_metric_dict()).T
    sfs_df.to_csv(sfs_df_filename)

    plot_sfs(sfs.get_metric_dict(), kind='std_dev')
    plt.savefig(sfs_figure)
    return sfs


def plot(y_test: np.array, y_pred: np.array) -> None:
    plt.hist(y_pred - y_test, bins=100)
    plt.show()


def feature_importance(pipe: Pipeline, X_test: np.array, y_test: np.array) -> None:
    results = permutation_importance(pipe, X_test, y_test, scoring='r2')
    importance = results.importances_mean
    for i, v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i, v))