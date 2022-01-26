from sklearn import tree


def load_model(data_set_name):
    if data_set_name == 'vehicle':
        p_1, p_2 = _load_vehicle()
    elif data_set_name == 'archive':
        p_1, p_2 = _load_archive()
    elif data_set_name == 'ionosphere':
        p_1, p_2 = _load_ionosphere()
    elif data_set_name == 'vol':
        p_1, p_2 = _load_vol()
    elif data_set_name == 'abalone':
        p_1, p_2 = _load_abalone()
    return tree.DecisionTreeClassifier(random_state=42, **p_1), tree.DecisionTreeClassifier(random_state=42, **p_2)


def _load_archive():
    params = {'max_depth': 25, 'min_samples_leaf': 41, 'max_leaf_nodes': 9}
    params_adasyn = {'max_depth': 25, 'min_samples_leaf': 40, 'max_leaf_nodes': 9}
    return params, params_adasyn


def _load_vehicle():
    params = {'max_depth': 6, 'min_samples_leaf': 7, 'max_leaf_nodes': 16}
    params_adasyn = {'max_depth': 8, 'min_samples_leaf': 11, 'max_leaf_nodes': 16}
    return params, params_adasyn


def _load_vol():
    params = {'max_depth': 7, 'min_samples_leaf': 3, 'max_leaf_nodes': 20}
    params_adasyn = {'max_depth': 20, 'min_samples_leaf': 28, 'max_leaf_nodes': 6}
    return params, params_adasyn


def _load_ionosphere():
    params = {'max_depth': 48, 'min_samples_leaf': 9, 'max_leaf_nodes': 15}
    params_adasyn = {'max_depth': 10, 'min_samples_leaf': 3, 'max_leaf_nodes': 10}
    return params, params_adasyn

def _load_abalone():
    params = {'max_depth': 10, 'min_samples_leaf': 5, 'max_leaf_nodes': 16}
    params_adasyn = {'max_depth': 7, 'min_samples_leaf': 17, 'max_leaf_nodes': 20}
    return params, params_adasyn
