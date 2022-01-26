import numpy as np


def cal_knn(data, others, K):
    p = np.sum(np.square(others), axis=1)
    q = np.sum(np.square(data), axis=1)
    distance = -2 * np.dot(data, others.T) + p.reshape(1, -1) + q.reshape(-1, 1)
    ner_index = distance.argsort()[:, 1:K + 1]
    return ner_index, distance


def adasyn(data, y, labels_name, d_h=0.75, beta=1, K=5):
    np.random.seed(42)
    min_label = labels_name.index[-1]
    max_label = labels_name.index[0]
    ms = data[y == min_label].shape[0]
    ml = data[y == max_label].shape[0]
    imbalance_degree = ms / ml
    if imbalance_degree > d_h:
        return data, y

    G = (ml - ms) * beta
    ratio = []
    p = np.sum(np.square(data), axis=1)
    distance = -2 * np.dot(data, data.T) + p.reshape(1, -1) + p.reshape(-1, 1)
    for index, d in enumerate(data):
        if y[index] == min_label:
            ner_index = distance[index].argsort()[1:K + 1]
            label = y[distance[index].argsort()[1:K + 1]]
            r = label[label == max_label].shape[0] / K
            ratio.append([index, r, ner_index])
        else:
            continue
    r = [ri[1] for ri in ratio]
    ratio_sum = sum(r)
    if ratio_sum == 0:
        print('data is easy to classify! No necessary to do ADASYN')
        return data, y

    g = [round(ri[1] / ratio_sum * G) for ri in ratio]
    new_data = []
    for index, info in enumerate(ratio):
        minority_point_index = info[0]
        x_i = data[minority_point_index]
        ner = cal_knn(np.expand_dims(x_i, 0), data[y == min_label], K)[0]
        for i in range(0, g[index]):
            random_index = np.random.choice(ner.shape[1])
            la = np.random.ranf(1)
            x_zi = data[y == min_label][ner[:, random_index]]
            generate_data = x_i + (x_zi - x_i) * la
            new_data.append(generate_data)
    new_data = np.array(new_data)
    new_data = new_data.squeeze(1)
    new_X = np.vstack((data, new_data))
    new_label = np.array([min_label for _ in range(len(new_data))])
    new_Y = np.hstack((y, new_label))
    return new_X, new_Y


def g_mean(y_true, y_pred, pos_label):
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    for label,pred_label in zip(y_true, y_pred):
        if label == pred_label and label == pos_label:
            tp += 1
        elif label != pred_label and label == pos_label: # 正例预测成了负例
            fn += 1
        elif label != pred_label and label!= pos_label: #负例预测成了正例
            fp += 1
        elif label == pred_label and label != pos_label:
            tn += 1

    return np.sqrt(tp*tn/((tp+fn)*(tn + fp)))


def my_adasyn(data, y, labels_name, d_h=0.75, beta=1, K=5):
    np.random.seed(42)
    min_label = labels_name.index[-1]
    max_label = labels_name.index[0]
    ms = data[y == min_label].shape[0]
    ml = data[y == max_label].shape[0]
    imbalance_degree = ms / ml
    if imbalance_degree > d_h:
        return data, y

    G = (ml - ms) * beta
    ratio = []
    p = np.sum(np.square(data), axis=1)
    distance = -2 * np.dot(data, data.T) + p.reshape(1, -1) + p.reshape(-1, 1)
    for index, d in enumerate(data):
        if y[index] == min_label:
            ner_index = distance[index].argsort()[1:K + 1]
            label = y[distance[index].argsort()[1:K + 1]]
            r = label[label == max_label].shape[0] / K
            ratio.append([index, r, ner_index])
        else:
            continue
    r = [ri[1] for ri in ratio]
    ratio_sum = sum(r)
    if ratio_sum == 0:
        print('data is easy to classify! No necessary to do ADASYN')
        return data, y

    g = [round(ri[1] / ratio_sum * G) for ri in ratio]
    new_data = []
    for index, info in enumerate(ratio):
        minority_point_index = info[0]
        x_i = data[minority_point_index]
        ner = cal_knn(np.expand_dims(x_i, 0), data[y == min_label], K)[0]
        for i in range(0, g[index]):
            generate_data = x_i
            for vec in range(3):
                random_index = np.random.choice(ner.shape[1])
                la = np.random.ranf(1)
                x_zi = data[y == min_label][ner[:, random_index]]
                generate_data = generate_data + (x_zi - x_i) * la

            new_data.append(generate_data)
    new_data = np.array(new_data)
    new_data = new_data.squeeze(1)
    new_X = np.vstack((data, new_data))
    new_label = np.array([min_label for _ in range(len(new_data))])
    new_Y = np.hstack((y, new_label))
    return new_X, new_Y