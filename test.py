from data_set import process_data_set
from model import load_model
from sklearn.metrics import precision_score, f1_score, recall_score
from adasyn import g_mean
import os

def test(model, xtest, ytest, pos_label):
    return [model.score(xtest, ytest),
            precision_score(ytest, model.predict(xtest), pos_label=pos_label),
            recall_score(ytest, model.predict(xtest), pos_label=pos_label),
            f1_score(ytest, model.predict(xtest), pos_label=pos_label),
            g_mean(ytest, model.predict(xtest), pos_label=pos_label),
            ]


def main():
    data_set_names = ['vehicle', 'archive', 'ionosphere', 'vol', 'abalone']
    paths = ['Vehicle', 'archive/diabetes.csv', 'ionosphere/ionosphere.csv', 'Vol_Reg/vowel-context.csv', 'abalone/abalone.csv']
    paths = [os.path.join('datasets', p) for p in paths]
    metrics_name = ['OA', 'Precision', 'recall', 'F1', 'g_mean']
    pos_labels = ['van', 1, 'b', 1, 18]
    result = []
    for data_set_name, path, pos_label in zip(data_set_names, paths, pos_labels):
        ori_model, adasyn_model = load_model(data_set_name)
        (new_x, new_y), (Xtrain, Ytrain), (Xtest, Ytest) = process_data_set(data_set_name, path)
        ori_model.fit(Xtrain, Ytrain)
        adasyn_model.fit(new_x, new_y)
        merge_result = [(i, j) for i, j in
                        zip(test(ori_model, Xtest, Ytest, pos_label), test(adasyn_model, Xtest, Ytest, pos_label))]
        result.append(merge_result)

    for r, n in zip(result, data_set_names):
        print('{}  data_set compare'.format(n))
        for name, answer in zip(metrics_name, r):
            print('{}: (DT:{})          (ADASYN:{})'.format(name, answer[0], answer[1]))
        print('=' * 80)



if __name__ == '__main__':
    main()
