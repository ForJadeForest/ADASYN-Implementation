# ADASYN-Implementation

Algorithm ADASYN implementation, using datasets given by [paper](https://ieeexplore.ieee.org/document/4633969) 

- Using 5 datasets

  - *Vehicle*

    > This dataset has a total of 846 data examples and 4 classes (opel, saab, bus and van). Each example is represented by 18 attributes. We
    >
    > choose “Van” as the minority class and collapse the remaining classes into one majority class. This gives us an imbalanced two-class dataset, with 199 minority class examples and 647
    >
    > majority class examples.

  - *Pima Indian Diabetes*

    > This is a two-class data set and is used to predict positive diabetes cases. It includes a total of 768 cases with 8 attributes. We use the positive cases as the minority class, which give us 268 minority class cases and 500 majority class cases.

  - *Vowel recognition*

    > This is a speech recognition dataset used to classify different vowels. The original dataset includes 990 examples and 11 classes. Each example is represented by 10 attributes.  Since each vowel in the original data set has 10 examples, we choose the fifirst vowel as the minority class and collapse the rest to be the majority class, which gives 90 and 900 minority and majority examples, respectively.

  - *Ionosphere*

    > This data set includes 351 examples with 2 classes (good radar returns versus bad radar returns). Each example is represented by 34 numeric attributes. We choose the “bad radar” instances as minority class and “good radar” instance as the majority class, which gives us 126 minority class examples and 225 majority class examples.

  - *Abalone*

    >  This data set is used to predict the age of abalone from physical measurements. The original data set includes 4177 examples and 29 classes, and each example is represented by 8 attributes. We choose class “18” as the minority class and class “9” as the majority class. In addition, we also removed the discrete feature (feature “sex”) in our current simulation. This gives us 42 minority class examples and 689 majority class examples; each represented by 7 numerical attributes.

Next is the result given by the paper
<img src="https://jadepicgo.oss-cn-shenzhen.aliyuncs.com/img/image-20220126105207670.png" alt="image-20220126105207670" style="zoom:50%;" />

Next is my result

|  Dataset   | Method |   OA   | Precision | Recall |   F1   | G_mean |
| :--------: | :----: | :----: | :-------: | :----: | :----: | :----: |
|  Vehicle   |   DT   | 0.9104 |  0.7344   | 0.9592 | 0.8319 | 0.9269 |
|  Vehicle   | ADASYN | 0.9151 |   0.746   | 0.9592 | 0.8393 | 0.9301 |
|  Archive   |   DT   | 0.7552 |  0.6571   | 0.6667 | 0.6619 | 0.7325 |
|  Archive   | ADASYN | 0.6927 |  0.5532   | 0.7536 | 0.638  | 0.7045 |
| Ionosphere |   DT   | 0.9205 |   0.931   | 0.8438 | 0.8852 | 0.902  |
| Ionosphere | ADASYN | 0.8977 |  0.8286   | 0.9062 | 0.8657 | 0.8995 |
|    Vol     |   DT   | 0.9718 |  0.8889   | 0.7619 | 0.8205 | 0.869  |
|    Vol     | ADASYN | 0.875  |    0.4    | 0.9524 | 0.5634 | 0.9091 |
|  Abalone   |   DT   | 0.9672 |   0.75    | 0.375  |  0.5   | 0.6106 |
|  Abalone   | ADASYN | 0.8962 |  0.2105   |  0.5   | 0.2963 | 0.6761 |
## File Struction
- `model.py`: load the parameters of decision trees
- `test.py`: show the result
- `data_set.py`: process the dataset
- `adasyn.py`: the algorithm ADASYN implementation and my improved version.
- `perform.ipynb`: test my improved adasyn algorithm performance
- `datasets`: conclude the five datasets
