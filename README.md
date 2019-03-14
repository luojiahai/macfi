# Model-Agnostic Contrastive Interpretation (MACI)

In the context of Interpretable Machine Learning, we define Contrastive Interpretation as an instance and its counter-factual case. We use perturbative approach to generate counter-factual instances.

This program generates a counter-factual instance from a given instance and a binary classifier. The algorithm is: (1) sample around the instance, (2) predict all samples using the predict function of the classifier, (3) calculate distances between samples and the instance, (4) find the sample that is predicted as the opposite class and has the smallest distance.

## run

python 3.5.2

```pip3 install -r requirements```

```python3 main.py```

## tabular interpretation

```
plain instance: [50. 50.]
counter-factual instance: [25.63443904 49.25723863]
plain instance predict proba: [5.17004319e-06 9.99994830e-01]
counter-factual instance predict proba: [0.65148763 0.34851237]
distance: 0.8353676595444577
```

## tabular interpretation (loan dataset)

```
plain instance: [1. 4. 1. 0. 0. 0. 3. 3. 1.]
counter-factual instance: [1. 4. 1. 0. 0. 0. 0. 0. 1.]
plain instance predict proba: [0.82648734 0.17351266]
counter-factual instance predict proba: [0.16351895 0.83648105]
distance: 1.4142135623730951
counter-factual description:
-- from last_fico_range_high > 749.00 to last_fico_range_high <= 649.00
-- from last_fico_range_low > 745.00 to last_fico_range_low <= 645.00
```

## text interpretation

```
plain instance: b"A famous quote : when you develop the ability to listen to 'anything' unconditionally without losing your temper or self
confidence, it means you are ......... 'MARRIED'"
counter-factual instance: b" famous quote :    the  to listen to ''    your temper or  ,    are ......... ''"
plain instance predict proba: [0.98753214 0.01246786]
counter-factual instance predict proba: [0.19574992 0.80425008]
distance: 37.445675782877565
```