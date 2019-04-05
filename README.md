# maci

In the context of Interpretable Machine Learning, we define Contrastive Interpretation as an instance and its counter-factual case. We use perturbative approach to generate counter-factual instances.

This program generates a counter-factual instance from a given instance and a binary classifier. The algorithm is: (1) sample around the instance, (2) predict all samples using the predict function of the classifier, (3) calculate distances between samples and the instance, (4) find the sample that is predicted as the opposite class and has the smallest distance.

## run

python 3.5.2

```pip3 install -r requirements```

```python3 main.py```

## tabular interpretation

```
plain instance: [30. 30.]
plain instance prediction: in
plain instance predict proba: [4.56489837e-14 1.00000000e+00]
counter-factual instance: [24.62786451 29.0376169 ]
counter-factual instance prediction: out
counter-factual instance predict proba: [0.58088957 0.41911043]
counter-factual distance: 0.18633412976260152
local absolute instance: [50.0460599  50.60408536]
local absolute instance prediction: in
local absolute instance predict proba: [2.72031224e-06 9.99997280e-01]
local absolute distance: 0.8272692239247156
```

## tabular interpretation (loan dataset)

### dataset properties

```
feature_names:
['loan_amnt',
 'home_ownership',
 'annual_inc',
 'desc',
 'inq_last_6mths',
 'revol_util',
 'last_fico_range_high',
 'last_fico_range_low',
 'pub_rec_bankruptcies']

categorical_names:
{0: ['loan_amnt <= 5400.00',
     '5400.00 < loan_amnt <= 10000.00',
     '10000.00 < loan_amnt <= 15000.00',
     'loan_amnt > 15000.00'],
 1: array(['MORTGAGE', 'NONE', 'OTHER', 'OWN', 'RENT'], dtype=object),
 2: ['annual_inc <= 40057.66',
     '40057.66 < annual_inc <= 59000.00',
     '59000.00 < annual_inc <= 82000.00',
     'annual_inc > 82000.00'],
 3: ['desc <= 4.00',
     '4.00 < desc <= 145.00',
     '145.00 < desc <= 392.00',
     'desc > 392.00'],
 4: ['inq_last_6mths <= 0.00',
     '0.00 < inq_last_6mths <= 1.00',
     'inq_last_6mths > 1.00'],
 5: ['revol_util <= 25.20',
     '25.20 < revol_util <= 49.10',
     '49.10 < revol_util <= 72.23',
     'revol_util > 72.23'],
 6: ['last_fico_range_high <= 649.00',
     '649.00 < last_fico_range_high <= 699.00',
     '699.00 < last_fico_range_high <= 749.00',
     'last_fico_range_high > 749.00'],
 7: ['last_fico_range_low <= 645.00',
     '645.00 < last_fico_range_low <= 695.00',
     '695.00 < last_fico_range_low <= 745.00',
     'last_fico_range_low > 745.00'],
 8: array(['-999', '0', '1', '2'], dtype=object)}

class_names:
['Good Loan', 'Bad Loan']
```

### results

```
plain instance: [1. 4. 1. 0. 0. 0. 3. 3. 1.]
plain instance prediction: Good Loan
plain instance predict proba: [0.82648734 0.17351266]
counter-factual instance: [1. 4. 1. 0. 0. 0. 0. 0. 1.]
counter-factual instance prediction: Bad Loan
counter-factual instance predict proba: [0.16351895 0.83648105]
counter-factual distance: 1.4142135623730951
counter-factual description: 
-- from last_fico_range_high > 749.00 to last_fico_range_high <= 649.00
-- from last_fico_range_low > 745.00 to last_fico_range_low <= 645.00
local absolute instance: [1. 4. 0. 0. 2. 2. 3. 3. 2.]
local absolute instance prediction: Good Loan
local absolute instance predict proba: [0.81821789 0.18178211]
local absolute distance: 1.7320508075688772
```

## tabular interpretation (breast cancer dataset)

### dataset properties

```
feature_names:
['radius_mean',
 'texture_mean',
 'perimeter_mean',
 'area_mean',
 'smoothness_mean',
 'compactness_mean',
 'concavity_mean',
 'concave_points_mean',
 'symmetry_mean',
 'fractal_dimension_mean']

categorical_names:
{0: ['radius_mean <= 11.70',
     '11.70 < radius_mean <= 13.37',
     '13.37 < radius_mean <= 15.78',
     'radius_mean > 15.78'],
 1: ['texture_mean <= 16.17',
     '16.17 < texture_mean <= 18.84',
     '18.84 < texture_mean <= 21.80',
     'texture_mean > 21.80'],
 2: ['perimeter_mean <= 75.17',
     '75.17 < perimeter_mean <= 86.24',
     '86.24 < perimeter_mean <= 104.10',
     'perimeter_mean > 104.10'],
 3: ['area_mean <= 420.30',
     '420.30 < area_mean <= 551.10',
     '551.10 < area_mean <= 782.70',
     'area_mean > 782.70'],
 4: ['smoothness_mean <= 0.09',
     '0.09 < smoothness_mean <= 0.10',
     '0.10 < smoothness_mean <= 0.11',
     'smoothness_mean > 0.11'],
 5: ['compactness_mean <= 0.06',
     '0.06 < compactness_mean <= 0.09',
     '0.09 < compactness_mean <= 0.13',
     'compactness_mean > 0.13'],
 6: ['concavity_mean <= 0.03',
     '0.03 < concavity_mean <= 0.06',
     '0.06 < concavity_mean <= 0.13',
     'concavity_mean > 0.13'],
 7: ['concave_points_mean <= 0.02',
     '0.02 < concave_points_mean <= 0.03',
     '0.03 < concave_points_mean <= 0.07',
     'concave_points_mean > 0.07'],
 8: ['symmetry_mean <= 0.16',
     '0.16 < symmetry_mean <= 0.18',
     '0.18 < symmetry_mean <= 0.20',
     'symmetry_mean > 0.20'],
 9: ['fractal_dimension_mean <= 0.06',
     '0.06 < fractal_dimension_mean <= 0.06',
     '0.06 < fractal_dimension_mean <= 0.07',
     'fractal_dimension_mean > 0.07']}

class_names:
['Benign', 'Malignant']
```

### results

```
plain instance: [3. 3. 3. 3. 1. 3. 3. 3. 2. 2.]
plain instance prediction: Malignant
plain instance predict proba: [0.00565852 0.99434148]
counter-factual instance: [0. 3. 3. 0. 1. 3. 3. 3. 2. 2.]
counter-factual instance prediction: Benign
counter-factual instance predict proba: [0.73770541 0.26229459]
counter-factual distance: 1.4142135623730951
counter-factual description: 
-- from radius_mean > 15.78 to radius_mean <= 11.70
-- from area_mean > 782.70 to area_mean <= 420.30
local absolute instance: [3. 3. 3. 3. 1. 3. 3. 3. 1. 2.]
local absolute instance prediction: Malignant
local absolute instance predict proba: [0.01651708 0.98348292]
local absolute distance: 1.7320508075688772
```

## note
this software is implemented based on the implementation of LIME (https://github.com/marcotcr/lime)
