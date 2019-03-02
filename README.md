# macfi

Python 3.5.2

```pip3 install -r requirements```

```python3 main.py```

### tabular interpretation

```
plain instance: [50. 50.]
counter-factual instance: [25.63443904 49.25723863]
plain instance predict proba: [5.17004319e-06 9.99994830e-01]
counter-factual instance predict proba: [0.65148763 0.34851237]
distance: 0.8353676595444577
```

### text interpretation

```
plain instance: b"A famous quote : when you develop the ability to listen to 'anything' unconditionally without losing your temper or self
confidence, it means you are ......... 'MARRIED'"
counter-factual instance: b" famous quote :    the  to listen to ''    your temper or  ,    are ......... ''"
plain instance predict proba: [0.98753214 0.01246786]
counter-factual instance predict proba: [0.19574992 0.80425008]
distance: 37.445675782877565
```