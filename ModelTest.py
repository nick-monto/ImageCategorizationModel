from ModelTrain import *
import pandas as pd

test_preds = model.predict(test, verbose=1)
results = pd.DataFrame(test_preds, columns=CATEGORIES)
results.insert(0, 'image', test_files)
results.head()
