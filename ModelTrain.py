from ModelKeras import *
from sklearn.metrics import log_loss

model.fit(X_train, y_train, batch_size=64,
          nb_epoch=1, validation_split=0.2,
          verbose=1, shuffle=True)
preds = model.predict(X_valid, verbose=1)
print("Validation Log Loss: {}".format(log_loss(y_valid, preds)))
