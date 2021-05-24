from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, precision_score, recall_score, accuracy_score, roc_auc_score
from sklearn import metrics
from mlxtend.plotting import plot_confusion_matrix
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

#MATRIZ DE CONFUSIÓN

model = load_model("mimodelo.h5")

batch_size = 20

validation_data_dir = './data/archive/validation'

validation_datagen = ImageDataGenerator(
        rescale=1./255
)

validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='categorical')

names = ['cloudy', 'rain', 'shine', 'sunrise']

predictions = model.predict_generator(generator=validation_generator)

y_pred = np.argmax(predictions, axis=1)
y_real = validation_generator.classes

matc = confusion_matrix(y_real, y_pred)

plot_confusion_matrix(conf_mat=matc, figsize=(9,9), class_names = names, show_normed=False)
plt.tight_layout()
plt.show()

print(metrics.classification_report(y_real,y_pred, digits = 4))
