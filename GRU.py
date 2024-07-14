import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall

data = pd.read_csv('fashion_reviews.csv')

data = data.drop(columns=['Title'])
data['Review Text'].fillna('', inplace=True)

le_division = LabelEncoder()
data['Division Name'] = le_division.fit_transform(data['Division Name'])

le_department = LabelEncoder()
data['Department Name'] = le_department.fit_transform(data['Department Name'])

le_class = LabelEncoder()
data['Class Name'] = le_class.fit_transform(data['Class Name'])

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(data['Review Text'])
sequences = tokenizer.texts_to_sequences(data['Review Text'])
word_index = tokenizer.word_index
padded_sequences = pad_sequences(sequences, maxlen=200)

data['Mean Rating'] = data.groupby('Clothing ID')['Rating'].transform('mean')

X = np.hstack([
    padded_sequences,
    data[['Age', 'Division Name', 'Department Name', 'Class Name', 'Mean Rating']].values
])
y = data['Recommended IND'].values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=200))
model.add(GRU(128, return_sequences=True))
model.add(Dropout(0.2))
model.add(GRU(64))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', 
              metrics=['accuracy', Precision(), Recall()])

history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

loss, accuracy, precision, recall = model.evaluate(X_val, y_val)
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
