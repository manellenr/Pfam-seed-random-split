import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, Bidirectional, LSTM
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

train_df = pd.read_csv("data/train_final.csv")
dev_df = pd.read_csv("data/dev_final.csv")
test_df = pd.read_csv("data/test_final.csv")

# Amino acid vocabulary
amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
               'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
char_dict = {aa: idx + 1 for idx, aa in enumerate(amino_acids)}  # 0 is reserved for padding

# Sequence preprocessing
def integer_encoding(df, seq_col='sequence'):
    return [[char_dict.get(aa, 0) for aa in seq] for seq in df[seq_col]]

def prepare_sequences(df, seq_col='sequence', maxlen=200):
    encoded = integer_encoding(df, seq_col)
    padded = pad_sequences(encoded, maxlen=maxlen, padding='post', truncating='post')
    return padded

maxlen = 200
train_pad = prepare_sequences(train_df, maxlen=maxlen)
val_pad = prepare_sequences(dev_df, maxlen=maxlen)
test_pad = prepare_sequences(test_df, maxlen=maxlen)

# Label encoding
le = LabelEncoder()
y_train_le = le.fit_transform(train_df["family_accession"])
y_val_le = le.transform(dev_df["family_accession"])
y_test_le = le.transform(test_df["family_accession"])

# Model
vocab_size = 21  # 20 amino acids + padding
embedding_dim = 200

input_seq = Input(shape=(maxlen,))
embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(input_seq)

bi_lstm_1 = Bidirectional(LSTM(200, return_sequences=True,
                               kernel_regularizer=l2(1e-4),
                               recurrent_regularizer=l2(1e-4),
                               bias_regularizer=l2(1e-4),
                               activity_regularizer=l2(1e-4)))(embedding_layer)

bi_lstm_2 = Bidirectional(LSTM(200,
                               kernel_regularizer=l2(1e-4),
                               recurrent_regularizer=l2(1e-4),
                               bias_regularizer=l2(1e-4),
                               activity_regularizer=l2(1e-4)))(bi_lstm_1)

x = Dropout(0.2)(bi_lstm_2)
output = Dense(len(le.classes_), activation='softmax')(x)

model = Model(inputs=input_seq, outputs=output)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

# Training
es = EarlyStopping(monitor='val_loss', patience=3, verbose=1)

history = model.fit(
    train_pad, y_train_le,
    epochs=20,
    batch_size=128,
    validation_data=(val_pad, y_val_le),
    callbacks=[es]
)

# Evaluation
print("\n[TEST SET EVALUATION]")
test_loss, test_acc = model.evaluate(test_pad, y_test_le, batch_size=128)
print(f"Test Loss: {test_loss:.4f} - Test Accuracy: {test_acc:.4f}")

# Predictions and report
y_pred_probs = model.predict(test_pad, batch_size=128)
y_pred_labels = np.argmax(y_pred_probs, axis=1)

print("\n[CLASSIFICATION REPORT]")
print(classification_report(y_test_le, y_pred_labels, target_names=le.classes_))

# Plotting
def plot_history(history):
    plt.figure(figsize=(12, 4))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train acc')
    plt.plot(history.history['val_accuracy'], label='val acc')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_history(history)
