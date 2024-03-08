import argparse
import datasets
import pandas
import tensorflow as tf
import numpy
from transformers import RobertaTokenizer
from transformers import TFAutoModel


#tokenizer = transformers.AutoTokenizer.from_pretrained("distilroberta-base")

roberta_model = TFAutoModel.from_pretrained('roberta-base')
#referrence used https://huggingface.co/transformers/v3.2.0/model_doc/roberta.html
def embed_text(examples):
    # Tokenize the text using the tokenizer
    inputs = tokenizer(
        examples["text"],
        return_tensors="tf",
        max_length=64,
        padding="max_length",
        truncation=True
    )

    # Obtain embeddings using the pre-trained RoBERTa model
    outputs = roberta_model(**inputs)

    # Return the last hidden state as embeddings
    return {"embeddings": outputs.last_hidden_state}


tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
def tokenize(examples):
    """Converts the text of each example to "input_ids", a sequence of integers
    representing 1-hot vectors for each token in the text"""
    return tokenizer(examples["text"], truncation=True, max_length=64,
                     padding="max_length")


def to_bow(example):
    """Converts the sequence of 1-hot vectors into a single many-hot vector"""
    vector = numpy.zeros(shape=(tokenizer.vocab_size,))
    vector[example["input_ids"]] = 1
    return {"input_bow": vector}



def train(model_path="model", train_path="train.csv", dev_path="dev.csv"):

    # load the CSVs into Huggingface datasets to allow use of the tokenizer
    hf_dataset = datasets.load_dataset("csv", data_files={
        "train": train_path, "validation": dev_path})

    # the labels are the names of all columns except the first
    labels = hf_dataset["train"].column_names[1:]

    def gather_labels(example):
        """Converts the label columns into a list of 0s and 1s"""
        # the float here is because F1Score requires floats
        return {"labels": [float(example[l]) for l in labels]}

    # convert text and labels to format expected by model
    hf_dataset = hf_dataset.map(gather_labels)
    hf_dataset = hf_dataset.map(tokenize, batched=True)
   
    hf_dataset= hf_dataset.map(embed_text,batched=True)

    # convert Huggingface datasets to Tensorflow datasets
    train_dataset = hf_dataset["train"].to_tf_dataset(
        columns="embeddings",
        label_cols="labels",
        batch_size=16,
        shuffle=True) 
    dev_dataset = hf_dataset["validation"].to_tf_dataset(
        columns="embeddings",
        label_cols="labels",
        batch_size=16)

   

    model = tf.keras.Sequential([
    #tf.keras.layers.Embedding(input_dim=tokenizer.vocab_size, output_dim=128, input_length=50265),
    #referrence used https://www.tensorflow.org/api_docs/python/tf/keras/layers/GRU
    tf.keras.Input(shape=(64,768)),
    #tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=256, return_sequences=True)),
    #tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=128, return_sequences=True)),
    #Using lstm looked more complex and prone to overfitting when compared to Gated Recurrent Unit


    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=128, return_sequences=True)),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(units=64,activation='relu'),
    tf.keras.layers.Dense(units=32,activation='relu'),
    tf.keras.layers.Dense(units=len(labels), activation='sigmoid')
])
    

    # model = tf.keras.Sequential([
    #     tf.keras.layers.Dense(units=256, input_dim=tokenizer.vocab_size, activation='relu',kernel_regularizer=tf.keras.regularizers.l1(0.1)),
    #     tf.keras.layers.Dense(units=128, activation='relu'),
    #     tf.keras.layers.Dropout(0.25),
       
    #     tf.keras.layers.Dense(units=128, activation='relu'),
        


    #     tf.keras.layers.Dense(units=128, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.1)),
    #     tf.keras.layers.Dense(
    #         units=len(labels),
    #         input_dim=tokenizer.vocab_size,
    #         activation='sigmoid')
    #     ])
    

    # specify compilation hyperparameters
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0002),
        loss=tf.keras.losses.binary_crossentropy,
        metrics=[tf.keras.metrics.F1Score(average="micro", threshold=0.5)])
    



    # fit the model to the training data, monitoring F1 on the dev data
    model.fit(
        train_dataset,
        epochs=8,
        validation_data=dev_dataset,
        callbacks=[tf.keras.callbacks.ModelCheckpoint(
    filepath=model_path,
    monitor="val_f1_score",
    mode="max",
    save_best_only=True)])
    


def predict(model_path="model", input_path="test-in.csv"):

    # load the saved model
    model = tf.keras.models.load_model(model_path)

    # load the data for prediction
    # use Pandas here to make assigning labels easier later
    df = pandas.read_csv(input_path)


    # create input features in the same way as in train()
    hf_dataset = datasets.Dataset.from_pandas(df)
    hf_dataset = hf_dataset.map(tokenize, batched=True)
    hf_dataset = hf_dataset.map(embed_text)
    hf_dataset = hf_dataset.map(to_bow)

    
    tf_dataset = hf_dataset.to_tf_dataset(
        columns="embeddings",
        batch_size=16)

    # generate predictions from model
    predictions = numpy.where(model.predict(tf_dataset) > 0.5, 1, 0)

    # assign predictions to label columns in Pandas data frame
    df.iloc[:, 1:] = predictions

    # write the Pandas dataframe to a zipped CSV file
    df.to_csv("submission.zip", index=False, compression=dict(
        method='zip', archive_name=f'submission.csv'))


if __name__ == "__main__":
    # parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices={"train", "predict"})
    args = parser.parse_args()

    # call either train() or predict()
    
    globals()[args.command]()

