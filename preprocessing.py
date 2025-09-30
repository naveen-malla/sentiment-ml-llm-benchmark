# ✅ preprocessing.py (was twitter.py)
import pandas as pd
import re
import nltk
from sklearn.preprocessing import LabelEncoder

nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

def load_and_preprocess_data(train_path, val_path):
    column_names = ["tweet_id", "entity", "sentiment", "tweet"]
    df_train = pd.read_csv(train_path, names=column_names, header=None)
    df_val = pd.read_csv(val_path, names=column_names, header=None)

    df_train = df_train.rename(columns={"tweet": "text", "sentiment": "label"})
    df_val = df_val.rename(columns={"tweet": "text", "sentiment": "label"})

    valid_labels = ["Positive", "Negative", "Neutral"]
    df_train = df_train[df_train['label'].isin(valid_labels)].dropna(subset=["text", "label"])
    df_val = df_val[df_val['label'].isin(valid_labels)].dropna(subset=["text", "label"])

    df_train['clean_text'] = df_train['text'].apply(clean_text)
    df_val['clean_text'] = df_val['text'].apply(clean_text)

    label_encoder = LabelEncoder()
    df_train['label_encoded'] = label_encoder.fit_transform(df_train['label'])
    df_val['label_encoded'] = label_encoder.transform(df_val['label'])

    return df_train['clean_text'], df_val['clean_text'], df_train['label_encoded'], df_val['label_encoded'], label_encoder

if __name__ == "__main__":
    # Optional testing
    load_and_preprocess_data("data/twitter_training.csv", "data/twitter_validation.csv")
