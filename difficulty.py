from pydoc import text
from datasets import load_dataset
from pathlib import Path
from transformers import AutoTokenizer, T5EncoderModel
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier
import pandas as pd
from tqdm import tqdm


def leetcode_dataset(csv_directory, filename):
    df = load_dataset('greengerong/leetcode', split='train').to_pandas()
    print(df.info())
    print(df.head())

    csv_dir = Path(csv_directory)
    data_dir = csv_dir / 'difficulty'
    data_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(data_dir / filename, index=False)


def multilang_dataframe(df):
    language_columns = ['java', 'c++', 'python', 'javascript']
    difficulty_map = {'Easy': 0, 'Medium': 1, 'Hard': 2}

    samples = []

    for _, row in df.iterrows():
        title = row.get('title').strip()
        content = row.get('content').strip()
        difficulty = row.get('difficulty')
        if difficulty not in difficulty_map:
            continue

        combined_text = f'{title}\n\n{content}'
        difficulty_label = difficulty_map[difficulty.strip()]

        for lang in language_columns:
            code = row.get(lang, '')
            if isinstance(code, str) and code.strip():
                samples.append({'text': f'Problem:\n{combined_text}\n\nCode ({lang}):\n{code.strip()}', 'difficulty': difficulty_label})

    return pd.DataFrame(samples)


def embed(text, model=None, tokenizer=None, device=None):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()


def model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained('Salesforce/codet5-base')
    model = T5EncoderModel.from_pretrained('Salesforce/codet5-base').to(device)
    model.eval()
    print(model)
    return tokenizer, model, device


def embed_leetcode(csv_directory, parquet_directory, tokenizer, model, device):
    csv_dir = Path(csv_directory)
    parquet_dir = Path(parquet_directory)
    parquet_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_dir / 'difficulty/leetcode.csv')
    print(df.info())
    print(df.head())
    print(df['difficulty'].value_counts())

    df = multilang_dataframe(df)
    print(df.info())
    print(df.head())
    print(df['difficulty'].value_counts())

    tqdm.pandas(desc='Embedding texts')
    embeddings = np.vstack(df['text'].progress_apply(embed, model=model, tokenizer=tokenizer, device=device))
    labels = df['difficulty'].values

    df['embedding'] = list(embeddings)
    df.to_parquet(parquet_dir / 'leetcode_embeddings.parquet', index=False)

    return embeddings, labels


def evaluate(embeddings, labels):
    X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, stratify=labels, test_size=0.2, random_state=42)
    
    clf = XGBClassifier(objective='multi:softprob', num_class=3, eval_metric='mlogloss')
    clf.fit(X_train, y_train)

    y_prob = clf.predict_proba(X_test)
    y_pred = clf.predict(X_test)
    roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
    print(y_prob)
    print(f'ROC AUC: {roc_auc:.4f}')

    report = classification_report(y_test, y_pred, target_names=['Easy', 'Medium', 'Hard'])
    print(report)


def train(model_directory, embeddings, labels):
    model_dir = Path(model_directory)
    model_dir.mkdir(parents=True, exist_ok=True)

    clf = XGBClassifier(objective='multi:softprob', num_class=3, eval_metric='mlogloss')
    clf.fit(embeddings, labels)
    clf.save_model(model_dir / 'xgboost_model.json')


def predict(model_directory, parquet_directory, tags_directory):
    model_dir = Path(model_directory)
    parquet_dir = Path(parquet_directory)
    tags_dir = Path(tags_directory)
    
    clf = XGBClassifier()
    clf.load_model(model_dir / 'xgboost_model.json')

    filenames = ['android.parquet', 'r.parquet', 'python.parquet', 'web.parquet', 'csharp.parquet', 'java.parquet', 'other.parquet']
    names = ['android', 'r', 'python', 'web', 'csharp', 'java', 'other']
    for name, filename in zip(names, filenames):
        print(f'Processing: {name}')

        df = pd.read_parquet(parquet_dir / filename)
        print(df.info())
        print(df.head())

        df['embedding'] = df['embedding'].apply(lambda x: np.array(x, dtype=np.float32))
        X = np.stack(df['embedding'].values)
        
        y_prob = clf.predict_proba(X)
        y_pred = clf.predict(X)

        print(y_pred[:10])
        print(y_prob[:10])

        print(np.unique(y_pred, return_counts=True))

        df['Medium'] = y_prob[:, 1]
        df['Hard'] = y_prob[:, 2]

        df = df.drop('embedding', axis=1)

        df.to_csv(tags_dir / f'{name}_c.csv', index=False)


load_dataset = False
embed_dataset = False
evaluate_model = False
train_model = False
predict_difficulty = True

if __name__ == '__main__':
    if load_dataset:
        leetcode_dataset('data/csv/', 'leetcode.csv')

    if embed_dataset:
        tokenizer, model, device = model()
        embeddings, labels = embed_leetcode('data/csv/', 'data/parquet/', tokenizer, model, device)
    elif evaluate_model or train_model:
        df = pd.read_parquet(f'data/parquet/leetcode_embeddings.parquet')
        print(df.info())
        print(df.head())
        embeddings = np.stack(df['embedding'].values)
        labels = df['difficulty'].values

    if evaluate_model:
        evaluate(embeddings, labels)
    
    if train_model:
        train('model/', embeddings, labels)

    if predict_difficulty:
        predict('model/', 'data/parquet/', 'data/csv/tags/')
    