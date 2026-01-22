from transformers import AutoTokenizer, T5EncoderModel
import torch
import pandas as pd
from tqdm import tqdm
import sys


def question(row, lang):
    title = row.get('Title').strip()
    content = row.get('text')
    code = row.get('code')

    if pd.notna(content) and pd.notna(code):
        return f'Problem:\n{title}\n\n{content.strip()}\n\nCode ({lang}):\n{code.strip()}'
    elif pd.notna(content):
        return f'Problem:\n{title}\n\n{content.strip()}'
    elif pd.notna(code):
        return f'Problem:\n{title}\n\nCode ({lang}):\n{code.strip()}'
    else:
        return f'Problem:\n{title}'
    

def embedding(text, model, tokenizer, device):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained('Salesforce/codet5-base')
    model = T5EncoderModel.from_pretrained('Salesforce/codet5-base').to(device)
    model.eval()
    print(model)

    filename = sys.argv[1]
    lang = sys.argv[2]

    df = pd.read_csv(f'data/csv/tags/{filename}.csv')
    print(df.info())
    print(df.head())

    tqdm.pandas(desc='Questions')
    df['question'] = df.progress_apply(lambda row: question(row, lang), axis=1)

    df['question'][:10].to_csv('temp.csv', index=False)

    tqdm.pandas(desc='Embeddings')
    df['embedding'] = df['question'].progress_apply(lambda row: embedding(row, model, tokenizer, device))
    df['embedding'] = df['embedding'].apply(lambda x: x.tolist())
    df = df.drop('question', axis=1)

    df.to_parquet(f'data/parquet/{filename}.parquet', index=False)
    print(df.info())
    print(df.head())
