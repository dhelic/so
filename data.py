import xml.etree.ElementTree as ET
import html
import csv
import re
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import so


def read_xml_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        # Yield each line one at a time
        for line in file:
            yield line.strip()


def parse_xml(xml_dir, csv_dir):
    fields = ['Id', 'PostTypeId', 'AcceptedAnswerId', 'ParentId', 'CreationDate',
              'Score', 'ViewCount', 'Body', 'OwnerUserId', 'OwnerDisplayName', 
              'LastEditorUserId', 'LastEditorDisplayName', 'LastEditDate', 'LastActivityDate',
              'Title', 'Tags', 'AnswerCount', 'CommentCount', 'FavoriteCount',
              'ClosedDate', 'CommunityOwnedDate', 'ContentLicense']

    for path in xml_dir.iterdir():
        print(f'Processing file: {path.name}')
        csv_path = csv_dir / f'{path.stem}.csv'

        count = 0
        with open(csv_path, 'w', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fields, quotechar='"', quoting=csv.QUOTE_ALL)
            writer.writeheader()
            for line in read_xml_file(path):
                post = {field: '' for field in fields}
                for field in fields:
                    post[field] = ''
            
                if line.startswith('</posts>'):
                    print('Last line, skipping.')
                else:
                    row = ET.fromstring(line)
                    body = row.attrib['Body']
                    row.attrib['Body'] = html.unescape(body)
                    for key in row.attrib:
                        post[key] = row.attrib[key]
                    writer.writerow(post)
            
                count += 1
                if count % 100000 == 0:
                    print(count)


def merge_csv(csv_dir, filename):
    all_files = list(csv_dir.glob('*.csv'))
    df_list = [pd.read_csv(file) for file in all_files]
    merged_df = pd.concat(df_list, ignore_index=True)

    data_dir = csv_dir / 'final'
    data_dir.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(data_dir / filename, index=False)


def extract_posts_csv(xml_directory, csv_directory, filename):
    xml_dir = Path(xml_directory)
    csv_dir = Path(csv_directory)
    csv_dir.mkdir(parents=True, exist_ok=True)

    parse_xml(xml_dir, csv_dir)
    merge_csv(csv_dir, filename)


def select_csv(csv_directory, filename):
    tqdm.pandas()

    csv_dir = Path(csv_directory)
    df = pd.read_csv(csv_dir / f'final/{filename}')
    print('---------------------------------------------')
    print('Initial DataFrame:')
    print(df)
    print(df.info())

    df['CreationDate'] = pd.to_datetime(df['CreationDate'])
    df = df[['PostTypeId', 'AcceptedAnswerId', 'CreationDate', 'Score', 'ViewCount', 'AnswerCount', 'CommentCount', 'FavoriteCount', 'Body', 'Title', 'Tags']]

    # select relevant data
    df = so.select_data(df)
    so.prepare_data(df)
    print('---------------------------------------------')
    print('DataFrame within relavant time period:')
    print(df.info())

    # parse html content and remove unparsable rows
    html_df = df['Body'].progress_apply(so.extract_texts)
    df = pd.concat([df, html_df], axis=1)
    df = df[~df['code'].str.contains('Could not parse the html', case=False, na=False)]
    df = df.drop('Body', axis=1)
    print('---------------------------------------------')
    print('DataFrame with parsed HTML:')
    print(df.info())

    p = Path(csv_dir / f'final/{filename}')
    selected_p = p.with_stem(f'{p.stem}_selected')
    df.to_csv(selected_p, index=False, errors='ignore')


def extract_tags_csv(csv_directory, filename):
    csv_dir = Path(csv_directory)
    tags_dir = csv_dir / 'tags'
    tags_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(csv_dir / f'final/{filename}')
    questions = so.questions(df)
    print(questions.info())

    # extract tags
    groups = [['python'], ['javascript', 'reactjs', 'html', 'node.js', 'css'], ['|java|'], ['c#'],  ['|r|'], ['android']]
    names = ['python', 'web', 'java', 'csharp', 'r', 'android']
    for name, tags in zip(names, groups):
        pattern = '|'.join(map(re.escape, tags))

        print('---------------------------------------------')
        print(f'Processing: {name}')
        
        tag_questions = questions[questions['Tags'].str.contains(pattern, case=False, na=False)]
        questions = questions[~questions['Tags'].str.contains(pattern, case=False, na=False)]
        print(tag_questions.info())
        tag_questions.to_csv(tags_dir / f'{name}.csv', index=False)
    
    print('---------------------------------------------')
    print(f'Processing other tags')
    print(questions.info())
    questions.to_csv(tags_dir / 'other.csv', index=False)


def prepare_dataset():
    xml_directory = 'data/xml/'
    csv_directory = 'data/csv/'

    filename = 'posts.csv'
    extract_posts_csv(xml_directory, csv_directory, filename)
    select_csv(csv_directory, filename)

    filename = 'posts_selected.csv'
    extract_tags_csv(csv_directory, filename)


if __name__ == '__main__':
   prepare_dataset()
