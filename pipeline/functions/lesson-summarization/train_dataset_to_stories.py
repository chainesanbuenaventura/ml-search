import pandas as pd
import click
import re, os
from tqdm import tqdm

def split_lines(text):
    text = [' '.join(i.split()) for i in re.split(r'\n{2,}', text)]
    text = [i for i in text if i]
    return text

def preprocess(file, train_folder):
#     file = '~/notebooks/Cognitive_Search/sash/data/feb_20/train_data_lesson_title.csv'
    df = pd.read_csv(file)
    df = df.dropna()
    
    df['paragraph'] = df['paragraph'].apply(split_lines)
    df = df.reset_index(drop=True)
    df.rename(columns={'human generated title':'title'}, inplace=True)

    path_story = './raw_data/{}/'.format(train_folder)

    if not os.path.isdir(path_story):
        print('Path does not exist...')
        print('Creating folder...')
        os.mkdir(path_story)

    for idx, rows in tqdm(df.iterrows()):
        fn = '{:05}.story'.format(idx)
        content = rows['paragraph'] + ['@highlight', rows['title']]
        content = '\n\n'.join(content)
        with open(path_story+fn, 'w+') as f:
            f.write(content)
            
    return

@click.group()
def cli():
    pass

@cli.command()
@click.argument('filename')
@click.argument('train_folder')
def convert(filename, train_folder):
    print(f'Converting {filename} to {train_folder}')
    preprocess(filename, train_folder)
    return

if __name__=='__main__':
    cli()
            
