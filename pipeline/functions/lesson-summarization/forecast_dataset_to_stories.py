import pandas as pd
import re, os
import click
from tqdm import tqdm

def split_lines(text):
    text = [' '.join(i.split()) for i in re.split(r'\n{2,}', text)]
    text = [i for i in text if i]
    return text

def preprocess(file, forecast_folder):
#     file = '~/notebooks/Cognitive_Search/sash/data/feb_20/ulm_forecasts.csv'
    df = pd.read_csv(file, usecols=[2,4,5])
    df['reference_id'] = df['reference_id'].apply(lambda x: 0 if x!=x else x).astype(int)
    df = df.where(df['isLesson']==1).dropna()
    df.drop('isLesson', axis=1, inplace=True)
    df['paragraph'] = df['paragraph'].apply(split_lines)
    df = df.reset_index(drop=True)
    df['reference_id'] = df['reference_id'].astype(int)
    df['title'] = df[['reference_id']].apply(lambda x: f'dummy lesson number {x.name} - {x[0]}', axis=1)

#     path_story = '../Presumm2/PreSumm/raw_data/eva_forecast_02_21_2020/'
    path_story = './raw_data/{}/'.format(forecast_folder)

    if not os.path.isdir(path_story):
        print('Path does not exist...')
        print('Creating folder...')
        os.mkdir(path_story)

    for idx, rows in tqdm(df.iterrows()):
        fn = '{:05} - {}.story'.format(idx, rows['reference_id'])
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
@click.argument('forecast_folder')
def convert(filename, forecast_folder):
    print(f'Converting {filename} to {forecast_folder}')
    preprocess(filename, forecast_folder)
    return

if __name__=='__main__':
    cli()
            
