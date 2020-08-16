import pandas as pd
import re, os
import click
from tqdm import tqdm
import sys
sys.path.append("pipeline/functions/DataFunctions")
sys.path.append("../DataFunctions")
from utils import *

class Preprocess:
    def __init__(self, args, df, output_folder):
        self.df = df
        self.output_folder = output_folder
        
    def split_lines(self, text):
        text = [' '.join(i.split()) for i in re.split(r'\n{2,}', text)]
        text = [i for i in text if i]
        return text

    def preprocess(self, args):
        if args.mode == "train":
            train_folder = self.output_folder
            
        #     file = '~/notebooks/Cognitive_Search/sash/data/feb_20/train_data_lesson_title.csv'
#             df = pd.read_csv(fisle)
            self.df = self.df.dropna()

            self.df['paragraph'] = self.df['paragraph'].apply(self.split_lines)
            self.df = self.df.reset_index(drop=True)
            self.df.rename(columns={'annotationTitle':'title'}, inplace=True)

            path_story = './raw_data/{}/'.format(train_folder)

            if not os.path.isdir(path_story):
                print(f'[{get_timestamp()} INFO] Path does not exist...')
                print(f'[{get_timestamp()} INFO] Creating folder...')
                os.mkdir(path_story)

            for idx, rows in tqdm(self.df.iterrows()):
                fn = '{:05}.story'.format(idx)
                content = rows['paragraph'] + ['@highlight', rows['title']]
                content = '\n\n'.join(content)
                with open(path_story+fn, 'w+') as f:
                    f.write(content)

            return
        elif args.mode == "predict":
            forecast_folder = self.output_folder
            
        #     file = '~/notebooks/Cognitive_Search/sash/data/feb_20/ulm_forecasts.csv'
#             df = pd.read_csv(file, usecols=[2,4,5])
            self.df['referenceId'] = self.df['referenceId'].apply(lambda x: 0 if x!=x else x).astype(int)
            self.df = self.df.where(self.df['isLesson']==1).dropna()
            self.df.drop('isLesson', axis=1, inplace=True)
            self.df['paragraph'] = self.df['paragraph'].apply(self.split_lines)
            self.df = self.df.reset_index(drop=True)
            self.df['referenceId'] = self.df['referenceId'].astype(int)
            self.df['title'] = self.df[['referenceId']].apply(lambda x: f'dummy lesson number {x.name} - {x[0]}', axis=1)

        #     path_story = '../Presumm2/PreSumm/raw_data/eva_forecast_02_21_2020/'
            path_story = './raw_data/{}/'.format(forecast_folder)

            if not os.path.isdir(path_story):
                print(f'[{get_timestamp()} INFO] Path does not exist...')
                print(f'[{get_timestamp()} INFO] Creating folder...')
                os.mkdir(path_story)

            for idx, rows in tqdm(self.df.iterrows()):
                fn = '{:05} - {}.story'.format(idx, rows['referenceId'])
                content = rows['paragraph'] + ['@highlight', rows['title']]
                content = '\n\n'.join(content)
                with open(path_story+fn, 'w+') as f:
                    f.write(content)

            return

# @click.group()
# def cli():
#     pass

# @cli.command()
# @click.argument('filename')
# @click.argument('forecast_folder')
# def convert(filename, forecast_folder):
#     print(f'Converting {filename} to {forecast_folder}')
#     preprocess(filename, forecast_folder)
#     return

# if __name__=='__main__':
#     cli()
            
