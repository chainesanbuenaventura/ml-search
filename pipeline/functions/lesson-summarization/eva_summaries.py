import pandas as pd
import glob, re, os
import click

sys.path.append("../DataFunctions")
import ElasticFunctions as ef
import AzureFunctions as af

def split_lines(text):
    text = [' '.join(i.split()) for i in re.split(r'\n{2,}', text)]
    text = [i for i in text if i]
    return text

def get_num(txt):
    return int(re.findall(r'\d+', txt)[0])

def get_ref(txt):
    return int(re.findall(r'\d+', txt)[1])

def merge_and_get_summaries(credentials, forecast_path):
#     forecast_path = 'eva_forecast_02_21_2020'
#     forecasted_lessons_file = '~/notebooks/Cognitive_Search/sash/data/feb_20/ulm_forecasts.csv'
    steps = 1000
    file = f'{forecast_path}.log.{148000+steps}'
#     path = '/data/home/admin01//notebooks/Jude/Presumm2/PreSumm/logs/'
    path = 'logs/'
    path = os.path.join(path, file)

    results = {}
    for suffix in ['gold', 'raw_src', 'candidate']:
        with open(f'{path}.{suffix}', 'r') as f:
            results[suffix] = f.readlines()

    df = pd.DataFrame({'human-generated': results['gold'], 'machine-generated': results['candidate']})
    df['lesson_num'] = df['human-generated'].apply(get_num)
    df['ref_id'] = df['human-generated'].apply(get_ref)
    
    # Get sentences
    # credentials = get_credentials(args.credentials)
    df2 = ef.getSentences(self.credentials)
    df2["machine generated"] = df["machine-generated"]

#     df = pd.read_csv(forecasted_lessons_file, usecols=[1,2,4,5])
#     df['reference_id'] = df['reference_id'].apply(lambda x: 0 if x!=x else x).astype(int)
#     df = df.where(df['isLesson']==1).dropna()
#     df.drop('isLesson', axis=1, inplace=True)
#     df['paragraph'] = df['paragraph'].apply(split_lines)
#     df = df.reset_index(drop=True)
#     df['reference_id'] = df['reference_id'].astype(int)
#     df['lesson_num'] = df.index
#     df.rename(columns={'Project Number':'project_number'}, inplace=True)

#     df_merged = df[['paragraph','reference_id','project_number','lesson_num']].merge(
#                     df_gen[['machine-generated','lesson_num']], on='lesson_num')
    
    ef.updateSentences(self.credentials, df2)
    print(df2.head())
    
    return

# @click.group()
# def cli():
#     pass

# @cli.command()
# @click.argument('forecast_path')
# @click.argument('forecasted_lessons_file')
# def get_summaries(forecast_path):
#     merge_and_get_summaries(forecast_path)
#     return

if __name__=='__main__':
    merge_and_get_summaries(credentials, forecast_path)
            
