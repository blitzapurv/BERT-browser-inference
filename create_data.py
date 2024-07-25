# Sample code for splitting ls data into train and val/test sets

import os, json
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import pandas as pd
from datasets import Dataset, DatasetDict


file1 = 'data/relevant_questions.csv'
file2 = 'data/irrelevant_questions.csv'

# Read CSV files
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# Concatenate
df = pd.concat([df1, df2], ignore_index=True)
df = shuffle(df, random_state=21)
train_df, test_df = train_test_split(df, train_size=0.8, random_state=21)

train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)

train_df.to_csv('./data/train.csv')
test_df.to_csv('./data/test.csv')

# Hugging Face datasets
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# DatasetDict
datasets = DatasetDict({
    'train': train_dataset,
    'test': test_dataset
})

datasets.save_to_disk('./data/intent_data')
print(datasets)