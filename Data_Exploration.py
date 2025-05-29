import pandas as pd

train_df = pd.read_csv("/home/manelle/train.csv")
dev_df = pd.read_csv("/home/manelle/dev.csv")
test_df = pd.read_csv("/home/manelle/test.csv")

def analyze_dataset(df, name):
    stats = {}
    stats['dataset'] = name
    stats['duplicated_sequences'] = df['sequence'].duplicated().sum()
    stats['null_sequences'] = df['sequence'].isnull().sum()
    stats['unique_sequences'] = df['sequence'].dropna().nunique()
    stats['null_families'] = df['family_accession'].isnull().sum()
    stats['unique_families'] = df['family_accession'].dropna().nunique()
    stats['average_sequence_length'] = df['sequence'].dropna().apply(len).mean()
    return stats

stats_train = analyze_dataset(train_df, 'train')
stats_dev = analyze_dataset(dev_df, 'dev')
stats_test = analyze_dataset(test_df, 'test')

all_stats_df = pd.DataFrame([stats_train, stats_dev, stats_test])
print(all_stats_df)

def clean_dataset(df, name):
    print(f"\n------ {name.upper()} ------")

    print("Shape before dropping duplicates:", df.shape)

    df = df.drop_duplicates()
    print("Shape after dropping duplicates:", df.shape)

    duplicated_rows = df['sequence'].duplicated(keep=False)
    duplicated_sequences = df[duplicated_rows]['sequence'].value_counts()

    to_remove = []
    for seq in duplicated_sequences.index:
        subset = df[df['sequence'] == seq]
        if subset['family_accession'].nunique(dropna=True) > 1:
            to_remove.append(seq)

    df = df[~df['sequence'].isin(to_remove)]
    print("Shape after removing ambiguous sequences:", df.shape)

    return df

train_df = clean_dataset(train_df, "train")
dev_df = clean_dataset(dev_df, "dev")
test_df = clean_dataset(test_df, "test")


