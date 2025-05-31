import pandas as pd

train_df = pd.read_csv("data/train.csv")
dev_df = pd.read_csv("data/dev.csv")
test_df = pd.read_csv("data/test.csv")

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

def clean_dataset(df, name):

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

def process_common_sequences(train_df, dev_df, test_df):

    train_seqs = set(train_df['sequence'].dropna().unique())
    test_seqs = set(test_df['sequence'].dropna().unique())
    dev_seqs = set(dev_df['sequence'].dropna().unique())

    train_test_common = train_seqs.intersection(test_seqs)
    train_dev_common = train_seqs.intersection(dev_seqs)
    test_dev_common = test_seqs.intersection(dev_seqs)

    print(f"Number of common sequences between train and test: {len(train_test_common)}")
    print(f"Number of common sequences between train and dev: {len(train_dev_common)}")
    print(f"Number of common sequences between test and dev: {len(test_dev_common)}")

    test_df_clean = test_df[~test_df['sequence'].isin(train_seqs)].copy()
    print(f"Test shape after removing sequences present in train: {test_df_clean.shape}")

    dev_df_clean = dev_df[~dev_df['sequence'].isin(train_seqs.union(set(test_df_clean['sequence'].dropna().unique())))].copy()
    print(f"Dev shape after removing sequences present in train and cleaned test: {dev_df_clean.shape}")

    print("\n=== Final dataset shapes ===")
    print(f"Train: {train_df.shape}")
    print(f"Test: {test_df_clean.shape}")
    print(f"Dev: {dev_df_clean.shape}")

    return train_df, dev_df_clean, test_df_clean

def analyze_class_distribution(df):
    print("\n=== Class Distribution Analysis ===")
    
    df['seq_length'] = df['sequence'].apply(len)
    
    class_counts = df['family_accession'].value_counts()
    
    print("Top 5 most frequent classes:")
    print(class_counts.head(5))
    
    top5_classes = class_counts.head(5).index
    
    for cls in top5_classes:
        class_data = df[df['family_accession'] == cls]
        
        class_sizes = class_data['seq_length']
        
        mean_size = class_sizes.mean()
        median_size = class_sizes.median()
        q1 = class_sizes.quantile(0.25)
        q3 = class_sizes.quantile(0.75)
        
        print(f"\nStatistics for class '{cls}':")
        print(f"Count: {len(class_sizes)}")
        print(f"Mean: {mean_size:.2f}")
        print(f"Median: {median_size}")
        print(f"1st quartile (Q1, 25%): {q1}")
        print(f"3rd quartile (Q3, 75%): {q3}")


train_df = clean_dataset(train_df, "train")
dev_df = clean_dataset(dev_df, "dev")
test_df = clean_dataset(test_df, "test")

train_df, dev_df, test_df = process_common_sequences(train_df, dev_df, test_df)

stats_train = analyze_dataset(train_df, 'train')
stats_dev = analyze_dataset(dev_df, 'dev')
stats_test = analyze_dataset(test_df, 'test')

all_stats_df = pd.DataFrame([stats_train, stats_dev, stats_test])
print(all_stats_df)

analyze_class_distribution(train_df)
analyze_class_distribution(dev_df)
analyze_class_distribution(test_df)

train_df.to_csv("data/train_final.csv", index=False)
dev_df.to_csv("data//dev_final.csv", index=False)
test_df.to_csv("data//test_final.csv", index=False)
