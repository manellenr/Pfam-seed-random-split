import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

train_df = pd.read_csv("/home/manelle/train_final.csv")
dev_df = pd.read_csv("/home/manelle/dev_final.csv")
test_df = pd.read_csv("/home/manelle/test_final.csv")

def analyze_sequence_lengths(df, dataset_name):
    df['sequence_length'] = df['sequence'].str.len()

    min_len = df['sequence_length'].min()
    max_len = df['sequence_length'].max()
    mean_len = df['sequence_length'].mean()
    median_len = df['sequence_length'].median()
    q1 = df['sequence_length'].quantile(0.25)
    q3 = df['sequence_length'].quantile(0.75)

    print(f"Statistics for {dataset_name} dataset")
    print(f"Minimum length: {min_len}")
    print(f"Maximum length: {max_len}")
    print(f"Mean length: {mean_len:.2f}")
    print(f"Median length: {median_len}")
    print(f"1st Quartile (Q1): {q1}")
    print(f"3rd Quartile (Q3): {q3}\n")

    fig = px.histogram(
        df, 
        x='sequence_length', 
        nbins=50, 
        marginal='rug', 
        opacity=0.7,
        title=f"Sequence Length Distribution - {dataset_name}",
        labels={'sequence_length': 'Sequence Length'}
    )

    fig.add_trace(go.Histogram(
        x=df['sequence_length'], 
        histnorm='probability density',
        name='Density',
        opacity=0.4,
        marker_color='orange',
        showlegend=False
    ))

    fig.update_layout(
        xaxis_title="Sequence Length",
        yaxis_title="Count",
        bargap=0.1,
        template="plotly_white"
    )

    fig.show()

def analyze_amino_acid_frequencies(df, dataset_name):
    all_sequences = ''.join(df['sequence'].dropna())
    
    amino_acid_counts = Counter(all_sequences)
    
    freq_df = pd.DataFrame(amino_acid_counts.items(), columns=['Amino Acid', 'Count'])
    freq_df = freq_df.sort_values(by='Count', ascending=False)

    fig = px.bar(
        freq_df,
        x='Amino Acid',
        y='Count',
        title=f"Amino Acid Frequency - {dataset_name}",
        labels={'Count': 'Frequency'},
        text='Count'
    )

    fig.update_layout(
        xaxis_title="Amino Acid",
        yaxis_title="Frequency",
        template="plotly_white"
    )

    fig.show()

analyze_sequence_lengths(train_df, "Train")
analyze_sequence_lengths(dev_df, "Dev")
analyze_sequence_lengths(test_df, "Test")

analyze_amino_acid_frequencies(train_df, "Train")
analyze_amino_acid_frequencies(dev_df, "Dev")
analyze_amino_acid_frequencies(test_df, "Test")