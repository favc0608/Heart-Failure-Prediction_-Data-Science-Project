import matplotlib.pyplot as plt
import seaborn as sns
import math
from optbinning import OptimalBinning, BinningProcess   


def plot_simple_hist(data, columns):
    rows = (len(columns) // 3) + (1 if len(columns) % 3 > 0 else 0)
    plt.figure(figsize=(15, 4 * rows))
    for i, col in enumerate(columns):
        plt.subplot(rows, 3, i + 1)
        sns.histplot(data[col], kde=True, color='teal')
        plt.title(col)
    plt.tight_layout()
    plt.show()


def plot_categorical_vs_target(data, cat_cols, target):
    n_cols = 3
    n_rows = math.ceil(len(cat_cols) / n_cols)
    
    plt.figure(figsize=(15, 5 * n_rows))
    
    for i, col in enumerate(cat_cols):
        plt.subplot(n_rows, n_cols, i + 1)
        sns.countplot(x=col, hue=target, data=data, palette='viridis')
        plt.title(f'{col} vs {target}')
        plt.xticks(rotation=15)
        
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(data, num_cols):
    corr = data[num_cols].corr()
    
    plt.figure(figsize=(12, 8))
    
    sns.heatmap(
        corr, 
        annot=True,          
        fmt=".2f",          
        cmap='coolwarm',     
        center=0,           
        linewidths=0.5       
    )
    
    plt.title("HEAT MAP CORRELATION", fontsize=15)
    plt.show()


def binningimportance(df,num_cols, target_col):

    for col in num_cols:
        print(f"\n Processing: {col}")
        print("-" * 40)
    
        optb= OptimalBinning(name=col, dtype="numerical", solver="cp", monotonic_trend="auto_asc_desc")
        optb.fit(df[col], df[target_col])
        binning_table = optb.binning_table.build()
        display(binning_table)
        iv = binning_table.loc['Totals', 'IV']
    
        if iv >= 0.3:
            strength = "Strong predictor"
        elif iv >= 0.1:
            strength = "Medium predictor"
        elif iv >= 0.02:
            strength = "Weak predictor"
        else:
            strength = "Not useful"

        print(f"\n Total IV: {iv:.4f} - {strength}")
        print("=" * 80)
        