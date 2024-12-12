
import pandas as pd

def load_bed_file(bed_path):
    # Read the original bed file
    df = pd.read_csv(bed_path, delimiter="\t", header=None,engine='python')

    # Set name for columns
    df.rename(columns={0:"chrom", 1:'start', 2:"end", 3:"name",4:"score", 5:"strand"}, inplace=True)
    try:
        df = df[["chrom", "start", "end", "name", "score", "strand"]]
    except:
        # probably has fewer columns that dataframe
        pass
    return df

bed_path = "/home/laceymw/rlbase/GSM3082833/SRX3892926_galGal6.broadPeak"
approximate_size = 20000
'''
df_og = load_bed_file(bed_path)
df = df_og.copy()
df['width'] = df['end'] - df['start']

_min = df['width'].min() - 1
_max = df['width'].max() + 1

if len(df) > approximate_size:
    # Find min and max to get approximate size of new dataset
    _min = df['width'].quantile(0.5 - 0.5 * approximate_size / len(df), interpolation="nearest")
    _max = df['width'].quantile(0.5 + 0.5 * approximate_size / len(df), interpolation="nearest")
    print(f"Filtering peak width: min: {_min} max: {_max}")
else:
    print(f"Dataset too small to filter:  min: {_min} max: {_max}")

# Filter the original dataframe without width column
filtered_df = df_og[df["width"].between(_min, _max)]

# Save the new file
save_bed_file(filtered_df, new_bed_path)

df = pd.read_csv(peak_file,
                 header=None,
                 sep='\t',
                 names = ["chrom",
                          "chromStart",
                          "chromEnd",
                          "name",
                          "score",
                          "strand",
                          "signalValue",
                          "pValue",
                          "qValue"])

interval_widths = abs(df["chromStart"] - df["chromEnd"])

df2 = df.assign(interval_width = interval_widths)'''