import pandas as pd

df = pd.read_csv(r"G:\HDFS\preprocessed\Event_traces.csv")

# Separate classes
df_success = df[df['Label'] == 'Success']
df_fail = df[df['Label'] == 'Fail']

# Number of fail samples
n_fail = len(df_fail)

# Randomly sample same number of Success logs
df_success_sampled = df_success.sample(n=n_fail, random_state=42)

# Combine
df_balanced = pd.concat([df_success_sampled, df_fail])

# Shuffle
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Check distribution
print(df_balanced['Label'].value_counts())

# Save
df_balanced.to_csv("balanced_log_dataset.csv", index=False)
