import pandas as pd

subset_df = pd.read_csv("../newsimages_25_v1.1/subset.csv")
newsarticles_df = pd.read_csv("../newsimages_25_v1.1/newsarticles.csv")
subset_df.columns = newsarticles_df.columns

subset_df.to_csv("subset1.csv", index=False)
print(f"subset1.csv created successfully.")
