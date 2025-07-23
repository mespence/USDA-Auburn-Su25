import pandas as pd

#this just assigns p or np based on whether voltage is above or below a value
def predict1(row):
    if row["voltage"] < 0.051:
        return "NP"
    else:
        return "P"

#def predict2(row):

# print("Running Model")
test_df = pd.read_csv(r"/Users/cathy/coding/bugs2025/USDA-Auburn-Su25/Exploration/Cole/probesplitter/gooddata/sharpshooter_b11_labeled.csv", index_col=0)
test_df.rename(columns={"pre_rect": "voltage"}, inplace = True)
# predictions = predict([test_df])[0]
# print("Model run.")
# print()
# print(predictions)

# print("Saving output...")
# rf_model.save()
# test_df["labels"] = predictions
# test_df.to_csv("out.csv")
# print("Output saved.")



#ground_truth_path = r"/Users/cathy/coding/bugs2025/USDA-Auburn-Su25/Exploration/Cole/probesplitter/gooddata/sharpshooter_b11_labeled.csv"
#output_path = r"/Users/cathy/coding/bugs2025/USDA-Auburn-Su25/Exploration/Cole/probesplitter/out.csv"

df_new = test_df.copy()

df_new["predicted_label"] = df_new.apply(lambda row: predict1(row), axis=1)
df_new.drop("labels", axis=1, inplace=True)

df_new.to_csv("out2.csv")
print("Output saved.")