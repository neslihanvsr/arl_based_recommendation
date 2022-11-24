# Business Problem solving using Association Rule Based Recommender System

# Data set:
#
# The dataset consists of the services customers receive and the categories of these services.
# It contains the date and time information of each service received.

# UserId: Customer ID
# ServiceId: Anonymized services belonging to each category.
# A ServiceId can be found under different categories and refers to different services under different categories.
#  (Example: Service with CategoryId 7 and ServiceId 4 is honeycomb cleaning,
#   while service with CategoryId 2 and ServiceId 4 is furniture assembly)
# CategoryId: Anonymized categories. (Example: Cleaning, transportation, renovation category)
# CreateDate: The date the service was purchased


import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules

# preparing the data
df_ = pd.read_csv("recommendation systems/data_set.csv")
df = df_.copy()
df.head()

# ServiceID represents a different service for each CategoryID.
# Combine ServiceID and CategoryID with "_" to create a new variable to represent the services.
df["Service"] = [str(row[1]) + "_" + str(row[2]) for row in df.values]
df.head()

# The data set consists of the date and time the services are received, there is no basket definition (invoice, etc.).
# In order to apply Association Rule Learning, a basket (invoice, etc.) definition must be created.
# Here, the definition of basket is the services that each customer receives monthly.
#  For example; A basket of 9_4, 46_4 services received by the customer with id 7256 in the 8th month of 2017;
#   the 9_4, 38_4 services received in the 10th month of 2017 represent another basket.
#  Baskets must be identified with a unique ID.
# To do this, first create a new date variable containing only the year and month.
# Combine UserID and the newly created date variable with "_" and assign it to a new variable called ID.

df["CreateDate"] = pd.to_datetime(df["CreateDate"])
df.head()
df["NEW_DATE"] = df["CreateDate"].dt.strftime("%Y-%m")
df.head()
df["BasketID"] = [str(row[0]) + "_" + str(row[5]) for row in df.values]
df.head()

df[df["UserId"] == 25446]

# Create Association Rules

# Create the cart service pivot table as below:
# Service         0_8  10_9  11_11  12_7  13_11  14_7  15_1  16_8  17_5  18_4..
# SepetID
# 0_2017-08        0     0      0     0      0     0     0     0     0     0..
# 0_2017-09        0     0      0     0      0     0     0     0     0     0..
# 0_2018-01        0     0      0     0      0     0     0     0     0     0..
# 0_2018-04        0     0      0     0      0     1     0     0     0     0..
# 10000_2017-08    0     0      0     0      0     0     0     0     0     0..

invoice_product_df = df.groupby(['BasketID', 'Service'])['Service'].count().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)
invoice_product_df.head()

# Create association rules:
frequent_itemsets = apriori(invoice_product_df, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
rules.head()

# Use the arl_recommender function to recommend a service to a user who had the last 15_1 service:

def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in sorted_rules["antecedents"].items():
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"]))
    recommendation_list = list({item for item_list in recommendation_list for item in item_list})
    return recommendation_list[:rec_count]

arl_recommender(rules,"15_1", 3)
