import pandas as pd

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


class RecommendationSystem:

    def __init__(self, dataset):

        self.dataset = dataset

        self.rules = None

        self.train()

    def train(self):

        df = pd.read_csv(self.dataset)

        basket = (
            df.groupby(['TransactionID', 'Product'])['Product']
            .count()
            .unstack()
            .fillna(0)
        )

        basket = (basket > 0).astype(int)
        
        frequent_itemsets = apriori(
            basket,
            min_support=0.2,
            use_colnames=True
        )

        self.rules = association_rules(
            frequent_itemsets,
            metric="confidence",
            min_threshold=0.5
        )

    def recommend(self, cart):

        recommendations = set()

        for item in cart:

            matched = self.rules[
                self.rules['antecedents'].apply(
                    lambda x: item in x
                )
            ]

            for _, row in matched.iterrows():

                for product in row['consequents']:

                    if product not in cart:

                        recommendations.add(product)

        return list(recommendations)
