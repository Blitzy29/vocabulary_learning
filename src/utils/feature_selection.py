import lightgbm
from sklearn.feature_selection import RFE


def apply_feature_selection(self, dataset):
    X_train = dataset[self.vardict["transformed"]]
    y_train = dataset[self.vardict["target"]]

    if self.feature_selection == 'Recursive Feature Elimination':

        lightgbm_reg = lightgbm.LGBMRegressor()
        rfe = RFE(estimator=lightgbm_reg, n_features_to_select=None, verbose=0)
        rfe = rfe.fit(X_train, y_train)

        if self.global_config['show_plot']:
            print("\n ----------")
            print("  Recursive Feature Elimination")
            print(" ----------")
            for i_ranking in range(max(rfe.ranking_)):
                for feature in [
                    var
                    for var, ranking in zip(self.vardict["preprocessed"], rfe.ranking_)
                    if ranking == i_ranking
                ]:
                    print("{:2d}. {}".format(i_ranking, feature))
                if i_ranking == 1:
                    print("\n ---------- \n")

        return [
            var for var, support in zip(self.vardict["preprocessed"], rfe.support_) if support
        ]