"""
1. Ordinary linear regression
"""
def linear_regression(x_train, y_train, x_test, y_test):
    from sklearn import linear_model
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    value = linear.score(x_test, y_test)
    return "{0:.2f}".format(value)


"""
2. Ridge regression
"""
def linear_ridge(x_train, y_train, x_test, y_test):
    from sklearn import linear_model
    linear = linear_model.Ridge(alpha=.5)
    linear.fit(x_train, y_train)
    value = linear.score(x_test, y_test)
    return "{0:.2f}".format(value)


"""
3. Lasso regression
"""
def linear_lasso(x_train, y_train, x_test, y_test):
    from sklearn import linear_model
    linear = linear_model.Lasso(alpha=0.1)
    linear.fit(x_train, y_train)
    value = linear.score(x_test, y_test)
    return "{0:.2f}".format(value)


"""
4. LassoLars regression
"""
def linear_lassolars(x_train, y_train, x_test, y_test):
    from sklearn import linear_model
    linear = linear_model.LassoLars(alpha=.1)
    linear.fit(x_train, y_train)
    value = linear.score(x_test, y_test)
    return "{0:.2f}".format(value) 


"""
5. Bayesian Regression
"""
def linear_bayesian_reidge(x_train, y_train, x_test, y_test):
    from sklearn import linear_model
    linear = linear_model.BayesianRidge()
    linear.fit(x_train, y_train)
    value = linear.score(x_test, y_test)
    return "{0:.2f}".format(value)  


"""
6. RANSAC: RANdom SAmple Consensus
"""
def ransac_regressor(x_train, y_train, x_test, y_test):
    from sklearn import linear_model
    linear = linear_model.RANSACRegressor()
    linear.fit(x_train, y_train)
    value = linear.score(x_test, y_test)
    return "{0:.2f}".format(value)   


"""
7. Logistic Regression
"""
def logistic_regression(x_train, y_train, x_test, y_test):
    from sklearn import linear_model
    linear = linear_model.LogisticRegression()
    linear.fit(x_train, y_train)
    value = linear.score(x_test, y_test)
    return "{0:.2f}".format(value)


"""
8. Linear Discriminant Analysis
"""
def linear_discriminant_analysis(x_train, y_train, x_test, y_test):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
    linear = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
    linear.fit(x_train, y_train)
    value = linear.score(x_test, y_test)
    return "{0:.2f}".format(value)


"""
9. Quadratic Discriminant Analysis
"""
def quadratic_discriminant_analysis(x_train, y_train, x_test, y_test):
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    linear = QuadraticDiscriminantAnalysis(store_covariance=True)
    linear.fit(x_train, y_train)
    value = linear.score(x_test, y_test)
    return "{0:.2f}".format(value) 


"""
10. Normal Linear Discriminant
"""
def linear_discriminant_analysis_auto(x_train, y_train, x_test, y_test):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    linear = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    linear.fit(x_train, y_train)
    value = linear.score(x_test, y_test)
    return "{0:.2f}".format(value)  


"""
11. Shrinkage Linear Discriminant
"""
def linear_discriminant_analysis_none(x_train, y_train, x_test, y_test):
    import operator
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    linear = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=None)
    linear.fit(x_train, y_train)
    value = linear.score(x_test, y_test)
    return "{0:.2f}".format(value)   


"""
12. Classification SVM
"""
def svc(x_train, y_train, x_test, y_test):
    from sklearn import svm
    clf = svm.SVC()
    clf.fit(x_train, y_train)
    value = clf.score(x_test, y_test)
    return "{0:.2f}".format(value)    


"""
13. Regression SVM
"""
def svr(x_train, y_train, x_test, y_test):
    from sklearn import svm
    clf = svm.SVR()
    clf.fit(x_train, y_train)
    value = clf.score(x_test, y_test)
    return "{0:.2f}".format(value)


"""
14. Linear SVM classifier
"""
def linear_svc(x_train, y_train, x_test, y_test):
    from sklearn import svm
    clf = svm.LinearSVC()
    clf.fit(x_train, y_train)
    value = clf.score(x_test, y_test)
    return "{0:.2f}".format(value)


"""
15. Classification SGD
"""
def sgd_classifier(x_train, y_train, x_test, y_test):
    from sklearn.linear_model import SGDClassifier
    clf = SGDClassifier()
    clf.fit(x_train, y_train)
    value = clf.score(x_test, y_test)
    return "{0:.2f}".format(value)


"""
16. Classification SGD
"""
def sdg_regressor(x_train, y_train, x_test, y_test):
    from sklearn.linear_model import SGDRegressor
    clf = SGDRegressor()
    clf.fit(x_train, y_train)
    value = clf.score(x_test, y_test)
    return "{0:.2f}".format(value)


"""
17. Nearest Centroid Classifier
"""
def nearest_centroid(x_train, y_train, x_test, y_test):
    from sklearn.neighbors import NearestCentroid
    clf = NearestCentroid()
    clf.fit(x_train, y_train)
    value = clf.score(x_test, y_test)
    return "{0:.2f}".format(value)


"""
18. KNeighborsClassifier
"""
def kneighbors_classifier(x_train, y_train, x_test, y_test):
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier()
    clf.fit(x_train, y_train)
    value = clf.score(x_test, y_test)
    return "{0:.2f}".format(value)


"""
19. Gaussian Naive Bayes
"""
def gaussian_nb(x_train, y_train, x_test, y_test):
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    clf.fit(x_train, y_train)
    value = clf.score(x_test, y_test)  
    return "{0:.2f}".format(value)


"""
20. Bernoulli Naive Bayes
"""
def bernoulli_nb(x_train, y_train, x_test, y_test):
    from sklearn.naive_bayes import BernoulliNB
    clf = BernoulliNB()
    clf.fit(x_train, y_train)
    value = clf.score(x_test, y_test)
    return "{0:.2f}".format(value)


"""
21. Multinomial Naive Bayes
"""
def multinomial_nb(x_train, y_train, x_test, y_test):
    from sklearn.naive_bayes import MultinomialNB
    clf = MultinomialNB()
    clf.fit(x_train, y_train)
    value = clf.score(x_test, y_test)
    return "{0:.2f}".format(value)


"""
22. Classification decision trees
"""
def decision_tree_classifier(x_train, y_train, x_test, y_test):
    from sklearn import tree
    tree = tree.DecisionTreeClassifier()
    tree.fit(x_train, y_train)
    value = tree.score(x_test, y_test)
    return "{0:.2f}".format(value)


"""
23. Regressor decision trees
"""
def decision_tree_regressor(x_train, y_train, x_test, y_test):
    from sklearn import tree
    tree = tree.DecisionTreeRegressor()
    tree.fit(x_train, y_train)
    value = tree.score(x_test, y_test)
    return "{0:.2f}".format(value)


"""
24. Bagging meta-estimator
"""
def bagging_classifier(x_train, y_train, x_test, y_test):
    from sklearn.ensemble import BaggingClassifier
    ensem = BaggingClassifier()
    ensem.fit(x_train, y_train)
    value = ensem.score(x_test, y_test)
    return "{0:.2f}".format(value)


"""
25. Forests of randomized trees
"""
def random_forest_classifier(x_train, y_train, x_test, y_test):
    from sklearn.ensemble import RandomForestClassifier
    ensem = RandomForestClassifier(n_estimators=10)
    ensem.fit(x_train, y_train)
    value = ensem.score(x_test, y_test)
    return "{0:.2f}".format(value)


"""
26. Extremely Randomized Trees
"""
def extra_trees_classifier(x_train, y_train, x_test, y_test):
    from sklearn.ensemble import ExtraTreesClassifier
    ensem = ExtraTreesClassifier()
    ensem.fit(x_train, y_train)
    value = ensem.score(x_test, y_test)
    return "{0:.2f}".format(value)


"""
27. AdaBoost
"""
def ada_boost_classifier(x_train, y_train, x_test, y_test):
    from sklearn.ensemble import AdaBoostClassifier
    ensem = AdaBoostClassifier()
    ensem.fit(x_train, y_train)
    value = ensem.score(x_test, y_test)
    return "{0:.2f}".format(value)


"""
28. Gradient Tree Boosting Classifier
"""
def gradient_boosting_classifier(x_train, y_train, x_test, y_test):
    from sklearn.ensemble import GradientBoostingClassifier
    ensem = GradientBoostingClassifier(random_state=0)
    ensem.fit(x_train, y_train)
    value = ensem.score(x_test, y_test)
    return "{0:.2f}".format(value)


"""
29. Gradient Tree Boosting Regressor
"""
def gradient_boosting_regressor(x_train, y_train, x_test, y_test):
    from sklearn.ensemble import GradientBoostingRegressor
    ensem = GradientBoostingRegressor(random_state=0)
    ensem.fit(x_train, y_train)
    value = ensem.score(x_test, y_test)
    return "{0:.2f}".format(value)


"""
30. Histogram-Based Gradient Boosting
"""
def hist_gradient_boosting_classifier(x_train, y_train, x_test, y_test):
    from sklearn.experimental import enable_hist_gradient_boosting
    from sklearn.ensemble import HistGradientBoostingClassifier
    ensem = HistGradientBoostingClassifier()
    ensem.fit(x_train, y_train)
    value = ensem.score(x_test, y_test)
    return "{0:.2f}".format(value)


"""
31. LabelSpreading 
"""
def label_spreading(x_train, y_train, x_test, y_test):
    from sklearn.semi_supervised import LabelSpreading
    sel = LabelSpreading()
    sel.fit(x_train, y_train)
    value = sel.score(x_test, y_test)
    return "{0:.2f}".format(value)


"""
32. LabelPropagation 
"""
def label_propagation(x_train, y_train, x_test, y_test):
    from sklearn.semi_supervised import LabelPropagation
    sel = LabelPropagation()
    sel.fit(x_train, y_train)
    value = sel.score(x_test, y_test)
    return "{0:.2f}".format(value)


"""
33. Classification neural network 
"""
def mlp_classifier(x_train, y_train, x_test, y_test):
    from sklearn.neural_network import MLPClassifier
    nn = MLPClassifier()
    nn.fit(x_train, y_train)
    value = nn.score(x_test, y_test)
    return "{0:.2f}".format(value)


"""
34. Regressor neural network 
"""
def mlp_regressor(x_train, y_train, x_test, y_test):
    from sklearn.neural_network import MLPRegressor
    nn = MLPRegressor()
    nn.fit(x_train, y_train)
    value = nn.score(x_test, y_test)
    return "{0:.2f}".format(value)







def main_function(x_train, y_train, x_test, y_test):
    predictions_dict  = dict()
    
    predictions_dict['linear_regression'] = linear_regression(x_train, y_train, x_test, y_test)
    predictions_dict['linear_ridge'] = linear_ridge(x_train, y_train, x_test, y_test)
    predictions_dict['linear_lasso'] = linear_lasso(x_train, y_train, x_test, y_test)
    predictions_dict['linear_lassolars'] = linear_lassolars(x_train, y_train, x_test, y_test)
    predictions_dict['linear_bayesian_reidge'] = linear_bayesian_reidge(x_train, y_train, x_test, y_test)
    predictions_dict['ransac_regressor'] = ransac_regressor(x_train, y_train, x_test, y_test)
    predictions_dict['logistic_regression'] = logistic_regression(x_train, y_train, x_test, y_test)
    
    predictions_dict['linear_discriminant_analysis'] = linear_discriminant_analysis(x_train, y_train, x_test, y_test)
    predictions_dict['quadratic_discriminant_analysis'] = quadratic_discriminant_analysis(x_train, y_train, x_test, y_test)
    predictions_dict['linear_discriminant_analysis_auto'] = linear_discriminant_analysis_auto(x_train, y_train, x_test, y_test)
    predictions_dict['linear_discriminant_analysis_none'] = linear_discriminant_analysis_none(x_train, y_train, x_test, y_test)
    
    predictions_dict['svc'] = svc(x_train, y_train, x_test, y_test)
    predictions_dict['svr']  = svr(x_train, y_train, x_test, y_test)
    predictions_dict['linear_svc'] = linear_svc(x_train, y_train, x_test, y_test)
    
    predictions_dict['sgd_classifier'] = sgd_classifier(x_train, y_train, x_test, y_test)
    predictions_dict['sdg_regressor'] = sdg_regressor(x_train, y_train, x_test, y_test)
    
    predictions_dict['nearest_centroid'] =  nearest_centroid(x_train, y_train, x_test, y_test)
    predictions_dict['kneighbors_classifier'] = kneighbors_classifier(x_train, y_train, x_test, y_test)
    
    predictions_dict['gaussian_nb'] = gaussian_nb(x_train, y_train, x_test, y_test)
    predictions_dict['bernoulli_nb'] = bernoulli_nb(x_train, y_train, x_test, y_test)
    predictions_dict['multinomial_nb'] = multinomial_nb(x_train, y_train, x_test, y_test)
    
    predictions_dict['decision_tree_classifier'] = decision_tree_classifier(x_train, y_train, x_test, y_test)
    predictions_dict['decision_tree_regressor'] = decision_tree_regressor(x_train, y_train, x_test, y_test)
    
    predictions_dict['bagging_classifier'] = bagging_classifier(x_train, y_train, x_test, y_test)
    predictions_dict['random_forest_classifier'] = random_forest_classifier(x_train, y_train, x_test, y_test)
    predictions_dict['extra_trees_classifier'] = extra_trees_classifier(x_train, y_train, x_test, y_test)
    predictions_dict['ada_boost_classifier'] = ada_boost_classifier(x_train, y_train, x_test, y_test)
    predictions_dict['gradient_boosting_classifier'] = gradient_boosting_classifier(x_train, y_train, x_test, y_test)
    predictions_dict['gradient_boosting_regressor'] = gradient_boosting_regressor(x_train, y_train, x_test, y_test)
    predictions_dict['hist_gradient_boosting_classifier'] = hist_gradient_boosting_classifier(x_train, y_train, x_test, y_test)
    
    predictions_dict['label_spreading'] = label_spreading(x_train, y_train, x_test, y_test)
    predictions_dict['label_propagation'] = label_propagation(x_train, y_train, x_test, y_test)
    
    predictions_dict['mlp_classifier'] = mlp_classifier(x_train, y_train, x_test, y_test)
    predictions_dict['mlp_regressor'] =mlp_regressor(x_train, y_train, x_test, y_test)
    
    return predictions_dict

def main(val):
    from numpy import load
    x_test = load(val[0])
    x_train = load(val[1])
    y_test = load(val[2])
    y_train = load(val[3])

    result = main_function(x_train, y_train, x_test, y_test) 
    return result