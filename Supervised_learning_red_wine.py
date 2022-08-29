import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve, ShuffleSplit, cross_val_score
from sklearn import metrics
from sklearn import tree
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

# function for plotting learning curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    _, axes = plt.subplots(1, 1, figsize=(5, 5))
    axes.set_title(title)
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    # Plot learning curve
    axes.grid()
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes.legend(loc="best")
    return plt
##################################################################### processing data ############################################################################################################
df = pd.read_csv("winequality-red.csv", sep=',')
data = df.as_matrix()
x = data[:,0:-1]
y = data[:,-1]
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=.3,random_state=1,shuffle=False)
##################################################################### Decision Tree ############################################################################################################
decision_tree1 = DecisionTreeClassifier()
decision_tree1.fit(train_x, train_y)
y_predict = decision_tree1.predict(test_x)
print("decision tree before pruning accuracy score(red wine):",metrics.accuracy_score(test_y, y_predict))

tree.plot_tree(decision_tree1)
plt.title("decision_tree1(red wine)")
plt.savefig("decision_tree1(red wine).png")
plt.close()

max_depth = []
accuracy_gini = []
accuracy_entropy = []
for i in range(1,30):
    decision_tree = DecisionTreeClassifier(criterion="gini", max_depth=i)
    decision_tree.fit(train_x, train_y)
    prediction = decision_tree.predict(test_x)
    accuracy_gini.append(metrics.accuracy_score(test_y, prediction))
    decision_tree = DecisionTreeClassifier(criterion="entropy", max_depth=i)
    decision_tree.fit(train_x, train_y)
    prediction = decision_tree.predict(test_x)
    accuracy_entropy.append(metrics.accuracy_score(test_y, prediction))
    max_depth.append(i)
data = pd.DataFrame({"accuracy_gini":pd.Series(accuracy_gini),
 "accuracy_entropy":pd.Series(accuracy_entropy),
 "max_depth":pd.Series(max_depth)})

plt.plot("max_depth","accuracy_gini", data=data, label="gini")
plt.plot("max_depth","accuracy_entropy", data=data, label="entropy")
plt.xlabel("max_depth")
plt.ylabel("accuracy")
plt.legend()
plt.title("Decision tree: accuracy vs max_depth(red wine)")
plt.savefig("Decision tree: accuracy vs max_depth(red wine).png")
plt.close()

best_score = metrics.accuracy_score(test_y, y_predict)
criterions = ["gini","entropy"]
max_depths = list(range(1,30))
min_samples_leafs = list(range(1,5))
best_parameters={"criterion":"gini","max_depth":None,"min_sample_leaf":1}
for k in max_depths:
    for criterion in criterions:
        for min_samples_leaf in min_samples_leafs:
            clf = DecisionTreeClassifier(max_depth=k,criterion=criterion,min_samples_leaf=min_samples_leaf).fit(train_x,train_y)
            score = metrics.accuracy_score(test_y,clf.predict(test_x))
            if(score>best_score):
                best_parameters["max_depth"]=k
                best_parameters["criterion"]=criterion
                best_parameters["min_sample_leaf"]=min_samples_leaf
                best_score=score
# determining best k
print("decision tree: the optimal parameters:", best_parameters)
print("decision tree: the optimal accuracy score with optimal parameters:", best_score)

decision_tree2 = DecisionTreeClassifier(criterion="gini", max_depth=4, min_samples_leaf = 1)
decision_tree2.fit(train_x, train_y)
y_predict = decision_tree2.predict(test_x)
print("decision tree after pruning accuracy score",metrics.accuracy_score(test_y, y_predict))
tree.plot_tree(decision_tree2)
plt.title("decision_tree2(red wine)")
plt.savefig("decision_tree2(red wine).png")
plt.close()

title = "Decision Tree Learning Curves:red_wine_quality(red wine)"
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
estimator = DecisionTreeClassifier(criterion='gini', max_depth=4, min_samples_leaf=1)
plot_learning_curve(estimator, title, train_x, train_y, ylim=(0.0, 1.01),
                    cv=cv, n_jobs=4)
plt.savefig("Decision Tree Learning Curves:red_wine_quality(red wine).png")
plt.close()

##################################################################### Boosting ############################################################################################################

boosting1 = GradientBoostingClassifier(random_state=0).fit(train_x, train_y)
y_predict = boosting1.predict(test_x)
print("boosting before pruning accuracy score:(red wine)",metrics.accuracy_score(test_y, y_predict))


best_score = metrics.accuracy_score(test_y, y_predict)
n_estimators = [50,100,150]
max_depths = [3,6,12]
best_parameters={"n_estimators":100,"max_depth":3}
for k in max_depths:
    for n_estimator in n_estimators:
        clf = GradientBoostingClassifier(n_estimators=n_estimator,max_depth=k).fit(train_x,train_y)
        score = metrics.accuracy_score(test_y,clf.predict(test_x))
        if(score>best_score):
            best_parameters["max_depth"]=k
            best_parameters["n_estimators"]=n_estimator
            best_score=score
# determining best k
print("boosting: the optimal parameters(red wine):", best_parameters)
print("boosting: the optimal accuracy score with optimal parameters(red wine):", best_score)

title = "Boosting Learning Curves:red_wine_quality(red wine)"
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
estimator = GradientBoostingClassifier()
plot_learning_curve(estimator, title, train_x, train_y,
                    cv=cv, ylim=(0.0, 1.01),n_jobs=-1)
plt.savefig("Boosting Tree Learning Curves:red_wine_quality(red wine).png")
plt.close()

##################################################################### KNN ############################################################################################################
knn = KNeighborsClassifier()
knn.fit(train_x, train_y)
# predict the response
y_predict = knn.predict(test_x)
# evaluate accuracy
print("KNN: before optimization accuracy score(red wine):",metrics.accuracy_score(test_y, y_predict))

# creating odd list of K for KNN
neighbors = list(range(1, 100, 2))
# empty list that will hold accuracy scores
accuracy_scores = []
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k).fit(train_x,train_y)
    score = metrics.accuracy_score(test_y,knn.predict(test_x))
    accuracy_scores.append(score)
# determining best k
optimal_k = neighbors[accuracy_scores.index(max(accuracy_scores))]
print("KNN: the optimal number of neighbors(red wine):", optimal_k)
print("KNN: the optimal accuracy score with optimal number of neighbors(red wine):", max(accuracy_scores))

# plot accuracy score vs k
plt.plot(neighbors, accuracy_scores)
plt.xlabel("Number of Neighbors K")
plt.ylabel("accuracy score")
plt.title("KNN accuracy score vs number of neighbors(red wine)")
plt.savefig("KNN accuracy score vs number of neighbors(red wine).png")
plt.close()

best_score = max(accuracy_scores)
neighbors = list(range(1, 100, 2))
weights = ["uniform","distance"]
algorithms = ["auto","ball_tree", "kd_tree", "brute"]
best_parameters={"neighbors":71,"weights":"uniform","algorithms":"auto"}
for k in neighbors:
    for weight in weights:
        for algorithm in algorithms:
            knn = KNeighborsClassifier(n_neighbors=k,weights=weight,algorithm=algorithm).fit(train_x,train_y)
            score = metrics.accuracy_score(test_y,knn.predict(test_x))
            if(score>best_score):
                best_parameters["neighbors"]=k
                best_parameters["weights"]=weight
                best_parameters["algorithms"]=algorithm
                best_score=score
# determining best k
print("KNN: the optimal parameters(red wine):", best_parameters)
print("KNN: the optimal accuracy score with optimal parameters(red wine):", best_score)


title = "KNN Learning Curves:red_wine_quality(red wine)"
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
estimator = KNeighborsClassifier()
plot_learning_curve(estimator, title, train_x, train_y,
                    cv=cv, ylim=(0.0, 1.01),n_jobs=4)
plt.savefig("KNN Learning Curves:red_wine_quality(red wine).png")
plt.close()


##################################################################### SVM ############################################################################################################
clf = SVC()
clf.fit(train_x, train_y)
y_predict = clf.predict(test_x)
print("SVC: before optimization accuracy score:",metrics.accuracy_score(test_y, y_predict))

Cs = [0.01,0.1,1,10,100,1000]
# empty list that will hold accuracy scores
accuracy_scores = []
for k in Cs:
    clf = SVC(C=k).fit(train_x,train_y)
    score = metrics.accuracy_score(test_y,clf.predict(test_x))
    accuracy_scores.append(score)
# determining best k
optimal_k = Cs[accuracy_scores.index(max(accuracy_scores))]
print("SVC: the optimal C:", optimal_k)
print("SVC: the optimal accuracy score with optimal C:", max(accuracy_scores))
# plot accuracy score vs C
plt.plot(Cs, accuracy_scores)
plt.xlabel("C")
plt.ylabel("accuracy score")
plt.title("SVC: accuracy score vs C")
plt.savefig("SVC: accuracy score vs C.png")
plt.close()


best_score = max(accuracy_scores)
Cs = [0.01,0.1,1,10,100,1000]
kernels = ["linear","poly","rbf","sigmoid"]
best_parameters={"C":1000,"kernel":"rbf"}
for k in Cs:
    for kernel in kernels:
        clf = SVC(C=k,kernel=kernel).fit(train_x,train_y)
        score = metrics.accuracy_score(test_y,clf.predict(test_x))
        if(score>best_score):
            best_parameters["C"]=k
            best_parameters["kernel"]=kernel
            best_score=score
# determining best k
print("SVC: the optimal parameters:", best_parameters)
print("SVC: the optimal accuracy score with optimal parameters:", best_score)

title = "SVC Learning Curves:red_wine_quality"
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
estimator = SVC()
plot_learning_curve(estimator, title, train_x, train_y,
                    cv=cv, ylim=(0.0, 1.01),n_jobs=-1)
plt.savefig("SVC Learning Curves:red_wine_quality.png")
plt.close()

##################################################################### Neural Network############################################################################################################
clf = MLPClassifier()
clf.fit(train_x, train_y)
y_predict = clf.predict(test_x)
print("MLP: before optimization accuracy score:",metrics.accuracy_score(test_y, y_predict))

best_score = metrics.accuracy_score(test_y, y_predict)
hidden_layer_sizes = [(10,),(50,),(100,),(500,),(1000,)]
learning_rate_inits = [0.001, 0.01, 0.1]
solvers= ["sgd", "adam"]
best_parameters={"hidden_layer_sizes":(100,),"learning_rate_init":0.001,"solver":"adam"}
for k in hidden_layer_sizes:
    for learning_rate_init in learning_rate_inits:
        for solver in solvers:
            clf = MLPClassifier(hidden_layer_sizes=k,learning_rate_init=learning_rate_init,solver=solver,random_state=0).fit(train_x,train_y)
            score = metrics.accuracy_score(test_y,clf.predict(test_x))
            if(score>best_score):
                best_parameters["hidden_layer_sizes"]=k
                best_parameters["learning_rate_init"]=learning_rate_init
                best_parameters["solver"]=solver
                best_score=score
# determining best k
print("MLP: the optimal parameters:", best_parameters)
print("MLP: the optimal accuracy score with optimal parameters:", best_score)

train_accuracy_score = []
test_accuracy_score = []
for i in range(1,200):
    clf = MLPClassifier(hidden_layer_sizes=(1000,), learning_rate_init=0.001, solver="adam", random_state=0,max_iter = i).fit(
        train_x, train_y)
    test_prediction=clf.predict(test_x)
    test_accuracy_score.append(metrics.accuracy_score(test_y, test_prediction))

    train_prediction=clf.predict(train_x)
    train_accuracy_score.append(metrics.accuracy_score(train_y, train_prediction))

plt.xlabel("Number of Iterations")
plt.ylabel("Accuracy Score")
plt.title("MLP Learning Curve")
plt.plot(train_accuracy_score, label='train_accuracy_score')
plt.plot(test_accuracy_score,label='test_accuracy_score')
plt.ylim([0.1, 1.05])
plt.legend()
plt.savefig("MLP Learning Curve.png")
plt.close()



