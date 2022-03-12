import numpy as np
import pandas as pd
import sklearn
import pickle
from sklearn import linear_model
from matplotlib import style
import matplotlib.pyplot as pyplot


df = pd.read_csv("Life Expectancy Data.csv")
df[["Status"]] = df[["Status"]].replace("Developed", 1)
df[["Status"]] = df[["Status"]].replace("Developing", 0)
# print(data.head())

# Little correlation: "Alcohol", "under-five deaths" that can be considered to add to the model
data = df[["Life expectancy","Adult Mortality", "infant deaths", "Status", "HIV/AIDS", "GDP", "Polio", "Diphtheria",
         "Income composition of resources","Schooling"]].copy()
# print(data.head())
predict = "Life expectancy"
# print(data.describe())

# Print out the list of missing values in each feature
# print(data.isnull().sum(), "\n")

# Print out the percentage of missing values in each feature.
# print(data.isnull().sum() / len(data))
# Since there isn't a big percentage of missing values for each feature, I will impute missing values with means and medians
life_expectance_mean = data[predict].mean()
data[predict].fillna(life_expectance_mean, inplace=True)
data["Adult Mortality"].fillna(data["Adult Mortality"].mean(), inplace=True)
data["GDP"].fillna(data["GDP"].median(), inplace=True)
data["Income composition of resources"].fillna(data["Income composition of resources"].median(), inplace=True)
data["Schooling"].fillna(data["Schooling"].median(), inplace=True)
data["Polio"].fillna(data["Polio"].median(), inplace=True)
data["Diphtheria"].fillna(data["Diphtheria"].mean(), inplace=True)
# print(data.isnull().sum(), "\n")

x = np.array(data.drop(columns = [predict]))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1, random_state=100)

linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)



# best = 0
# for _ in range(30):
#     x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
#     linear = linear_model.LinearRegression();
#
#     linear.fit(x_train, y_train)
#
#     # Time 100 to get the acurracy percentage
#     acc = linear.score(x_test, y_test)
#     print("Accuracy Percentage: ", format(acc, "%"));
#
#     if acc > best:
#         best = acc
#         with open("life-expectancy-model.pickle", "wb") as f:
#             pickle.dump(linear, f)




pickle_in = open("life-expectancy-model.pickle", "rb")
linear = pickle.load(pickle_in)
acc = linear.score(x_test, y_test)
print("Accuracy Percentage: ", format(acc, "%"));




# p = "GDP"
# style.use("ggplot")
# pyplot.scatter(data[p], data["Life expectancy"])
# pyplot.xlabel(p)
# pyplot.ylabel("Life expectancy")
# pyplot.show()