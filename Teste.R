install.packages("mlr")
install.packages("mlbench")
install.packages("clue")
library(mlr)
library(mlbench)
library(clue)
### CLASSIFICAÇÃO
data(BreastCancer, package = "mlbench")
df = BreastCancer
df$Id = NULL
classif.task = makeClassifTask(id = "BreastCancer", data = df, target = "Class")
classif.task
head(BreastCancer)

### REGRESSAO
data(BostonHousing, package = "mlbench")
regr.task = makeRegrTask(id = "bh", data = BostonHousing, target = "medv")
regr.task
head(BostonHousing)

### CLUSTER
data(mtcars, package = "datasets")
cluster.task = makeClusterTask(data = mtcars)
cluster.task

### ANÁLISE DE SOBREVIDA
data(lung, package = "survival")
lung$status = (lung$status == 2) # convert to logical
surv.task = makeSurvTask(data = lung, target = c("time", "status"))
surv.task
head(lung)

n = nrow(iris)
iris.train = iris[seq(1, n, by = 2), -5]
iris.test = iris[seq(2, n, by = 2), -5]
task = makeClusterTask(data = iris.train)
mod = train("cluster.kmeans", task)

newdata.pred = predict(mod, newdata = iris.test)
newdata.pred