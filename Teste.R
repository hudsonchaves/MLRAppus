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




library(mlr)
data(iris)

## Define the task:
task = makeClassifTask(id = "tutorial", data = iris, target = "Species")

# id: caracter .ID string do objeto. O default é o nome do banco de dados do R

# data: um dataframe contendo as variáveis explicativas (features) e a variável dependente (target)

# target: [character(1)]|[character(2)]. Nome da variável dependente. Para análise de sobrevida existem os nomes do tempo de sobrevida e a coluna do evento. 

# weights: numérico e opcional, vetor de pesos não negativos a serem usados no ajuste. Não pode ser usado em modelos cost-sensitive. O default é NULL que significa que os pesos são iguais a zero

# blocking: um fator opcional do mesmo tamanho do número de observações. Observações com o mesmo blocking level "belong together". Especificamente, eles são inseridos no conjunto de treinamento ou
# teste durante a reamostragem. O default é NULL que significa que não haverá bloqueio. 

# positive: definir qual a classe receberá o label de positivo. O default é o primeiro nível do fator da variável dependente. 

# fixup.data: Deverá alguma limpeza básica nos dados ser realizada? Atualmente, isto significa remover os níveis vazios de fatores das colunas. Possiveis escolhas são: "no" = não faça isso, 

# "warn"=fazer, mas alertar sobre, "quiet"= fazer e não avisar". O default é "warn".

# check.data = lógico. Os dados devem ser analisados antes da tarefa? O tempo pode ser um fator para não deixar isso ser processado. O default é TRUE

# costs = dataframe. Uma matriz numérica ou data frame contendo os custos de má classificação. Assumimos o caso geral de custos específicos. Isto significa que temos n linhas, correspondendo às

# observações, no mesmo tamanhdo de data. As colunas correspondem às classes e seus nomes são os labels das classes. Cada entrada (i,j) da matriz especifica o custo da classe j predita para a observação i

# censoring = caracter. Tipo de censura. As escolhas permitidas são "rcens" para dados da direita censorados (default). "lcens" para dados da esquerda censorados e "icens" para dados censorados usando o formato 
# "interval2". Ver o pacote Surv para mais detalhes


########
## Define the learner:
########

lrn = makeLearner("classif.lda")

# Para classificação pode-se definir o "predict.type" para termos a probabilidade predita e o valor máximo que seleciona o label. 
# O threshold usado para definir o label pode ser alterado pela função "setThreshold".

# cl: caracter. Classe da aprendizagem. Por convenção, todas as aprendizagem de classificação começam com "cassif" e todas de regressão
# com "regr", todos de análise de sobrevida com "surv" e todas de cluster com "cluster". 

# id: caracter. Id string para o objeto. Usado para mostrar o objeto. O default é "cl". 

# predict.type: caracter. Classificação: "response"=(=labels) ou "prob"(=probabilidades e labels selecionados com máxima probabilidade).
# Regressão: "response"(=y médio) ou "se"(=erro padrão e y médio). Survival: "resposta"(=algum tipo de risco ordenável) ou 
# "prob"(= probabilidades dependentes do tempo). Cluster: "resposta"(=IDS cluster) ou "prob"(=fuzzy cluster membership probabilities). 
# Default é "response"

# fix.factors: lógico. Em alguns casos, ocorre problemas com a aprendizagem para variáveis categóricas durante a predição. Se uma nova variável
# tem níveis de fator LESS que durante o trainamento (um subconjunto estrito), a aprendizagem pode produzir um erro como  
# "type of predictors in new data do not match that of the training data". Neste caso, pode-se reparar o problema definindo esta opção com "TRUE". 
# Adicionaremos os níveis faltantes do fator à variável teste (mas presente no treinamento) à esta variável. 


## Define the resampling strategy:
rdesc = makeResampleDesc(method = "CV", stratify = TRUE)



## Do the resampling:
r = resample(learner = lrn, task = task, resampling = rdesc, show.info = FALSE)

## Get the mean misclassification error:
r$aggr
#> mmce.test.mean 
#>           0.02