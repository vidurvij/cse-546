from joke_loader import data_load
from Matrix_Completion import Matrix_Completion, SVD

train, test, n, m = data_load(True)
# A = Matrix_Completion(train,test,n,m)
# A.Regression(1)
SVD(train,test)
