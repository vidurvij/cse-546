from joke_loader import data_load
from Matrix_Completion import Matrix_Completion, SVD, Q5

# train, test, n, m = data_load(False)
train,trainl, test, n, m = data_load(True)
# print(trainl.shape,train.shape)
A = Matrix_Completion(train,test,n,m)
# print(A.train.shape)
# A.cross_validation_als()
# SVD(trainl,train,test)
Q5(train,trainl,test)
