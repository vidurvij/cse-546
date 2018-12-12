from joke_loader import data_load
from Matrix_Completion import Matrix_Completion, SVD, Q5

# train, test, n, m = data_load(False)
<<<<<<< HEAD
train,trainl, test, n, m = data_load(True)
=======
trainl,train, test, n, m = data_load(True)
>>>>>>> 9c2dd3d62afce27fc4bf74e990e8c45f52e36d2d
# print(trainl.shape,train.shape)
# A = Matrix_Completion(train,test,n,m)
# print(A.train.shape)
# A.cross_validation_als()
<<<<<<< HEAD
# SVD(trainl,train,test)
Q5(train,trainl,test)
=======
SVD(trainl,train,test)
>>>>>>> 9c2dd3d62afce27fc4bf74e990e8c45f52e36d2d
