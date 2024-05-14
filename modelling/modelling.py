from model.randomforest import RandomForest
from model.svm import SVM
from model.gnb import GNB

def model_predict(data, df, name):
    
    print("RandomForest")
    model = RandomForest("RandomForest", data.get_embeddings(), data.get_type())
    model.train(data)
    model.predict(data.X_test)
    model.print_results(data)

    print("SVM")
    model = SVM("SVM", data.get_embeddings(), data.get_type())
    model.train(data)
    model.predict(data.X_test)
    model.print_results(data)

    print("GNB")
    model = GNB("GNB", data.get_embeddings(), data.get_type())
    model.train(data)
    model.predict(data.X_test)
    model.print_results(data)


