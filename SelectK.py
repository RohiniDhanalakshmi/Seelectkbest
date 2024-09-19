import pandas as pd 
import numpy as np
data=pd.read_csv("CKD.csv")
data=pd.get_dummies(data,dtype=int,drop_first=True)
indep=data[['age', 'bp', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hrmo', 'pcv','wc', 'rc', 'sg_b', 'sg_c', 'sg_d', 'sg_e', 
            'rbc_normal', 'pc_normal', 'pcc_present', 'ba_present', 'htn_yes', 'dm_yes', 'cad_yes','appet_yes', 'pe_yes', 'ane_yes']]
dep=data[['classification_yes']]
def selectkbest(indep,dep,n):
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    from sklearn.feature_selection import RFE
    test=SelectKBest(score_func=chi2,k=n)
    fit1=test.fit(indep,dep)
    selectk_feature=fit1.transform(indep)
    return selectk_feature
def split_scaler(indep,dep):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    x_train,x_test,y_train,y_test=train_test_split(indep,dep,test_size=0.30,random_state=0)
    sc=StandardScaler()
    x_train=sc.fit_transform(x_train)
    x_test=sc.transform(x_test)
    return x_train,x_test,y_train,y_test
def cm_prediction(classifier,x_test):
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import classification_report
    y_pred=classifier.predict(x_test)
    cm=confusion_matrix(y_test,y_pred)
    accuracy=accuracy_score(y_test,y_pred)
    report=classification_report(y_test,y_pred)
    return classifier,accuracy,report,x_test,y_test,cm
def logistic(x_train,y_train,x_test):
    from sklearn.linear_model import LogisticRegression
    classifier=LogisticRegression(random_state=0)
    classifier.fit(x_train,y_train)
    classifier,accuracy,report,x_test,y_test,cm=cm_prediction(classifier,x_test)
    return classifier,accuracy,report,x_test,y_test,cm
def svm_linear(x_train,y_train,x_test):
    from sklearn.svm import SVC
    classifier=SVC(kernel='linear',random_state=0)
    classifier.fit(x_train,y_train)
    classifier,accuracy,report,x_test,y_test,cm=cm_prediction(classifier,x_test)
    return classifier,accuracy,report,x_test,y_test,cm
def svm_NL(x_train,y_train,x_test):
    from sklearn.svm import SVC
    classifier=SVC(kernel='rbf',random_state=0)
    classifier.fit(x_train,y_train)
    classifier,accuracy,report,x_test,y_test,cm=cm_prediction(classifier,x_test)
    return classifier,accuracy,report,x_test,y_test,cm
def Navie(x_train,y_train,x_test):
    from sklearn.naive_bayes import GaussianNB
    classifier=GaussianNB()
    classifier.fit(x_train,y_train)
    classifier,accuracy,report,x_test,y_test,cm=cm_prediction(classifier,x_test)
    return classifier,accuracy,report,x_test,y_test,cm
def Knn(x_train,y_train,x_test):
    from sklearn.neighbors import KNeighborsClassifier
    classifier=KNeighborsClassifier()
    classifier.fit(x_train,y_train)
    classifier,accuracy,report,x_test,y_test,cm=cm_prediction(classifier,x_test)
    return classifier,accuracy,report,x_test,y_test,cm
def Decision(x_train,y_train,x_test):
    from sklearn.tree import DecisionTreeClassifier
    classifier=DecisionTreeClassifier()
    classifier.fit(x_train,y_train)
    classifier,accuracy,report,x_test,y_test,cm=cm_prediction(classifier,x_test)
    return classifier,accuracy,report,x_test,y_test,cm
def Random(x_train,y_train,x_test):
    from sklearn.ensemble import RandomForestClassifier
    classifier=RandomForestClassifier()
    classifier.fit(x_train,y_train)
    classifier,accuracy,report,x_test,y_test,cm=cm_prediction(classifier,x_test)
    return classifier,accuracy,report,x_test,y_test,cm
def selectk_classification(acclog,accsvml,accsvmnl,accknn,accnav,accdes,accrf):
    dataframe=pd.DataFrame(index=['chisquare'],columns=['logistic','Svml','Svmnl','Knn','Navie','Decision','Random'])
    for number,idex in enumerate(dataframe.index):
        dataframe['logistic'][idex]=acclog[number]
        dataframe['Svml'][idex]=accsvml[number]
        dataframe['Svmnl'][idex]=accsvmnl[number]
        dataframe['Knn'][idex]=accknn[number]
        dataframe['Navie'][idex]=accnav[number]
        dataframe['Decision'][idex]=accdes[number]
        dataframe['Random'][idex]=accrf[number]
        return dataframe
kbest=selectkbest(indep,dep,5)
acclog=[]
accsvml=[]
accsvmnl=[]
accknn=[]
accnav=[]
accdes=[]
accrf=[]
x_train,x_test,y_train,y_test=split_scaler(kbest,dep)
classifier,accuracy,report,x_test,y_test,cm=logistic(x_train,y_train,x_test)
acclog.append(accuracy)
classifier,accuracy,report,x_test,y_test,cm=svm_linear(x_train,y_train,x_test)
accsvml.append(accuracy)
classifier,accuracy,report,x_test,y_test,cm=svm_NL(x_train,y_train,x_test)
accsvmnl.append(accuracy)
classifier,accuracy,report,x_test,y_test,cm=Knn(x_train,y_train,x_test)
accknn.append(accuracy)
classifier,accuracy,report,x_test,y_test,cm=Navie(x_train,y_train,x_test)
accnav.append(accuracy)
classifier,accuracy,report,x_test,y_test,cm=Decision(x_train,y_train,x_test)
accdes.append(accuracy)
classifier,accuracy,report,x_test,y_test,cm=Random(x_train,y_train,x_test)
accrf.append(accuracy)
selectk_classification(acclog,accsvml,accsvmnl,accknn,accnav,accdes,accrf)