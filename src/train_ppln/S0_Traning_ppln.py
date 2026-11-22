from src.train_ppln.s1_DataCleaning import Data_Cleaning
from src.train_ppln.s2_Tranform import Transform
from src.train_ppln.s3_Act_Training import Trainning
import joblib

if __name__ == "__main__":
    #obj_cleaning = Data_Cleaning
    #df = obj_cleaning.readFile()
    #df = obj_cleaning.Handel_Missing(df)
    #df = obj_cleaning.Handle_Duplicates(df)
    #df = obj_cleaning.Replace_Uneve(df)
    #df = obj_cleaning.convertDur_Min(df)
    #df = obj_cleaning.DateTimeConverion(df)
    #df = obj_cleaning.uncessaryCol(df)
    #df = obj_cleaning.saveCleanedData(df)

   # Obj_Trans = Transform()
    #df = Obj_Trans.readData()
   # X_Train,X_Test,Y_Train_True,Y_Test_True = Obj_Trans.Train_Test_Split(df)
   # trans_obj = Obj_Trans.transform_Data()
   # X_Train_Trans = trans_obj.fit_transform(X_Train)
   # X_Train_Trans.to_csv("Airtifacts/Trans_Data/X_train_trans.csv",index = False)
   # joblib.dump(trans_obj,"Airtifacts/pkl_TransFile/trans_Obj.pkl")
   # X_Test_Trans = trans_obj.transform(X_Test)
   # X_Test_Trans.to_csv("Airtifacts/Trans_Data/X_test_trans.csv",index=False)

    Obj_Train = Trainning()
    X_test, X_train, X_train_trans, X_test_trans, y_train_true, y_test_true = Obj_Train.readData()
    Obj_Train.Model_Training(X_train_trans,X_test_trans,X_train,X_test,y_train_true, y_test_true)
