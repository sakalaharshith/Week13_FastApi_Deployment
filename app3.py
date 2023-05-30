import uvicorn
from fastapi import FastAPI
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl
from fastapi.middleware.cors import CORSMiddleware

app=FastAPI()
origins=['*']
app.add_middleware(
CORSMiddleware, 
allow_origins=origins,
allow_credentials=True,
allow_methods=['*'],
allow_headers=['*']
)
model=pkl.load(open('random_forest_model.pkl','rb'))
from pydantic import BaseModel
class persistency_model(BaseModel):
    Ntm_Speciality:int
    Dexa_During_Rx:int
    Frag_Frac_During_Rx:int
    Tscore_Bucket_During_Rx:int
    Comorb_Encounter_For_Screening_For_Malignant_Neoplasms:int
    Comorb_Encounter_For_Immunization:int
    Comorb_Encntr_For_General_Exam_W_O_Complaint_Susp_Or_Reprtd_Dx:int
    Comorb_Other_Joint_Disorder_Not_Elsewhere_Classified:int
    Comorb_Long_Term_Current_Drug_Therapy:int 
    Comorb_Dorsalgia:int
    Comorb_Personal_History_Of_Other_Diseases_And_Conditions:int
    Comorb_Other_Disorders_Of_Bone_Density_And_Structure:int
    Comorb_Osteoporosis_without_current_pathological_fracture:int
    Comorb_Gastro_esophageal_reflux_disease:int
    Concom_Systemic_Corticosteroids_Plain:int
    Concom_Cephalosporins:int
    Concom_Macrolides_And_Similar_Types:int
    Concom_Broad_Spectrum_Penicillins:int
    Concom_Anaesthetics_General:int
    Concom_Viral_Vaccines:int

@app.get('/')
def welcome():
    return {"message":"You are clear enough to send data for output prediction"}

@app.post('/predict')
def predict_output(data:persistency_model):
    data=data.dict()
    print(data)
    print("Hello")
    Ntm_Speciality=data['Ntm_Speciality']
    Dexa_During_Rx=data['Dexa_During_Rx']
    Frag_Frac_During_Rx=data['Frag_Frac_During_Rx']
    Tscore_Bucket_During_Rx=data['Tscore_Bucket_During_Rx']
    Comorb_Encounter_For_Screening_For_Malignant_Neoplasms=data['Comorb_Encounter_For_Screening_For_Malignant_Neoplasms']
    Comorb_Encounter_For_Immunization=data['Comorb_Encounter_For_Immunization']
    Comorb_Encntr_For_General_Exam_W_O_Complaint_Susp_Or_Reprtd_Dx=data['Comorb_Encntr_For_General_Exam_W_O_Complaint_Susp_Or_Reprtd_Dx']
    Comorb_Other_Joint_Disorder_Not_Elsewhere_Classified=data['Comorb_Other_Joint_Disorder_Not_Elsewhere_Classified']
    Comorb_Long_Term_Current_Drug_Therapy=data['Comorb_Long_Term_Current_Drug_Therapy'] 
    Comorb_Dorsalgia=data['Comorb_Dorsalgia']
    Comorb_Personal_History_Of_Other_Diseases_And_Conditions=data['Comorb_Personal_History_Of_Other_Diseases_And_Conditions']
    Comorb_Other_Disorders_Of_Bone_Density_And_Structure=data['Comorb_Other_Disorders_Of_Bone_Density_And_Structure']
    Comorb_Osteoporosis_without_current_pathological_fracture=data['Comorb_Osteoporosis_without_current_pathological_fracture']
    Comorb_Gastro_esophageal_reflux_disease=data['Comorb_Gastro_esophageal_reflux_disease']
    Concom_Systemic_Corticosteroids_Plain=data['Concom_Systemic_Corticosteroids_Plain'] 
    Concom_Cephalosporins=data['Concom_Cephalosporins']
    Concom_Macrolides_And_Similar_Types=data['Concom_Macrolides_And_Similar_Types']
    Concom_Broad_Spectrum_Penicillins=data['Concom_Broad_Spectrum_Penicillins']
    Concom_Anaesthetics_General=data['Concom_Anaesthetics_General'] 
    Concom_Viral_Vaccines=data['Concom_Viral_Vaccines']
    output=model.predict([[Ntm_Speciality, Dexa_During_Rx,Frag_Frac_During_Rx,Tscore_Bucket_During_Rx,Comorb_Encounter_For_Screening_For_Malignant_Neoplasms,
    Comorb_Encounter_For_Immunization,
    Comorb_Encntr_For_General_Exam_W_O_Complaint_Susp_Or_Reprtd_Dx,
    Comorb_Other_Joint_Disorder_Not_Elsewhere_Classified,
    Comorb_Long_Term_Current_Drug_Therapy, Comorb_Dorsalgia,
    Comorb_Personal_History_Of_Other_Diseases_And_Conditions,
    Comorb_Other_Disorders_Of_Bone_Density_And_Structure,
    Comorb_Osteoporosis_without_current_pathological_fracture,
    Comorb_Gastro_esophageal_reflux_disease,
    Concom_Systemic_Corticosteroids_Plain, Concom_Cephalosporins,
    Concom_Macrolides_And_Similar_Types, Concom_Broad_Spectrum_Penicillins,
    Concom_Anaesthetics_General, Concom_Viral_Vaccines]])
    print("the output is",output)
    category=""
    if output[0]==0:
        category='Non-Persistent'
    if output[0]==1:
        category='Persistent'    
    return category



from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
def metrics_model(X,y_test,predicted_output,model):
  score=accuracy_score(y_test,predicted_output)
  print('The accuracy of the model is :',score)
  f1_Score=f1_score(y_test,predicted_output)
  print("The F1-Score of the model is: ",f1_Score)
  Precision_Score=precision_score(y_test,predicted_output)
  print('The Precision of the model is :',Precision_Score)
  roc_auc=roc_auc_score(y_test,predicted_output)
  print("The ROC-AUC Score for the model is",roc_auc)
  fpr,tpr,thresholds=roc_curve(y_test,predicted_output)
  plt.plot(fpr,tpr)
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.show()
  print("the cross validation score is",cross_val_score(model,EDA_Features,Label_Feature,cv=5))
  print("The cross validation F1-score is ",cross_val_score(model,EDA_Features,Label_Feature,cv=5,scoring='f1') )
  confusion_matrix1=confusion_matrix(y_test,predicted_output)
  cm_display=ConfusionMatrixDisplay(confusion_matrix=confusion_matrix1,display_labels=['Non-persistent','Persistent'])
  cm_display.plot()
  plt.show()

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
df = pd.read_excel("Healthcare_dataset.xlsx", sheet_name="Dataset")
encoder=LabelEncoder()
df=df[['Ntm_Speciality', 'Dexa_During_Rx','Frag_Frac_During_Rx','Tscore_Bucket_During_Rx','Comorb_Encounter_For_Screening_For_Malignant_Neoplasms',
 'Comorb_Encounter_For_Immunization',
 'Comorb_Encntr_For_General_Exam_W_O_Complaint,_Susp_Or_Reprtd_Dx',
 'Comorb_Other_Joint_Disorder_Not_Elsewhere_Classified',
 'Comorb_Long_Term_Current_Drug_Therapy', 'Comorb_Dorsalgia',
 'Comorb_Personal_History_Of_Other_Diseases_And_Conditions',
 'Comorb_Other_Disorders_Of_Bone_Density_And_Structure',
 'Comorb_Osteoporosis_without_current_pathological_fracture',
 'Comorb_Gastro_esophageal_reflux_disease',
 'Concom_Systemic_Corticosteroids_Plain', 'Concom_Cephalosporins',
 'Concom_Macrolides_And_Similar_Types', 'Concom_Broad_Spectrum_Penicillins',
 'Concom_Anaesthetics_General', 'Concom_Viral_Vaccines','Persistency_Flag']].apply(encoder.fit_transform)
EDA_Features=df[['Ntm_Speciality', 'Dexa_During_Rx','Frag_Frac_During_Rx','Tscore_Bucket_During_Rx','Comorb_Encounter_For_Screening_For_Malignant_Neoplasms',
 'Comorb_Encounter_For_Immunization',
 'Comorb_Encntr_For_General_Exam_W_O_Complaint,_Susp_Or_Reprtd_Dx',
 'Comorb_Other_Joint_Disorder_Not_Elsewhere_Classified',
 'Comorb_Long_Term_Current_Drug_Therapy', 'Comorb_Dorsalgia',
 'Comorb_Personal_History_Of_Other_Diseases_And_Conditions',
 'Comorb_Other_Disorders_Of_Bone_Density_And_Structure',
 'Comorb_Osteoporosis_without_current_pathological_fracture',
 'Comorb_Gastro_esophageal_reflux_disease',
 'Concom_Systemic_Corticosteroids_Plain', 'Concom_Cephalosporins',
 'Concom_Macrolides_And_Similar_Types', 'Concom_Broad_Spectrum_Penicillins',
 'Concom_Anaesthetics_General', 'Concom_Viral_Vaccines']]

Label_Feature=df['Persistency_Flag']
df.rename({'Comorb_Encntr_For_General_Exam_W_O_Complaint,_Susp_Or_Reprtd_Dx':'Comorb_Encntr_For_General_Exam_W_O_Complaint_Susp_Or_Reprtd_Dx'}, axis=1, inplace=True)
print(len(df.columns))
print("The EDA Features are as follows",len(EDA_Features.columns))
df.to_csv('encoded.csv',index=False)
print("yes")

X_train, X_test, y_train, y_test = train_test_split( EDA_Features, Label_Feature, test_size=0.20)
output=model.predict(X_test)
metrics_model(X_test,y_test,output,model)

if '__name__'=='__main__':
    uvicorn.run(app)



