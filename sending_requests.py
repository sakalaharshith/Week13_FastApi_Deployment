import json
import requests
import pandas as pd
import random
url="http://35.179.88.8/predict"
# just assigning some random values to input data dictionary
input_data={
  "Ntm_Speciality": 2,
  "Dexa_During_Rx": 1,
  "Frag_Frac_During_Rx": 1,
  "Tscore_Bucket_During_Rx": 1,
  "Comorb_Encounter_For_Screening_For_Malignant_Neoplasms": 0,
  "Comorb_Encounter_For_Immunization": 1,
  "Comorb_Encntr_For_General_Exam_W_O_Complaint_Susp_Or_Reprtd_Dx": 1,
  "Comorb_Other_Joint_Disorder_Not_Elsewhere_Classified": 0,
  "Comorb_Long_Term_Current_Drug_Therapy": 1,
  "Comorb_Dorsalgia": 1,
  "Comorb_Personal_History_Of_Other_Diseases_And_Conditions": 1,
  "Comorb_Other_Disorders_Of_Bone_Density_And_Structure": 0,
  "Comorb_Osteoporosis_without_current_pathological_fracture": 0,
  "Comorb_Gastro_esophageal_reflux_disease": 0,
  "Concom_Systemic_Corticosteroids_Plain": 1,
  "Concom_Cephalosporins": 0,
  "Concom_Macrolides_And_Similar_Types": 1,
  "Concom_Broad_Spectrum_Penicillins": 1,
  "Concom_Anaesthetics_General": 0,
  "Concom_Viral_Vaccines": 0
}
# Reading the label encoded data file 
data=pd.read_csv('encoded.csv')
# Picking a random data point or record from dataset and sending a post request to model api which is deployed in ec2 instance.
random_number=random.randint(0,len(data))
single_piece_of_data=data.iloc[random_number,:]
print(single_piece_of_data)
count=0
for keys in input_data.keys():
    input_data[keys]=int(single_piece_of_data[count])
    count=count+1
print(input_data)
json_data=json.dumps(input_data)

# Response is being sent by api and is collected here as the response contains class output
response=requests.post(url,data=json_data,verify=False)
category=""
# printing expected output and actual output just to check how well the model is predicting.
print(" ")
print("the predicted output is",response.text)
if single_piece_of_data['Persistency_Flag']==0:
    category='Non-Persistent'
if single_piece_of_data['Persistency_Flag']==1:
        category='Persistent'  

print("actual output is :",category)