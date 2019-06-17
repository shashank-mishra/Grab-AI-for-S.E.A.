#Libraries for initial data loading and preprocessing
import pandas as pd
import geohash2 as gh
import time
from sklearn.preprocessing import StandardScaler

#libraries for linear regression data
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

#libraries for Random forest regression data
from sklearn.ensemble import RandomForestRegressor

#libraries for gradient boost regression data
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics 


#Start time to measure how much time our program took to run
start_time=time.time()

#Reading our csv file as input to ouer program
data=pd.read_csv("traffic_management_train.csv")

timestamp_seconds=[]
length_dataset=len(data['timestamp']) #Length of our dataset

#Here I am doing data wrangling and converting string format data to int format so that it can be used in our model
for i in range(length_dataset):
    if(len(data['timestamp'][i])==5):   
        hour=int(data['timestamp'][i][0:2])
        min=int(data['timestamp'][i][3:5])
    elif(len(data['timestamp'][i])==4):
        if(data['timestamp'][i][2]==':'):
            hour=int(data['timestamp'][i][0:2])
            min=int(data['timestamp'][i][3])
        elif(data['timestamp'][i][1]==':'):
            hour=int(data['timestamp'][i][0])
            min=int(data['timestamp'][i][2:4])
    elif(len(data['timestamp'][i])==3):
        hour=int(data['timestamp'][i][0])
        min=int(data['timestamp'][i][2])
    
    time_seconds=(hour*3600)+(min*360)
    timestamp_seconds.append(time_seconds) 

#Adding new coloumn -timestamp_seconds- that can be used in our model easily
data['Timestamp_seconds']= timestamp_seconds
#dropping timestamp string as its of no use
data=data.drop(['timestamp'],axis=1) 

#sorted value according to name of place(i.e geohash6 value)
new_data= data.sort_values(['geohash6'], ascending=True, na_position='first') 
       
#Checking wheather any cell is null or not                        
new_data.isnull().sum()     
 #If any cell is Null then inserting something in it                                                             
new_data.dropna(inplace=True)                                                

#This list contains name of all places present in our dataset
name_of_place=list(new_data['geohash6'].unique()) 
#This is to calculate number of unique place present in given dataset(currently 1329)
num_of_place=len(new_data['geohash6'].unique())   
        
y=new_data['geohash6'].value_counts() 
#This gives count of frequency of particular place in list format
x=new_data['geohash6'].value_counts() 
#converting list to dataframe just for easy access            
x=pd.DataFrame(x)
#sorting data according to their name so that any place can be accessed easily
x=x.sort_index()

#This model function will contain  algorithm used for training and testing data(currently)
def model_lr(dataframe_place):#taking dataframe of place as input and Time taken=420-460 seconds
    X=dataframe_place.drop(['geohash6','demand'],axis=1)
    Y=dataframe_place.demand   
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    
    """linear regression"""
 
    reg = LinearRegression()
    reg.fit(X_train,Y_train)
    #prediction will store the predicted values corresponding to test values
    prediction = reg.predict(X_test)
    #accuracy will tell the accuracy of linear regression to predict output for corresping test values
    accuracy=r2_score(Y_test,prediction)       
    #function is returning accuracy as output
    return (accuracy)

def model_rf(dataframe_place):
    X=dataframe_place.drop(['geohash6','demand'],axis=1)
    Y=dataframe_place.demand   
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    
    """  Random Forest  """
 
    rf = RandomForestRegressor()
    rf.fit(X_train,Y_train)
    #prediction will store the predicted values corresponding to test values
    prediction = rf.predict(X_test)
    #accuracy will tell the accuracy of linear regression to predict output for corresping test values
    accuracy=rf.score(X_test,Y_test)         
    #function is returning accuracy as output
    return (accuracy)

def model_gb(dataframe_place):#taking dataframe of place as input and Time taken=420-460 seconds
    X=dataframe_place.drop(['geohash6','demand'],axis=1)
    Y=dataframe_place.demand   
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    
    """Gradient Boosting Regression"""
 
    reg = GradientBoostingRegressor()
    reg.fit(X_train,Y_train)
    #prediction will store the predicted values corresponding to test values
    prediction = reg.predict(X_test)
    #accuracy will tell the accuracy of linear regression to predict output for corresping test values
    accuracy=reg.score(X_test,Y_test)       
    #function is returning accuracy as output
    return (accuracy)


place_arr=[]
accuracy_arr=[]
start=0
end=int(x.values[0])
count=0
place_insufficient_data=[]

print("---------------------------------------------------------------")
#Used for loop to pass dataframe of each place and calculating accuracy for particular prediction, This loop also returns number of cities with insufficient data in it
for j in range(1,num_of_place):
    df_place=new_data.iloc[start:end,:]
    if(len(df_place['demand'])<1500):  #Setting threshhold of datapoint required for prediction to 10
        place_insufficient_data.append(df_place.iloc[0][0])
        count+=1
        start=end
        end=end+int(x.values[j])
        continue
    accuracy_model=model_gb(df_place)  #calling model here
    place_arr.append(list(df_place['geohash6'])[0])
    accuracy_arr.append(accuracy_model)
    start=end
    end=end+int(x.values[j])
print("---------------------------------------------------------------")

print("These cities have less than 1500 datapoint please enter sufficient number of data point to get correct prediction\n !!",place_insufficient_data,"\n\nThank you 🙂 !!")
print("\nTotal number of cities with insufficient data are: ",count,"\n")

print("---------------------------------------------------------------")

#Creating list to store latitude and longitude of particular geahash value
geohash_lat=[]
geohash_lon=[]

#this loop decodes value of geahash6 coloumn ad returns latitude and longitude which will be stored in a list
for j in range(len(place_arr)):
    geohash_lat.append(float(gh.decode(place_arr[j])[0]))
    geohash_lon.append(float(gh.decode(place_arr[j])[1]))

#Finally making dataframe which will contain name of place, its latitude and Longitude, and accuracy of model prediction for particular place    
list_of_tuples2 = list(zip(place_arr, geohash_lat, geohash_lon, accuracy_arr))
df = pd.DataFrame(list_of_tuples2, columns = ['Place','Place_latitude','Place_longitude','Accuracy'])    
df.to_csv("Neural_Network_model.csv")

#End time to measure how much time our program took to run
end_time=time.time()

print("The places in data with their respective latitude and longitude can be predicted with accuracy as given below:\n\n\n",df)
print("\n Total time taken to execute program is:", end_time-start_time, "seconds")

