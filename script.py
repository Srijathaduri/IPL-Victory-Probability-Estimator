import numpy as np
import pandas as pd
match_df = pd.read_csv("matches.csv")
delivery_df=pd.read_csv("deliveries.csv") 
match_df.drop(["toss_winner","toss_decision","player_of_match","umpire1","umpire2","umpire3"],axis=1, inplace=True)
delivery_df.drop(["bowler","is_super_over","dismissal_kind","fielder","wide_runs","bye_runs", "legbye_runs", "noball_runs", "penalty_runs"],axis=1, inplace=True)
total_score_df=(delivery_df.groupby(['match_id','inning']).sum()['total_runs'].reset_index())


total_score_df=total_score_df[total_score_df['inning']==1]
match_df=(match_df.merge(total_score_df[['match_id','total_runs']],left_on="id",right_on='match_id'))

teams=list(match_df['team1'].unique())
teams=['Sunrisers Hyderabad', 'Mumbai Indians', 'Gujarat Titans','Royal Challengers Bangalore', 'Kolkata Knight Riders','Kings XI Punjab', 'Chennai Super Kings', 'Rajasthan Royals','Delhi Capitals']


match_df["team1"]=match_df["team1"].str.replace('Delhi Daredevils','Delhi Capitals')
match_df['team2']=match_df['team2'].str.replace('Delhi Daredevils','Delhi Capitals')

match_df["team1"]=match_df["team1"].str.replace('Deccan Chargers','Sunrisers Hyderabad')
match_df['team2']=match_df['team2'].str.replace('Deccan Chargers','Sunrisers Hyderabad')

match_df["team1"]=match_df["team1"].str.replace('Gujarat Lions','Gujarat Titans')
match_df['team2']=match_df['team2'].str.replace('Gujarat Lions','Gujarat Titans')

match_df=match_df[match_df['team1'].isin(teams)]
match_df=match_df[match_df['team2'].isin(teams)]

#removing dl_applied=1 records or rows
match_df=match_df[match_df['dl_applied']==0]
match_df=match_df[['match_id','city','winner','total_runs']]


#delivary dataset
delivery_df=delivery_df[delivery_df['inning']==2]

delivery_df=match_df.merge(delivery_df,on='match_id')
delivery_df.rename(columns={"total_runs_x":"total_runs","total_runs_y":"Ball_score"}, inplace=True)

delivery_df["Score"]=delivery_df[["match_id","Ball_score"]].groupby("match_id").cumsum()["Ball_score"]

#new column which has traget
delivery_df["target_left"]=(delivery_df["total_runs"]+1)-delivery_df['Score']
print(delivery_df)

#for 1 over 6 balls  and we need see how many ball are remaind 
delivery_df["Remaining_balls"]=(120-((delivery_df["over"]-1)*6 +delivery_df["ball"]))

#how many wicktes leftout
delivery_df["player_dismissed"]=delivery_df["player_dismissed"].fillna("0") 

delivery_df["player_dismissed"]=delivery_df["player_dismissed"].apply(lambda x:x if x=='0' else "1").astype('int64')

delivery_df["Wickets"]=delivery_df[["match_id","player_dismissed"]].groupby("match_id").cumsum()["player_dismissed"].values
delivery_df["Wickets"]=10-delivery_df["Wickets"]

#current_rate_run (crr)=runs scored / No of Overs completed
delivery_df["crr"]=(delivery_df['Score'])*6/(120-delivery_df['Remaining_balls'])

#required run rate =Remaindind score or target/No of overs letf
delivery_df["rrr"]=(delivery_df['target_left'])*6/(delivery_df['Remaining_balls'])
print(delivery_df)
#creating column as result where 1 means won and 0 means loss
def result(row):
    if row['batting_team']==row["winner"]:
        return 1 
    else:
        return 0 
delivery_df["result"]=delivery_df.apply(result,axis=1) 
Model_df=delivery_df[['batting_team','bowling_team','city','Score','Wickets','Remaining_balls','target_left','crr', 'rrr', 'result']]



Model_df[Model_df['city'].isna()]
print(Model_df[Model_df['city'].isna()])


Model_df.dropna(inplace=True)
print(Model_df.isnull().sum())
Model_df=Model_df[Model_df['Remaining_balls']!=0]
print(Model_df.describe())
Model_df=Model_df.sample(Model_df.shape[0])
print(Model_df.sample())

#SPLITTING THE DATA INTO TARGETS AND FEATURES 
X=Model_df.iloc[:,:-1]
Y=Model_df.iloc[:,-1]
print(X)
print(Y)

#SPLITTING THE DATA to train and test 
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test =train_test_split(X,Y,test_size=0.2,random_state=2)
print(X_train)
print(X_test)
print(Y_train)
print(Y_test)

#lableing 
#one hot encoder #transfer non numerical to numerical
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import OneHotEncoder

trf = ColumnTransformer([
    ('trf', OneHotEncoder(sparse_output=False, drop='first'), ['batting_team', 'bowling_team', 'city'])
], remainder='passthrough')

print(trf)
#building the model 
from sklearn.linear_model import LogisticRegression 
from sklearn.pipeline import Pipeline
pipe=Pipeline(steps=[
    ("step1",trf),("step2",LogisticRegression(solver='liblinear'))
])
pipe.fit(X_train,Y_train)
Y_prediction=pipe.predict(X_test)

#Accuracy score 
from sklearn.metrics import accuracy_score
accuracy_score(Y_test,Y_prediction)


#exporting model 
import pickle as pkl 
pkl.dump(pipe,open('model.pkl','wb'))
pkl.dump(teams,open('team.pkl','wb'))
cities=list(Model_df['city'].unique())
pkl.dump(cities,open('city.pkl','wb'))
print(Model_df.columns)