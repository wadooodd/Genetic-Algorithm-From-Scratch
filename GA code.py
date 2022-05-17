#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from copy import deepcopy 


# In[2]:


import warnings
warnings.filterwarnings("ignore")


# In[3]:


import random


# In[4]:


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[5]:


df=pd.read_excel('Training_Data.xlsx',inplace=True)
df


# In[6]:


df.head().T


# In[7]:


df.isnull().sum()


# In[8]:


df.dtypes


# # Data Preprocessing

# In[9]:


def Preprocessing(df):
    df['Race']=df['Race'].fillna('White')
    df['ChronicKidneyDisease']=df['ChronicKidneyDisease'].fillna('UnKnown')
    df['DiabetesMellitus']=df['DiabetesMellitus'].fillna('UnKnown')
    df['Anemia']=df['Anemia'].fillna('UnKnown')
    df['ChronicObstructivePulmonaryDisease']=df['ChronicObstructivePulmonaryDisease'].fillna('UnKnown')
    df['Depression ']=df['Depression '].fillna('UnKnown')
    df['EncounterId'] = df['EncounterId'].astype(str)
    #removing PH string from encounterid
    df['EncounterId'] = df['EncounterId'].map(lambda rmv: rmv.lstrip('PH'))
    df['EncounterId'] = df['EncounterId'].map(lambda rmv: rmv.lstrip('W'))
    df['EncounterId'] = df['EncounterId'].map(lambda rmv: rmv.lstrip('V'))
    df['EncounterId'] = df['EncounterId'].map(lambda rmv: rmv.lstrip('D'))


# In[10]:


Preprocessing(df)


# In[11]:


df


# # Creating Population

# In[12]:


df.columns


# In[13]:


df['ReadmissionWithin_90Days'].unique


# In[14]:


population=[]


# In[15]:


cols=df.columns.to_list()
cols.remove("ReadmissionWithin_90Days")
cols


# In[16]:


#def Create_population():
#    cols=df.columns.to_list()
#    for i in range(0,40):
#        listt=list()
#        while(True):
#            listt=random.sample(cols,30)
#            if "ReadmissionWithin_90Days" in listt:
#                if len(np.unique(listt))== 30:
#                    if listt not in population:
#                        print(listt)
#                        break
#            population.append(listt)


# In[17]:


def Create_population():
    for i in range(0,40):
        listt=list()
        for j in range(0,30):
            listt=random.sample(cols,30)
            if listt not in population:
                print(listt)
                break
        population.append(listt)


# In[18]:


Create_population()


# # Fitness Function

# In[19]:


#creating temporary dataframe
tempdf=pd.read_excel('Training_Data.xlsx',inplace=True)
tempdf


# In[20]:


Preprocessing(tempdf)


# In[21]:


tempdf.head()


# In[22]:


labelencoder=LabelEncoder()


# In[23]:


tempdf['DischargeDisposision']=labelencoder.fit_transform(tempdf['DischargeDisposision'])
tempdf['Gender']=labelencoder.fit_transform(tempdf['Gender'])
tempdf['Race']=labelencoder.fit_transform(tempdf['Race'])
tempdf['DiabetesMellitus']=labelencoder.fit_transform(tempdf['DiabetesMellitus'])
tempdf['ChronicKidneyDisease']=labelencoder.fit_transform(tempdf['ChronicKidneyDisease'])
tempdf['Anemia']=labelencoder.fit_transform(tempdf['Anemia'])
tempdf['Depression ']=labelencoder.fit_transform(tempdf['Depression '])
tempdf['ChronicObstructivePulmonaryDisease']=labelencoder.fit_transform(tempdf['ChronicObstructivePulmonaryDisease'])
tempdf['ReadmissionWithin_90Days']=labelencoder.fit_transform(tempdf['ReadmissionWithin_90Days'])
tempdf.head()


# In[24]:


tempdf[population[0]]


# In[25]:


fitness_score=[]


# In[26]:


lg = LogisticRegression()


# In[27]:


def Fitness_function(col):
    #cols=population[i]
    #print(tempdf[col])
    X=tempdf[col]
    Y=tempdf['ReadmissionWithin_90Days']
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                     Y, test_size=0.25, 
                                                     random_state=1)
    lg.fit(X_train,y_train)
    pred = lg.predict(X_test)
    print("Accuracy = "+ str(accuracy_score(y_test,pred)*100))
    fitness_score.append(accuracy_score(y_test,pred)*100)
    #    print(fitness_score)
    


# In[28]:


for i in population:
    Fitness_function(i)
    


# In[29]:


fitness_score


# In[30]:


print(population[1])


# In[31]:


tempdf[population[1]].head()


# # Selection

# ## Roulette's Wheel

# In[32]:


import numpy.random as npr
def selection( fit):
    max = sum([c for c in fit])
    selection_probs = [c/max for c in fit]
    return fit.index(fit[npr.choice(len(fit), p=selection_probs)])


# fitness_copy=deepcopy(fitness_score)
# parent1=selection(fitness_copy)
# parent2=selection(fitness_copy)
# while parent1==parent2:
#     parent2=selection(fitness_score)
#     
# print(parent1)
# print(parent2)

# In[33]:


def parentselect():
    fitness_copy=deepcopy(fitness_score)
    global parent1
    global parent2
    parent1=selection(fitness_copy)
    parent2=selection(fitness_copy)
    while parent1==parent2:
        parent2=selection(fitness_score)
    print(population[parent1])
    print(population[parent2])
    return parent1,parent2
    

    
    


# In[34]:


parentselect()


# In[35]:


print(parent1)
print(parent2)


# # Crossover

# In[36]:


p1=population[parent1]
p2=population[parent2]


# In[37]:


p1


# In[38]:


p2


# In[39]:


def crossover(par1,par2):
    #CHROMOSOMES
    #chromosomes=Create_population(deepcopy(population))   
    global child1
    global child2
    #CROSSOVER
    random_slice_index=np.random.randint(0,25)
    child1 = population[parent1][0:random_slice_index]+ population[parent2][random_slice_index:]
    child2 = population[parent2][0:random_slice_index]+ population[parent1][random_slice_index:]
    return child1,child2
    print(child1)
    print(child2)


# In[40]:


#Print childs result of crossover
crossover(parent1,parent2)


# # Mutation

# In[41]:


#MUTATION
mutation_prob=np.random.randint(0,30)
mutation_threshold=80
if mutation_prob<=mutation_threshold:
    random_index=np.random.randint(0,30)
    col=tempdf.columns.to_list()
    #random1=random.sample(col,1)[0]
    #random2=random.sample(col,1)[0]
    child1[random_index]=random.sample(col,1)[0]
    child2[random_index]=random.sample(col,1)[0]
    print(child2)


# In[42]:


def mutation(c1,c2):                                        
    mutation_prob=np.random.randint(0,100)
    mutation_threshold=80
    if mutation_prob<=mutation_threshold:
        random_index=np.random.randint(0,30)
        c1_ri=np.random.randint(0,57)                  
        c1_rc=df.columns[c1_ri]
        while True:
            if c1_rc not in c1:
                c1_rc=df.columns[c1_ri]
                c1[random_index]=c1_rc
                print(c1)
                break
            else:
                c1_ri=np.random.randint(0,57)                  
                c1_rc=df.columns[c1_ri]
                #print(c1_rc)
                
                
        c2_ri=np.random.randint(0,57)
        c2_rc=df.columns[c2_ri]
        c2[random_index]=c2_rc
        while True:
            if c2_rc not in c2:
                c2_rc=df.columns[c2_ri]
                c2[random_index]=c2_rc
                print(c2)
                break
            else:
                c2_ri=np.random.randint(0,57)    
                c2_rc=df.columns[c2_ri]
                #print(c2_rc)
        
    
    child1_fitness=Fitness_function(c1)
    child2_fitness=Fitness_function(c2)

    return(child1_fitness,child2_fitness)
   


# In[43]:


mutation(child1,child2)


# In[44]:


indx1=17 


# #Doing interchanging in mutation
# def mutation(parent1,parent2):
#     n = len(parent1)
#     pos_1 = random.randint(0,n-1)
#     pos_2 = random.randint(0,n-1)
#     #print(pos_1, pos_2)
#     def swap(sol, posA, posB):
#         result = sol.copy()
#         elA = sol[posA]
#         elB = sol[posB]
#         result[posA] = elB
#         result[posB] = elA
#         return result
#     global child1_m
#     global child2_m
#     child1_m = swap(parent1, pos_1, pos_2)
#     child2_m = swap(parent2, pos_1, pos_2)
#     return child1_m,child2_m

# # Final Result

# ## Accuracy With All Features

# In[45]:


def Fitness_function1():
    #cols=population[i]
    X=tempdf.drop('ReadmissionWithin_90Days',1)
    Y=tempdf['ReadmissionWithin_90Days']
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                     Y, test_size=0.25, 
                                                     random_state=1)
    lg.fit(X_train,y_train)
    pred = lg.predict(X_test)
    print("Accuracy = "+ str(accuracy_score(y_test,pred)*100))
    return accuracy_score(y_test,pred)*100
    #fitness_score.append(accuracy_score(y_test,pred)*100)
    #    print(fitness_score)
    


# In[46]:


without = Fitness_function1()


# ## Accuracy After Applying GA

# In[47]:


epoch=100
while(max(fitness_score) != 74 and epoch!=0):
    #Selection of Parent's using roulette's wheel
    #parentselect()
    parent1=selection(fitness_score)
    parent2=selection(fitness_score)
    while parent1==parent2:
        parent2=selection(fitness_score)
    #Crossover
    crossover(parent1,parent2)
    while True:
        child1,child2=crossover(parent1,parent2)
        
        if len(np.unique(child1))==30 and len(np.unique(child2))==30:
            break
    #mutation
    #mutation(child1,child2)
    child1_fitness,child2_fitness=mutation(child1,child2)
    #Creating new dataframes with selected columns
    #newdf1=pd.DataFrame(df,columns=child1_m)
    #newdf2=pd.DataFrame(df,columns=child2_m)
    #Fitness score
    #score1=Fitness_function(child1_m)
    #score2=Fitness_function(child2_m)
    
    epoch=epoch-1
  
index=fitness_score.index(max(fitness_score))
print("Accuracy after running GA : ",max(fitness_score))

    
    


# In[48]:


print("Features Selected by GA: ")
print(population[index]) 


# In[49]:


import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.set_ylim(60,80)
ax.set_title("Comparison between results")
label = ['without feature', 'with feature scaling']
data = [without,max(fitness_score)]
ax.bar(label,data, color = 'red', width = 0.5)
plt.show()


# In[ ]:




