# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 11:41:28 2019

@author: Ying Li

"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.offline as py
from sklearn.preprocessing import StandardScaler # standard scaler
from sklearn.decomposition import PCA # principal component analysis
from matplotlib import rc
from sklearn.cluster import KMeans # k-means clustering
from scipy.cluster.hierarchy import dendrogram, linkage # dendrograms

#Import Data
file = 'finalExam_Mobile_App_Survey_Data.xlsx'
data = pd.read_excel(file)
data.head(10)

####################################
# Code for data analysis
####################################




##############################################################################
##############################################################################
######                      Data Exploration Analysis                 ########
##############################################################################
##############################################################################
#check data info
data.info()#No NaN

#All observations unique?
len(data['caseID'].unique())#all unique

#Any Ternimate questionair?
sum(data['q2r10']==1)#No ternimate 

#
data.describe()

#drop caseID column
data = data.drop(['caseID'], axis=1)

##########
#  Demographical Data
##########
#Plot age
sns.distplot(data['q1']).set(xlim=(1, max(data['q1'])))
plt.title('Age', fontsize = 25)
plt.xlabel('Age')
plt.show()
'''
Under 18 (1)
18-24 (2)
25-29 (3)
30-34 (4)
35-39 (5)
40-44 (6)
45-49 (7)
50-54 (8)
55-59 (9)
60-64 (10)
65 or over (11)


Most interviewers are between ... to ... years old
'''

#Plot gender
'''
Q57: Gender
1 = Male
2 = Female
'''
data['q57'].plot.hist()
plt.title('Male vs Female Ratio', fontsize = 25)
sns.despine()
plt.xlabel('Gender')
plt.xticks(np.array([1,2]) ,['Male','Female'], fontsize = 13)

'''
About even
'''

#Plot education

'''
Q48: Highest level of education
1 = Some high school
2 = High school graduate
3 = Some college
4 = College graduate
5 = Some post-graduate studies 
6 = Post graduate degree 
'''
data['q48'].plot.hist()
plt.title('Education', fontsize = 25)
sns.despine()
plt.xlabel('Education')
plt.xticks(np.array([1,2,3,4,5,6]),
           ['Some high school','High school graduate','Some college',
            'College graduate','Some post-graduate studies','Post graduate degree'], 
           rotation=90, 
           fontsize = 13)
plt.show()

#Plot Marital Status
'''
Q49: Marital Status
1 = Married
2 = Single
3 = Single with a partner
4 - Separated/Widowed/Divorced
'''
data['q49'].plot.hist()
plt.title('Marital Status', fontsize = 25)
sns.despine()
plt.xlabel('Marital Status')
plt.xticks(np.array([1,2,3,4]),
           ['Married','Single','Single with a partner','Separated/Widowed?Divorced'], 
           rotation=90, 
           fontsize = 13)

plt.show()

#Plot Children
'''
Q50: Have children in any of these groups?
1 = No children
2 = Yes, under 6 yrs old
3 = Yes, 6-12 yrs old
4 = Yes, 13-17 yrs old
5 = Yes, 18+ yrs old
'''

children = data.loc[:,'q50r1':'q50r5']

children_plot = children.sum().sort_values(ascending=False)

children_plot.plot(kind='bar')

plt.xlabel('Children')

plt.xticks(np.array([0,1,2,3,4]),
           ['No','<6','6-12','13-17','18+'], 
           rotation=90, 
           fontsize = 13)
plt.show()



#Plot Race
'''             
Q54: What's your race?
1 = White/Caucasian
2 = Black/African American
3 = Asian
4 = Native Hawaiian/Other Pacific Islander
5 = American Indian/Alask Native
6 = Other
'''
data['q54'].plot.hist()
plt.title('Race', fontsize = 25)
sns.despine()
plt.xlabel('Race')

plt.xticks(np.array([1,2,3,4,5,6]),
           ['White/Caucasian',
            'Black/African American',
            'Asian',
            'Native Hawaiian/Other Pacific Islander',
            'American Indian/Alask Native',
            'Other'], 
           rotation=90, 
           fontsize = 13)
plt.show()

'''
Q55: Are you Hispanic/Latino?
1 = Yes
2 = No
'''
data['q55'].plot.hist()
plt.title('Hispanic/Latino', fontsize = 25)
sns.despine()
plt.xlabel('his/lat')
plt.xticks(np.array([1,2]) ,['Yes','No'], fontsize = 13)


#Plot income
'''
Q56: Household annual income b4 taxes
1 = Under $10,000
2 = $10,000-$14,999
3 = $15,000-$19,999
4 = $20,000-$29,999
5 = $30,000-$39,999
6 = $40,000-$49,999
7 = $50,000-$59,999
8 = $60,000-$69,999
9 = $70,000-$79,999
10 = $80,000-$89,999
11 = $90,000-$99,999
12 = $100,000-$124,999
13 = $125,000-$149,999
14 = $150,000 and over 
'''
sns.distplot(data['q56']).set(xlim=(1, max(data['q56'])))
plt.title('Income', fontsize = 25)
plt.xlabel('Income')
plt.show()


##########
#  Device
##########
#sub device dataframe
device = data.loc[:,'q2r1':'q2r9']


'''
iPhone (r1)
iPod touch (r2)
Android (r3)
BlackBerry (r4)
Nokia (r5)
Windows Phone or Windows Mobile (r6)
HP/ Palm WebOS (r7)
Tablet (iPad, Xoom, Galaxy Tab etc.) (r8)
Other smartphone (please specify) (r9)
'''
'''
Drop q2r10 later
'''
#Plot device frequency
device_plot = device.sum().sort_values(ascending=False)

device_plot.plot(kind='bar')
plt.show()

'''
iPhone used most
Android

Trgeting IOS and Android system
'''


##########
#  Apps
##########
#sub app usage dataframe
apps = data.loc[:,'q4r2':'q4r11']

'''
Q4: Use any of these apps?
1.Music and Sound Identification Apps (r1)
2.TV Check-in Apps (r2)
3.Entertainment Apps (i.e. U2) (r3)
4.TV Show Apps (i.e. Glee) ...
5.Gaming Apps
6.Social Networking Apps (i.e. Facebook)
7 General News Apps (i.e. Yahoo! News)
8 Shopping Apps
9 Specific Publication News Apps (i.e. New York Times)
10 Other
11 None (r11)
'''

#Plot app frequency
app_plot = apps.sum().sort_values(ascending=False)

app_plot.plot(kind='bar')
plt.show()

'''
Social network and gaming apps have most usage

'''

#How many apps do you have?


'''
Q11 - How many apps do they have?
1-5 - 1
6-10 - 2
11-30 - 3
31+ - 4
Don’t know - 5
None - 6 (Skip to q13)
'''

data['q11'].plot.hist()
plt.title('How many apps do you have?', fontsize = 25)
sns.despine()
plt.xlabel('apps number')
plt.xticks(np.array([1,2,3,4,5,6]),
           ['1-5','6-10','11-30','31+',"Don't know",'None'], 
           rotation=90, 
           fontsize = 13)

'''
most people have ... apps
'''


#How many apps were free to download?
'''
Q12 - What % were free apps?
1 None of my Apps were free 
2 1% - 25% 
3 26% - 50% 
4 51% - 75%
5 76% - 99%
6 All of my Apps were free 
'''

data['q12'].plot.hist(bins=11)
plt.title('What % were free apps?', fontsize = 25)
plt.xlabel('apps number')
plt.xticks(np.array([1,2,3,4,5,6]),
           ['No free','1%-25%','26%-50%','51%-75%','76%-99%','All free'], 
           rotation=90, 
           fontsize = 13)

'''
Most are free 
'''


##########
#  Website
##########
'''
Q13: How many times per week do you visit these websites?
1 = very often
2 = sometimes
3 = rarely
4 = almost never
                Q13r1 - Facebook
                Q13r2 - Twitter
                Q13r3 - Myspace
                Q13r4 - Pandora Radio
                Q13r5 - Vevo(music)
                Q13r6 - YouTube
                Q13r7 - AOL Radio
                Q13r8 - Last.fm
                Q13r9 - Yahoo Entertainment and Music
                Q13r10 - IMDB
                Q13r11 - LinkedIn
                Q13r12 - Netflix

'''
website = data.copy()
website = website.loc[:,'q13r1':'q13r12']

 
# y-axis in bold
rc('font', weight='bold')
 
# Values of each group
q_data=pd.DataFrame()
for column in website.columns[0:]:
    q_data = q_data.append(website[column].value_counts().sort_index())

veryoften = q_data.loc[:,1]
sometimes = q_data.loc[:,2]
rarely = q_data.loc[:,3]
almostnever = q_data.loc[:,4]

# The position of the bars on the x-axis
r = [0,1,2,3,4,5,6,7,8,9,10,11]
 
# Names of group and bar width
names = ['Facebook','Twitter','Myspace','Pandora Radio','Vevo','YouTube',
         'AolRadio','Last.fm','Yahoo E&M','IMDB','LinkedIn','Netflix']
barWidth = 1
 
# Create almostnever
plt.bar(r, almostnever, color='black', edgecolor='white', width=barWidth)
# Create rarely
plt.bar(r, rarely, bottom=almostnever, color='grey', edgecolor='white', width=barWidth)
# Create sometimes
plt.bar(r, sometimes, bottom=almostnever+rarely, color='yellow', edgecolor='white', width=barWidth)
# Crate veryoften
plt.bar(r, veryoften, bottom=almostnever+rarely+sometimes, color='orange', edgecolor='white', width=barWidth)

        
# Custom X axis
plt.xticks(r, names, fontweight='bold', rotation=90)
plt.xlabel("group")
 
# Show graphic
plt.show()



##########
#  Attitude
##########

#attitude to new things
'''
Q24: How much do you agree/disagree with these statements?
1 = Agree Strongly
2 = Agree
3 = Agree Somewhat
4 = Disagree Somewhat
5 = Disagree
6 = Disagree Strongly
    Technology:
                Q24r1 - I try to keep up with technological developments
                Q24r2 - People often ask my advice when they are looking to 
                buy technology or electronic products
                Q24r3 - I enjoy purchasing new gadgets and appliances
                Q24r4 - I think there is too much technology in everyday life
                Q24r5 - I enjoy using technology to give me more control over 
                my life
                Q24r6 - I look for web tools and Apps that help me save time
        Music:
                Q24r7 - Music is an important part of my life
           TV:
                Q24r8 - I like learning more about my favorite TV shows when 
                I’m not watching them
 Social Media:
                Q24r9 - I think there is too much information out there today
                from the internet and sites like Facebook
                Q24r10 - I’m always checking on friends and family through
                Facebook or other networking websites
                Q24r11 - The internet makes it easier to keep in touch with
                family and friends
                Q24r12 - The internet makes it easy to avoid seeing or having 
                to speak with family and friends

'''

att_new = data.copy()
att_new = att_new.loc[:,'q24r1':'q24r12']


# y-axis in bold
rc('font', weight='bold')
 
# Values of each group
att_new_data=pd.DataFrame()
for column in att_new.columns[0:]:
    att_new_data = att_new_data.append(att_new[column].value_counts().sort_index())

att_new_data=att_new_data.fillna(0)
sa = att_new_data.loc[:,1]
a = att_new_data.loc[:,2]
aso = att_new_data.loc[:,3]
dso = att_new_data.loc[:,4]
d = att_new_data.loc[:,5]
sd = att_new_data.loc[:,6]

# The position of the bars on the x-axis
r = [0,1,2,3,4,5,6,7,8,9,10,11]
 
# Names of group and bar width
names = ['1','2','3','4','5','6','7','8','9','10','11','12']
barWidth = 1
 
# Create strongly agree
plt.bar(r, sa, color='orangered', edgecolor='white', width=barWidth)
# Create agree
plt.bar(r, a, bottom=sa, color='indianred', edgecolor='white', width=barWidth)
# Create agree some
plt.bar(r, aso, bottom=sa+a, color='salmon', edgecolor='white', width=barWidth)
# Crate disagree some
plt.bar(r, dso, bottom=sa+a+aso, color='lightgray', edgecolor='white', width=barWidth)
# Crate disagree 
plt.bar(r, d, bottom=sa+a+aso+dso, color='grey', edgecolor='white', width=barWidth)
# Crate strongly disagree
plt.bar(r, sd, bottom=sa+a+aso+dso+d, color='k', edgecolor='white', width=barWidth)


# Custom X axis
plt.xticks(r, names, fontweight='bold')
plt.xlabel("group")
 
# Show graphic
plt.show()


#Social character
'''
q25. And how much do you agree or disagree with each of the following? 
1 = Agree Strongly
2 = Agree
3 = Agree Somewhat
4 = Disagree Somewhat
5 = Disagree
6 = Disagree Strongly


r1. I consider myself an opinion leader
r2. I like to stand out from others 
r3. I like to offer advice to others 
r4. I like to take the lead in decision making 
r5. I’m the first of my friends and family to try new things 
r6. Responsibility is overrated; I’d rather be told what to do 
r7. I like being in control 
r8. I’m a risk taker 
r9. I think of myself as creative 
r10. I am an optimistic person 
r11. I am very active and always on the go
r12. I always feel stretched for time
'''

cha = data.copy()
cha = cha.loc[:,'q25r1':'q25r12']


# y-axis in bold
rc('font', weight='bold')
 
# Values of each group
cha_data=pd.DataFrame()
for column in cha.columns[0:]:
    cha_data = cha_data.append(cha[column].value_counts().sort_index())

cha_data=cha_data.fillna(0)
sa = cha_data.loc[:,1]
a = cha_data.loc[:,2]
aso = cha_data.loc[:,3]
dso = cha_data.loc[:,4]
d = cha_data.loc[:,5]
sd = cha_data.loc[:,6]

# The position of the bars on the x-axis
r = [0,1,2,3,4,5,6,7,8,9,10,11]
 
# Names of group and bar width
names = ['1','2','3','4','5','6','7','8','9','10','11','12']
barWidth = 1
 
# Create strongly agree
plt.bar(r, sa, color='orangered', edgecolor='white', width=barWidth)
# Create agree
plt.bar(r, a, bottom=sa, color='indianred', edgecolor='white', width=barWidth)
# Create agree some
plt.bar(r, aso, bottom=sa+a, color='salmon', edgecolor='white', width=barWidth)
# Crate disagree some
plt.bar(r, dso, bottom=sa+a+aso, color='lightgray', edgecolor='white', width=barWidth)
# Crate disagree 
plt.bar(r, d, bottom=sa+a+aso+dso, color='grey', edgecolor='white', width=barWidth)
# Crate strongly disagree
plt.bar(r, sd, bottom=sa+a+aso+dso+d, color='k', edgecolor='white', width=barWidth)


# Custom X axis
plt.xticks(r, names, fontweight='bold')
plt.xlabel("group")
 
# Show graphic
plt.show()





#Attitude to purchase
'''
Q26 - How much do you agree/disagree with these statements?
1 = Agree Strongly
2 = Agree
3 = Agree Somewhat
4 = Disagree Somewhat
5 = Disagree
6 = Disagree Strongly
      Shopping:
                r3. I am always on the lookout for a bargain, discounts or 
                deals
                r4. I derive enjoyment from any kind of shopping
                r5. I like package deals because I don’t have to do as much 
                planning
                r6. I’m always shopping online
                r7. I prefer to buy designer brands.
         Apps:
                r8. I can’t get enough Apps
                r9. It’s not the number of Apps you have but how cool they are 
                that really matters
                r10. I love showing off my new Apps to others
                r11. My children have an impact on the Apps I download
                r12. It’s usually worth spending a few extra dollars to get 
                extra App features
    Lifestyle:
                r13. There’s no point in earning money if I’m not going to 
                spend it
                r14. I am influenced by what’s hot and what’s not
                r15. I buy brands that reflect my style
                r16. I tend to make impulse purchases
                r17. Above all else, I think of my mobile phone as a source of
                entertainment
                r18. I find I’m often attracted to luxury brands.

'''

att_pur = data.copy()
att_pur = att_pur.loc[:,'q26r3':'q26r17']
att_pur['q26r18'] = data['q26r18']

# y-axis in bold
rc('font', weight='bold')
 
# Values of each group
att_pur_data=pd.DataFrame()
for column in att_pur.columns[0:]:
    att_pur_data = att_pur_data.append(att_pur[column].value_counts().sort_index())

att_pur_data=att_pur_data.fillna(0)
sa = att_pur_data.loc[:,1]
a = att_pur_data.loc[:,2]
aso = att_pur_data.loc[:,3]
dso = att_pur_data.loc[:,4]
d = att_pur_data.loc[:,5]
sd = att_pur_data.loc[:,6]

# The position of the bars on the x-axis
r = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
 
# Names of group and bar width
names = ['3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18']
barWidth = 1
 
# Create strongly agree
plt.bar(r, sa, color='orangered', edgecolor='white', width=barWidth)
# Create agree
plt.bar(r, a, bottom=sa, color='indianred', edgecolor='white', width=barWidth)
# Create agree some
plt.bar(r, aso, bottom=sa+a, color='salmon', edgecolor='white', width=barWidth)
# Crate disagree some
plt.bar(r, dso, bottom=sa+a+aso, color='lightgray', edgecolor='white', width=barWidth)
# Crate disagree 
plt.bar(r, d, bottom=sa+a+aso+dso, color='grey', edgecolor='white', width=barWidth)
# Crate strongly disagree
plt.bar(r, sd, bottom=sa+a+aso+dso+d, color='k', edgecolor='white', width=barWidth)


# Custom X axis
plt.xticks(r, names, fontweight='bold')
plt.xlabel("group")
 
# Show graphic
plt.show()




################################
#Code for Model
################################



##############################################################################
##############################################################################
######                         Feature Engineering                    ########
##############################################################################
##############################################################################

# Change scale of negative sentiments to positive 

def reverseScoring(df, high, cols):
    """Reverse scores on given columns:
     df = your data frame,
     high = highest score available +1,
     cols = the columns you want reversed in list form"""
    df[cols] = high - df[cols]
    return df
 
neg_sentiments = ['q24r4', 'q24r9', 'q25r6','q25r7']

data = reverseScoring(data, 7, neg_sentiments)


#Combine few columns
data['tv'] = data.q4r2 + data.q4r4 + data.q13r6 + data.q13r10 + data.q13r12 + data.q24r8 + data.q26r17

data['shopping'] = data.q12 + data.q4r8 + data.q4r3 + data.q25r5 + data.q26r3 + data.q26r4 + data.q26r5 + data.q26r6 + data.q26r7 + data.q26r12 + data.q26r13 + data.q26r15 + data.q26r16

data['social'] = data.q4r6 + data.q13r1 + data.q13r2 + data.q13r3 + data.q13r11 + data.q24r9 + data.q24r10 + data.q24r11+ data.q24r12

data['news'] = data.q4r7 + data.q4r9 + data.q13r7

data['effect'] = data.q24r6 + data.q25r12

data['music'] = data.q4r1 + data.q13r4 + data.q13r5 + data.q13r8 + data.q13r9 + data.q24r7

data['e&g'] = data.q4r5 + data.q4r3 + data.q26r17

data['tech'] = data.q24r1 + data.q24r2 + data.q24r4 + data.q24r5

data['opinion'] = data.q11 + data.q25r1 + data.q25r2 + data.q25r3 + data.q25r4 + data.q25r6 + data.q25r7 + data.q25r8 + data.q25r9 + data.q25r10 + data.q25r11 + data.q26r8 + data.q26r9+ data.q26r10+ data.q26r11+ data.q26r14

#Drop original
data_new = data.drop(data.loc[:,'q4r1':'q4r11'],axis=1)
data_new = data_new.drop(data_new.loc[:,'q13r1':'q26r17'],axis=1)
data_new = data_new.drop(['q11','q12'],axis=1)


########################
# Correlation analysis
########################

#fig, ax = plt.subplots(figsize = (100, 100))

df_corr = data_new.corr().round(2)


#find top pairs
def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=100):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

print("Top Absolute Correlations")
print(get_top_abs_correlations(df_corr, 20))




########################
# Prepare PCA
########################

########################
# Step 1: Remove demographic information
########################
data_clean = data_new.drop(['q1','q48','q49','q50r1','q50r2','q50r3',
                        'q50r4','q50r5','q54','q55','q56','q57'],axis=1)

data_clean = data_clean.drop(data_clean.loc[:,'q2r1':'q2r10'],axis=1)

# clean dataset
customer_features_reduced = data_clean.iloc[ : , : ]



########################
# Step 2: Scale to get equal variance
########################
scaler = StandardScaler()


scaler.fit(customer_features_reduced)


X_scaled_reduced = scaler.transform(customer_features_reduced)




########################
# Step 3: Run PCA without limiting the number of components
########################


customer_pca_reduced = PCA(n_components = None,
                           random_state = 508)


customer_pca_reduced.fit(X_scaled_reduced)


X_pca_reduced = customer_pca_reduced.transform(X_scaled_reduced)




# Explained variance as a ratio of total variance
customer_pca_reduced.explained_variance_ratio_ #top 3



########################
# Step 4: Analyze the scree plot to determine how many components to retain
########################
fig, ax = plt.subplots(figsize=(10, 8))

features = range(customer_pca_reduced.n_components_)


plt.plot(features,
         customer_pca_reduced.explained_variance_ratio_,
         linewidth = 2,
         marker = 'o',
         markersize = 10,
         markeredgecolor = 'black',
         markerfacecolor = 'grey')


plt.title('Reduced Wholesale Customer Scree Plot')
plt.xlabel('PCA feature')
plt.ylabel('Explained Variance')
plt.xticks(features)
plt.show()


########################
# Step 5: Run PCA again based on the desired number of components
########################

customer_pca_reduced = PCA(n_components = 3,
                           random_state = 508)


customer_pca_reduced.fit(X_scaled_reduced)


########################
# Step 6: Analyze factor loadings to understand principal components
########################
factor_loadings_df = pd.DataFrame(pd.np.transpose(customer_pca_reduced.components_))


factor_loadings_df = factor_loadings_df.set_index(data_clean.columns[:])


print(factor_loadings_df)


factor_loadings_df.to_excel('final_step6_factor_loadings.xlsx')


########################
# Step 7: Analyze factor strengths per customer
########################

X_pca_reduced = customer_pca_reduced.transform(X_scaled_reduced)


X_pca_df = pd.DataFrame(X_pca_reduced)




###########################################
######  Combining PCA and Clustering         
###########################################


########################
# Step 1: Take your transformed dataframe
########################

print(X_pca_df.head(n = 5))


print(pd.np.var(X_pca_df))



########################
# Step 2: Scale to get equal variance
########################

scaler = StandardScaler()


scaler.fit(X_pca_df)


X_pca_clust = scaler.transform(X_pca_df)


X_pca_clust_df = pd.DataFrame(X_pca_clust)


print(pd.np.var(X_pca_clust_df))


X_pca_clust_df.columns = X_pca_df.columns



########################
# Step 3: Experiment with different numbers of clusters
########################

customers_k_pca = KMeans(n_clusters = 3,
                             random_state = 508)
    
    
customers_k_pca.fit(X_pca_clust_df)
    
    
customers_kmeans_pca = pd.DataFrame({'cluster': customers_k_pca.labels_})
    
    
print(customers_kmeans_pca.iloc[: , 0].value_counts())




########################
# Step 4: Analyze cluster centers
########################

centroids_pca = customers_k_pca.cluster_centers_


centroids_pca_df = pd.DataFrame(centroids_pca)


# Rename your principal components
centroids_pca_df.columns = ['Life for fun', 'Management', 'Outsiders']


print(centroids_pca_df)


# Sending data to Excel
centroids_pca_df.to_excel('customers_pca_centriods.xlsx')



########################
# Step 5: Analyze cluster memberships
########################

clst_pca_df = pd.concat([customers_kmeans_pca,
                         X_pca_clust_df],
                         axis = 1)


print(clst_pca_df)



########################
# Step 6: Reattach demographic information
########################

final_pca_clust_df = pd.concat([data_new.loc[:, ['q1','q48','q49','q50r1',
                                               'q50r2','q50r3','q50r4','q50r5',
                                               'q54','q55','q56','q57']], 
                                clst_pca_df], 
                                axis = 1)
final_pca_clust_df = pd.concat([data_new.loc[:, 'q2r1':'q2r10'], final_pca_clust_df],
                         axis = 1)


print(final_pca_clust_df.head(n = 5))


# Adding a productivity step
data_df = final_pca_clust_df


#########
# Step 6: Analyzing demographic variables with principal components
#########

def PCdemgraph(dem):
    for p in data_df.iloc[:, 22:]: #CLusters features
        fig, ax = plt.subplots(figsize = (8, 4))
        sns.boxplot(x = dem, #the demographic variable of interest
                    y =  p,
                    hue = 'cluster',
                    data = data_df)
        plt.ylim(-2, 10)
        plt.tight_layout()
    return plt.show()

#age
PCdemgraph('q1')

clut0 = data_df.loc[data_df['cluster']==0,:]
clut1 = data_df.loc[data_df['cluster']==1,:]
clut2 = data_df.loc[data_df['cluster']==2,:]

sns.distplot(clut0['q1']).set(xlim=(1, max(clut0['q1'])))
plt.title('Age', fontsize = 25)
plt.xlabel('Age')
plt.show()

sns.distplot(clut1['q1']).set(xlim=(1, max(clut1['q1'])))
plt.title('Age', fontsize = 25)
plt.xlabel('Age')
plt.show()


sns.distplot(clut2['q1']).set(xlim=(1, max(clut2['q1'])))
plt.title('Age', fontsize = 25)
plt.xlabel('Age')
plt.show()


#Income
PCdemgraph('q56')

#Gender
PCdemgraph('q57')

#Educaton
PCdemgraph('q48')

#Marrage
PCdemgraph('q49')

#CHild
PCdemgraph('q50r1')
PCdemgraph('q50r2')
PCdemgraph('q50r3')
PCdemgraph('q50r4')
PCdemgraph('q50r5')


