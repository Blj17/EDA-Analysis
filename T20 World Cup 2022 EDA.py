#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import cufflinks as cf
import plotly.express as px
import plotly.graph_objs as go
get_ipython().run_line_magic('matplotlib', 'inline')
from plotly.offline import download_plotlyjs, init_notebook_mode,plot,iplot
init_notebook_mode(connected=True)
cf.go_offline()


# In[124]:


T20_df=pd.read_csv('T20.csv')
T20_df.head()


# In[125]:


T20_df.shape


# In[126]:


T20_df.describe()


# In[127]:


T20_df.info()


# In[128]:


T20_df.isnull().sum()


# In[129]:


T20_df.columns


# In[130]:


Winners=T20_df.Winner.value_counts()
fig= px.bar(Winners,x=Winners.index,y=Winners.values,color='Winner',title='Number of matches won by Teams')
fig.show()


# At the end of the tournament __England__ is leading by winning five matches overall.

# In[131]:


T20_df['Won_By']=T20_df.Won_by.apply(lambda  x : x.split(' ')[1])


# In[132]:


T20_df['Won_By']=T20_df['Won_By'].map({"Wickets":"Chasing","Runs":"Defending","Run":"Defending"})


# In[133]:


T20_df['Won_By']=T20_df.Won_By.fillna('Draw')


# In[180]:


Won_by=T20_df.Won_By.value_counts()
px.bar(Won_by,x=Won_by.index,y=Won_by.values,color='Won_By',title='Number of matches Won by Defending or Chasing')


# Most Number of Teams won the match by __Defending__

# In[135]:


T20_df.Toss_decision.value_counts()


# In[136]:


T20_df['Toss_decision']=T20_df.Toss_decision.apply(lambda x : 'Draw' if x=='_' else x)


# In[137]:


T20_df.Toss_decision.value_counts()


# In[138]:


T20_df[T20_df.Toss_decision=='Draw']


# In[139]:


T20_df['Score_of_first_innings']=T20_df.Score_of_first_innings.apply(lambda x : '0' if x=='_' else x)


# In[140]:


T20_df['Score_of_first_innings']=T20_df.Score_of_first_innings.astype('int32')


# In[150]:


T20_df['Score_of_second_innings']=T20_df.Score_of_second_innings.apply(lambda x : '0' if x=='_' else x)


# In[151]:


T20_df['Score_of_second_innings']=T20_df.Score_of_second_innings.astype('int32')


# In[141]:


T20_df['Wkts_in_first_innings']=T20_df.Wkts_in_first_innings.apply(lambda x : '0' if x=='_' else x)


# In[142]:


T20_df['Wkts_in_first_innings']=T20_df.Wkts_in_first_innings.astype('int64')


# In[143]:


T20_df['Wkts_in_second_innings']=T20_df.Wkts_in_second_innings.apply(lambda x : '0' if x=='_' else x)


# In[144]:


T20_df['Wkts_in_second_innings']=T20_df.Wkts_in_second_innings.astype('int64')


# In[145]:


T20=T20_df['Toss_decision'].value_counts()
T20


# In[146]:


px.pie(T20,names=T20.index,values=T20.values,color='Toss_decision')


# In More than __50%__ Matches captains has choosen to __Bat__ after winning Toss

# In[181]:


fig=go.Figure()
fig.add_trace(go.Bar(y=T20_df['Wkts_in_first_innings'],x=T20_df['Match_id'],marker_color='gold',name='First Innings Wickets'))
fig.add_trace(go.Bar(y=T20_df['Wkts_in_second_innings'],x=T20_df['Match_id'],marker_color='lightgreen',name='Second Innings Wickests'))
fig.update_layout(height=600, width=800,barmode='group',title='Wickets Fallen')
fig.show()


# Obsevation:<br>
#     The __second Innings__ has Most __All outs__.

# In[184]:


fig=go.Figure()
fig.add_trace(go.Bar(y=T20_df.Score_of_first_innings,x=T20_df.Match_id,marker_color='orange'))
fig.update_layout(title_text='Run scored in First Innings')
fig.show()


# In[185]:


fig=go.Figure()
fig.add_trace(go.Bar(y=T20_df.Score_of_second_innings,x=T20_df.Match_id,marker_color='green'))
fig.update_layout(title_text='Run scored in Second Innings')
fig.show()


# In[176]:


T20_dfw=T20_df[T20_df['Toss_winner']==T20_df['Winner']]
TW=T20_dfw.Winner.value_counts()


# In[186]:


px.bar(TW,x=TW.index,y=TW.values,color='Winner',title='Teams Won both Toss and Match')


# __Summary:__<br>
#     __England__ has won Most Number.<br>
#     __Highest Score__ in first Innings was 205.<br>
#     __Lowest score__ in the first Innings was 79.<br>
#     __Highest Score__ in the Second Innings was 180.<br>
#     __Lowest Score__ in the Second Innings was 51.<br>
#     __England__ has Won Both the Toss and the Match for __4__ times.<br>
#     
#     

# In[ ]:




