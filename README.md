# Kaggle-Mercedes-Benz-Greener-Manufacturing-33th-Solution
https://www.kaggle.com/c/mercedes-benz-greener-manufacturing
###My final submit was an average of five models.
###model 1,2
Two of them I've made myself (weights 0.25 and 0.2)
one model use all feature but some outliers by
 for c in train.columns:<br>  
     if c not in train.columns[:11]:<br>  
         if  ((train[c].sum() < 1)):<br>  
             train.drop(c,axis = 1,inplace = True)<br>  
             test.drop(c,axis = 1,inplace = True) <br>  
and the other one use the result with regards to genetic programming
features = ['X118','X127','X47','X315','X311','X179','X314','X232','X29','X263','X261'] 
###model 3,4
Another two models use different parameter,but the data is same as above(wieght 0.25 and 0.2(genetic feature)), the different is:<br>  
I did PCA->GRP->SRP->ICA and the result of PCA is also include as the input of GRP, also the result of GRP is include as the input SRP, 
so did the ICA. That really interesting, the order of these four process can not change otherwise the LB would be worse.
<br>  
The last model is base on the public kernel by Jason:LB 0.57+ Stack lgbLasGBDT and average with xgb(weight 0.1)
