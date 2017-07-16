# Kaggle-Mercedes-Benz-Greener-Manufacturing-33th-Solution

My final submit was an average of five models.
Two of them I've made myself (weights 0.25 and 0.2)
one model use all feature but some outliers by
 for c in train.columns:
    if c not in train.columns[:11]:
        if  ((train[c].sum() < 1)):
            train.drop(c,axis = 1,inplace = True)
            test.drop(c,axis = 1,inplace = True) 
and the other one use the result with regards to genetic programming
features = ['X118',
            'X127',
            'X47',
            'X315',
            'X311',
            'X179',
            'X314',
### added by Tilii
            'X232',
            'X29',
            'X263',
            'X261']
            
Another two models use different parameter,but the data is same as above(wieght 0.25 and 0.2(genetic feature)), the different is:
I did PCA->GRP->SRP->ICA and the result of PCA is also include as the input of GRP, also the result of GRP is include as the input SRP, 
so did the ICA. That really interesting, the order of these four process can not change otherwise the LB would be worse.

The last model is base on the public kernel by Jason:LB 0.57+ Stack lgbLasGBDT and average with xgb(weight 0.1)
