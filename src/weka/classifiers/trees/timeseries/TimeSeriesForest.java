/*
Time Series Forest classifier (TSF) as described in 
@article{deng13forest,
author = {H. Deng and G. Runger and E. Tuv and M. Vladimir},
 title = {A time series forest for classification and feature extraction},
 journal = {Information Sciences},
 volume = {239},
 year = {2013}
}
https://dl.dropboxusercontent.com/u/45301435/time_series_forest.pdf

Original implementation can be found here

https://sites.google.com/site/houtaodeng/publications

 */
package weka.classifiers.trees.timeseries;

import java.util.*;
import weka.classifiers.*;
import weka.core.*;

/**
 *
 * @author ajb
1    forest = cell(ntree,1);      
2    for itree=1:ntree
3        inx = randsample(size(X,1),ceil(size(X,1)*2/2),1);%1: with replacement; 0: without replacement  
4        depth=0;
5        [tree1] = TMakeTree(depth,pre_l,X(inx,:),cls(inx,:),nVar,nMean,nSlope,sampleModeWSZ,sampleModePos,alpha,minWin,maxWin);
6        forest{itree}=tree1;
7    end
Line 3 simply samples the training set the train set size times. ceil(size(X,1)*2/2) =size(X,1)   

TMakeTree:
*   depth init to zero. 
*   pre_1
* Instances X=trainData
    * int n=X.nosAttributes-1
    * for i =1 to nosTrees
    *   TreeSet<TSFInterval> = sample(n,(int(
 */
public class TimeSeriesForest extends AbstractClassifier {
    int nosTrees=500;
    ArrayList<TSTree> trees;
    
 /** TreeSet<TSFInterval> sample(Instances train): algorithm 1 in the deng13
     * randomly samples a set of intervals <T1, T2>, where T1 is the set of 
     * starting time points of intervals, and T2 is the set of ending points. 
     * The function RandSampNoRep(set,sampleSize) randomly selects 
     * samplesize elements from set without replacement.
     * NOTE Algorithm 1 doesnt make much sense, but the Deng implementation
     * just calls matlab function randsample
    */ 
    TreeSet<TSFInterval> sample(Instances train){
        
        return null;
    }
    @Override
    public void buildClassifier(Instances data) throws Exception {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
    public static class TSFInterval{
        
    }
    public static class TSTree{
        
    }


}
