/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package bakeOffExperiments;

import bakeOffExperiments.Experiments;
import utilities.InstanceTools;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.filters.SimpleBatchFilter;
import weka.filters.unsupervised.attribute.RemoveUseless;

/**
 *
 * @author ajb
 */
public class ThreadedClassifierExperiment extends Thread{
    public String root;
    Instances train;
    Instances test;
    Classifier c;
    double testAccuracy;
    SimpleBatchFilter filter;
    String name;
    int resamples=100;
    int fold=0;
    public static boolean removeUseless=false;
    
    public ThreadedClassifierExperiment(Instances tr, Instances te, Classifier cl,String n, String path){
        train=tr;
        test=te;
        c=cl;
        filter=null;
        name=n;
        root=path;
    }
    public void setTransform(SimpleBatchFilter t){
        filter=t;
    }
    public double getTestAccuracy(){ 
        return testAccuracy;
    }
    public void singleExperiment(int fold){
        testAccuracy=0;
        double act;
        double pred;
        Experiments.singleSampleExperiment(train,test,c,fold,root);
    }
    
    public void resampleExperiment(){
        double[] foldAcc=new double[resamples];
        for(int i=0;i<resamples;i++){
//Filter out attributes with all the same values in the train data. These break BayesNet discretisation
                if(removeUseless){
                    InstanceTools.removeConstantTrainAttributes(train,test);
                }
                singleExperiment(i);
            }
        
            synchronized(Experiments.out){
                System.out.println(" finished ="+name);

                Experiments.out.writeString(name+",");
                for(int i=0;i<resamples;i++)
                    Experiments.out.writeString(foldAcc[i]+",");
                Experiments.out.writeString("\n");
            
        }
    }
    
    @Override
    public void run() {
		//Perform a simple experiment,
        if(resamples==1){
            singleExperiment(fold);
        }
        else
            resampleExperiment();
    }
    
}
