/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package applications;

import utilities.ClassifierTools;
import bakeOffExperiments.ThreadedClassifierExperiment;
import weka.classifiers.Classifier;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.lazy.DTW_1NN;
import weka.classifiers.lazy.IBk;
import weka.classifiers.lazy.kNN;
import weka.core.Instances;
import weka.core.shapelet.QualityMeasures;
import weka.filters.NormalizeCase;
import weka.filters.timeseries.shapelet_transforms.FullShapeletTransform;

/**
 *
 * @author ajb
 */
public class FoodSpectrograms {
    public static Instances[] train, test;
    public static Instances strawberryTrain, strawberryTest;
    public static Instances wineTrain, wineTest;
    public static Instances hamTrain, hamTest;
    public static String path="C:\\Users\\ajb\\Dropbox\\TSC Problems\\";
    public static String[] names={"Meat","Wine","Ham","Strawberry"};
    public static void loadData(){
        train=new Instances[4];
        test=new Instances[4];
        train[0]=ClassifierTools.loadData(path+"Meat\\Meat_TRAIN");
        test[0]=ClassifierTools.loadData(path+"Meat\\Meat_TEST");
        train[1]=ClassifierTools.loadData(path+"Wine\\Wine_TRAIN");
        test[1]=ClassifierTools.loadData(path+"Wine\\Wine_TEST");
        train[2]=ClassifierTools.loadData(path+"Ham\\Ham_TRAIN");
        test[2]=ClassifierTools.loadData(path+"Ham\\Ham_TEST");
        train[3]=ClassifierTools.loadData(path+"Strawberry\\Strawberry_TRAIN");
        test[3]=ClassifierTools.loadData(path+"Strawberry\\Strawberry_TEST");
        
        NormalizeCase nc =new NormalizeCase();
        for(int i=0;i<test.length;i++){
            try{
                train[i]=nc.process(train[i]);
                test[i]=nc.process(test[i]);
            }catch(Exception e){
                System.out.println(" Errereere");
                System.exit(0);
            
            }
        }
    }
    public static void shapeletClassifier(){
        int nosExp=3;
        ThreadedClassifierExperiment[] runs= new ThreadedClassifierExperiment[nosExp];
        Thread[] threads=new Thread[nosExp];
        for(int i=0;i<nosExp;i++){
            	Classifier c=new SMO();
		PolyKernel kernel = new PolyKernel();
		kernel.setExponent(1);
		((SMO)c).setKernel(kernel);

            FullShapeletTransform s = new FullShapeletTransform();
            s.setDebug(false);
            s.setNumberOfShapelets(train[i].numAttributes()/2);        
            int minLength=5;
            int maxLength=train[i].numAttributes()/4;
        //       int maxLength=(train.numAttributes()-1)/10;
            s.setShapeletMinAndMax(minLength, maxLength);
            s.setQualityMeasure(QualityMeasures.ShapeletQualityChoice.F_STAT);
            s.turnOffLog();            
            runs[i]=new ThreadedClassifierExperiment(train[i],test[i],c,"shapelet","test");
            runs[i].setTransform(s);
            threads[i]=new Thread(runs[i]);
        }
        for(int i=0;i<nosExp;i++)
            threads[i].start();
        
        try{
            for(int i=0;i<nosExp;i++)
                threads[i].join();
        }catch(InterruptedException e){
            System.out.println(" Interrupted!!");
        }
        for(int i=0;i<nosExp;i++)
            System.out.println(" ED Accuracy for "+names[i]+" is "+runs[i].getTestAccuracy());
        
    }

    
        public static void baselineClassifier(){
        int nosExp=4;
        ThreadedClassifierExperiment[] runs= new ThreadedClassifierExperiment[nosExp];
        Thread[] threads=new Thread[nosExp];
        
        
        for(int i=0;i<nosExp;i++){
            DTW_1NN c=new DTW_1NN();
            c.optimiseWindow(true);
                    
            runs[i]=new ThreadedClassifierExperiment(train[i],test[i],c,"baseline","test");
            threads[i]=new Thread(runs[i]);
        }
        for(int i=0;i<nosExp;i++)
            threads[i].start();
        
        try{
            for(int i=0;i<nosExp;i++)
                threads[i].join();
        }catch(InterruptedException e){
            System.out.println(" Interrupted!!");
        }
        for(int i=0;i<nosExp;i++)
            System.out.println(" DTWCV Accuracy for "+names[i]+" is "+runs[i].getTestAccuracy());
        
    }

    public static void main(String[] args){
        loadData();
 //       baselineClassifier();
        shapeletClassifier();
    }
}
