/*

 */
package PS_ACF_experiments;

import utilities.SaveCVAccuracy;
import bakeOffExperiments.*;
import tsc_algorithms.*;
import development.DataSets;
import fileIO.InFile;
import fileIO.OutFile;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.*;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.lazy.DTW_1NN;
import weka.classifiers.lazy.kNN;
import weka.classifiers.meta.OptimisedRotationForest;
import weka.classifiers.meta.RotationForest;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;
import weka.filters.NormalizeCase;
import weka.filters.timeseries.ACF;
import weka.filters.timeseries.BagOfPatternsFilter;
import weka.filters.timeseries.PowerSpectrum;

/**
 *
 * @author ajb
 */
public class Experiments extends Thread{
   Instances train;
   Instances test;
//For threaded version only
   String problem;
   String classifier;
    double testAccuracy;
    String name;
    int resamples=100;
    int fold=0;
    String path;
    String preds;
    double acc;
    public static boolean removeUseless=false;
    
        
    
    public Experiments(Instances tr, Instances te, String cls, String prob, String predictions,int res, int f){
        train=tr;
        test=te;
        classifier=cls;
        problem=prob;
        preds=predictions;
        resamples=res;
        fold=f;
    }
    public double getAcc(){return acc;}
   @Override
    public void run(){
        Classifier c=setClassifier(classifier);
        if(resamples==1){
            //Open up fold file to write to
            File f=new File(preds+"/fold"+fold+".csv");
            if(!f.exists() || f.length()==0){
                acc=singleSampleExperiment(train,test,c,fold,preds);
                System.out.println("Fold "+fold+" acc ="+acc);
            }
            else
                System.out.println("Fold "+fold+" already complete");
        }
        else{
            OutFile of=new OutFile(DataSets.resultsPath+classifier+"/"+problem+".csv");
            of.writeString(problem+",");
            double[] folds=resampleExperiment(train,test,c,100,of,preds);
            of.writeString("\n");
            double mean=0;
            for(double d:folds)
                mean+=d;
            System.out.println(" mean acc="+mean/folds.length);
        }
    }
   
//All classifier names  
    //<editor-fold defaultstate="collapsed" desc="Directory names for all classifiers">   

    
    //Global file to write to 
    static OutFile out;
    
    public static Classifier setClassifier(String classifier){
        Classifier c=null;
        switch(classifier){
            case "RIF_PS":
                c=new RandomIntervalForest();
                ((RandomIntervalForest)c).setFilter("PS");
                break;
            case "RIF_ACF":
                c=new RandomIntervalForest();
                ((RandomIntervalForest)c).setFilter("ACF");
                break;
            case "FixedIntervalForest":
                c=new FixedIntervalForest();
                ((FixedIntervalForest)c).useCV(true);
                break;
            case "ACF":
                c=new ACF_Ensemble();
                ((ACF_Ensemble)c).setClassifierType("WE");
                break;
            case "PS":
                c=new PS_Ensemble();
                ((PS_Ensemble)c).setClassifierType("WE");
                break;
             case "BOSS": case "BOSSEnsemble": 
                c=new BOSSEnsemble();
                break;
             case "LPS":
                c=new LPS();
                break; 
           default:
                System.out.println("UNKNOWN CLASSIFIER");
//                System.exit(0);
//                throw new Exception("Unknown classifier "+classifier);
        }
        return c;
    }
    
    
 //Do all the reps for one problem   
    public static void threadedSingleClassifierSingleProblem(String classifier, String problem,int reps, int start) throws Exception{
        
        Instances train=ClassifierTools.loadData(DataSets.problemPath+problem+"/"+problem+"_TRAIN");
        Instances test=ClassifierTools.loadData(DataSets.problemPath+problem+"/"+problem+"_TEST");
        File f=new File(DataSets.resultsPath+classifier);
        if(!f.exists())
            f.mkdir();
        String predictions=DataSets.resultsPath+classifier+"/Predictions";
        f=new File(predictions);
        if(!f.exists())
            f.mkdir();
        predictions=predictions+"/"+problem;
        f=new File(predictions);
        if(!f.exists())
            f.mkdir();
        f=new File(DataSets.resultsPath+classifier+"/"+problem+".csv");
        if(!f.exists()){
            out=new OutFile(DataSets.resultsPath+classifier+"/"+problem+".csv");
            out.writeString(problem+",");
        }
        else{
            out=new OutFile(DataSets.resultsPath+classifier+"/"+problem+"2.csv");
            out.writeString(problem+",");            
        }
        Experiments[] thr=new Experiments[reps-start];
        for (int i = start; i < reps; i++) {
            Instances[] data=InstanceTools.resampleTrainAndTestInstances(train,test, i);
            thr[i-start]=new Experiments(data[0],data[1],classifier,problem,predictions,1,i);
        }
//Do in batches        
        int processors= Runtime.getRuntime().availableProcessors();
        int count=0;
        Thread[] current=new Thread[processors];
        while(count<thr.length){
            if(thr.length-count<processors){
                processors=thr.length-count;
            }
            for(int i=0;i<processors;i++){
                System.out.println("\t starting repetition "+(start+count));
                current[i]=thr[count++];
                current[i].start();
            }
            for(int i=0;i<processors;i++)
                current[i].join();        
        System.out.println(" Finished the first "+(start+count)+" batches");
        }
        double[] accs=new double[reps-start];
        for (int i = 0; i < accs.length; i++) {
            accs[i]=thr[i].getAcc();
            out.writeString(accs[i]+",");
        }
    }
    public static void singleClassifierZeroFold(String classifier, int prob) throws Exception{
        String problem=DataSets.fileNames[prob];
        System.out.println("ZERO FOLD FOR CLASSIFIER "+classifier+" problem "+problem);
        File f=new File(DataSets.resultsPath+"/"+classifier);
        if(!f.exists())
            f.mkdir();
        String predictions=DataSets.resultsPath+"/"+classifier+"/Predictions";
        f=new File(predictions);
        if(!f.exists())
            f.mkdir();
        predictions=predictions+"/"+problem;
        f=new File(predictions);
        if(!f.exists())
            f.mkdir();
        f=new File(predictions+"/"+"TestFold0.csv");
        if(f.exists())
            f.delete();
        Instances train=ClassifierTools.loadData(DataSets.problemPath+problem+"/"+problem+"_TRAIN");
        Instances test=ClassifierTools.loadData(DataSets.problemPath+problem+"/"+problem+"_TEST");
        OutFile p=new OutFile(predictions+"/"+"TestFold0.csv");
// hack here to save internal CV for furhter ensembling         
        Classifier c=setClassifier(classifier);
        if(c instanceof SaveCVAccuracy)
           ((SaveCVAccuracy)c).setCVPath(predictions+"/internalCV_0.csv");
        if(c instanceof SaveableEnsemble)
           ((SaveableEnsemble)c).saveResults(predictions+"/internalCV_0.csv",predictions+"/internalTestPreds_0.csv");
        try{              
            c.buildClassifier(train);
            double acc=0;
            for(int j=0;j<test.numInstances();j++)
            {
                double act=test.instance(j).classValue();
                double pred=c.classifyInstance(test.instance(j));
                if(act==pred)
                    acc++;
                p.writeLine(act+","+pred);
            }
            acc/=test.numInstances();
            System.out.println(classifier+" on "+problem+" fold 0 accuracy ="+acc);
//            of.writeString(foldAcc[i]+",");

        }catch(Exception e)
        {
                System.out.println(" Error ="+e+" in method simpleExperiment"+e);
                e.printStackTrace();
                System.out.println(" TRAIN "+train.relationName()+" has "+train.numAttributes()+" attributes and "+train.numInstances()+" instances");
                System.out.println(" TEST "+test.relationName()+" has "+test.numAttributes()+" attributes"+test.numInstances()+" instances");

                System.exit(0);
        }
        
    }
    
  
    public static void threadedSingleClassifierSingleProblem(Classifier c, String results,String problem, int reps) throws Exception{
         ThreadedClassifierExperiment[] thr=new ThreadedClassifierExperiment[reps]; 
//Load train test
         int count=0;
         while(count<reps){
            for(int j=0;j<8;j++){
//                OutFile out=new OutFile(results+"fold"+count+".csv");
                Classifier cls=AbstractClassifier.makeCopy(c);
                Instances train=ClassifierTools.loadData(problem+"_TRAIN");
                Instances test=ClassifierTools.loadData(problem+"_TEST");
//Check results directory exists                
                thr[count]=new ThreadedClassifierExperiment(train,test,cls,problem,results);
//                thr[count].resamples=1;
                thr[count].start();
                System.out.println(" started rep="+count);
                count++;
            }
            for(int j=0;j<8;j++){
                 thr[count-j-1].join();
            }
            System.out.println(" finished batch="+count);

       }
         
    }
   
    
 
    
    public static void singleClassifier(String classifier,String problemName) throws Exception{
//
        int position=1;
        while(position<=DataSets.fileNames.length && !DataSets.fileNames[position-1].equals(problemName))
            position++;
        if(position<DataSets.fileNames.length){
            String[] args={classifier,position+""};
            singleClassifier(args);
        }
        else{
            System.out.println("Invalid problem name ="+problemName);
            System.exit(0);
        }
        
    } 
    public static void singleClassifier(String[] args) throws Exception{
//first gives the problem file  
        String classifier=args[0];
        String s=DataSets.fileNames[Integer.parseInt(args[1])-1];        
//        String problem=unfinished[Integer.parseInt(args[1])-1];
        System.out.println("Classifier ="+classifier+" problem ="+s);
        Classifier c=setClassifier(classifier);
        Instances train=ClassifierTools.loadData(DataSets.problemPath+s+"/"+s+"_TRAIN");
        Instances test=ClassifierTools.loadData(DataSets.problemPath+s+"/"+s+"_TEST");
        File f=new File(DataSets.resultsPath+classifier);
        if(!f.exists())
            f.mkdir();
        String predictions=DataSets.resultsPath+classifier+"/Predictions";
        f=new File(predictions);
        if(!f.exists())
            f.mkdir();
        predictions=predictions+"/"+s;
        f=new File(predictions);
        if(!f.exists())
            f.mkdir();
        
        OutFile of=new OutFile(DataSets.resultsPath+classifier+"/"+s+".csv");
        of.writeString(s+",");
        double[] folds;
        if(s.equals("ACF")){
            train=ACF.formChangeCombo(train);
            test=ACF.formChangeCombo(test);
            ((ACF_Ensemble) c).doACFTransform(false);
        }else if(s.equals("PS")){
            PowerSpectrum ps=((PS_Ensemble) c).getTransform();
            train=ps.process(train);
            test=ps.process(test);
            ((PS_Ensemble) c).doTransform(false);
        }
        folds=resampleExperiment(train,test,c,100,of,predictions);
        of.writeString("\n");
    }
    
    public static void singleClassifierAndFoldAndParameter(String[] args) throws Exception{
//first gives the problem file      
        String classifier=args[0];
//Need to make this generic
        Classifier c=setClassifier(classifier);
        if(!(c instanceof ParameterSplittable)){
            System.out.println("ERROR, CLASSIFIER "+classifier+" IS NOT SPLITTABLE");
            System.exit(0);
        }
        
            
        String s=args[1];
        int fold=Integer.parseInt(args[2])-1;
        int para=Integer.parseInt(args[3]);
        String ser=args[4];
        boolean serialise=false;
        if(ser!=null){
            if(ser.equals("TRUE") || ser.equals("1")|| ser.equals("true")) 
             serialise=true;
        }
            
        File f=new File(DataSets.resultsPath+classifier);
        if(!f.exists())
            f.mkdir();
        String predictions=DataSets.resultsPath+classifier+"/Predictions";
        f=new File(predictions);
        if(!f.exists())
            f.mkdir();
        predictions=predictions+"/"+s;
        f=new File(predictions);
        if(!f.exists())
            f.mkdir();
//Check whether fold already exists, if so, dont do it, just quit
        ((ParameterSplittable)c).setParamSearch(false);
        ((ParameterSplittable)c).setPara(para);
        f=new File(predictions+"/fold"+fold+"_"+para+"TRAIN.csv");
        
        if(!f.exists() || f.length()==0){
            Instances train=ClassifierTools.loadData(DataSets.problemPath+s+"/"+s+"_TRAIN");
            Instances test=ClassifierTools.loadData(DataSets.problemPath+s+"/"+s+"_TEST");
            Instances[] data=InstanceTools.resampleTrainAndTestInstances(train, test, fold);
            train=data[0];
            data=null;
            test=null;
            c.buildClassifier(train);
            OutFile of=new OutFile(predictions+"/fold"+fold+"_"+para+"TRAIN.csv");
            of.writeString(para+","+((ParameterSplittable)c).getParas()+","+((ParameterSplittable)c).getAcc());
            
 
            if(serialise){
                FileOutputStream fos = new FileOutputStream(predictions+"/"+classifier+fold+"_"+para+"TRAIN.ser");
                ObjectOutputStream outS = new ObjectOutputStream(fos);
                outS.writeObject(c);
                outS.close();
            }
        }
        else{
            System.out.println("File already exists "+f);
        }
            
    }
    
    public static void singleClassifierAndFold(String[] args){
//first gives the problem file      
        String classifier=args[0];
        String problem=args[1];
        int fold=Integer.parseInt(args[2])-1;
   
        Classifier c=setClassifier(classifier);
        Instances train=ClassifierTools.loadData(DataSets.problemPath+problem+"/"+problem+"_TRAIN");
        Instances test=ClassifierTools.loadData(DataSets.problemPath+problem+"/"+problem+"_TEST");
        File f=new File(DataSets.resultsPath+classifier);
        if(!f.exists())
            f.mkdir();
        String predictions=DataSets.resultsPath+classifier+"/Predictions";
        f=new File(predictions);
        if(!f.exists())
            f.mkdir();
        predictions=predictions+"/"+problem;
        f=new File(predictions);
        if(!f.exists())
            f.mkdir();
//Check whether fold already exists, if so, dont do it, just quit
        f=new File(predictions+"/fold"+fold+".csv");
        if(!f.exists() || f.length()==0){
      //      of.writeString(problem+","); );
            double acc=0;
            acc =singleSampleExperiment(train,test,c,fold,predictions);
            
            
 //       of.writeString("\n");
        }
    }
    public static void singleClassifierAndFold(String classifier,String problemName, int fold) throws Exception{
        int position=1;
        while(position<=DataSets.fileNames.length && !DataSets.fileNames[position-1].equals(problemName))
            position++;
        if(position<DataSets.fileNames.length){
            String[] args={classifier,position+"",(fold+1)+""};
            singleClassifier(args);
        }
        else{
            System.out.println("Invalid problem name ="+problemName);
            System.exit(0);
        }
        
    }    

    
//Static methds for experiments    
    public static double[] resampleExperiment(Instances train, Instances test, Classifier c, int resamples,OutFile of,String preds){

       double[] foldAcc=new double[resamples];
        for(int i=0;i<resamples;i++){
            File f=new File(preds+"/fold"+i+".csv");
            if(!f.exists() || f.length()==0){
            foldAcc[i]=singleSampleExperiment(train,test,c,i,preds);
                of.writeString(foldAcc[i]+",");
            }
            else
                of.writeString(",");
        }            
         return foldAcc;
    }
    public static double singleShapeletExperiment(Classifier c, int sample,String preds){
        Instances[] data=new Instances[2];
        data[0]=null;
        data[1]=null;
        
        double acc=0;
        double act,pred;
        OutFile p=new OutFile(preds+"/fold"+sample+".csv");
// hack here to save internal CV for furhter ensembling   
        if(c instanceof SaveCVAccuracy)
           ((SaveCVAccuracy)c).setCVPath(preds+"/internalCV_"+sample+".csv");       
        if(c instanceof SaveableEnsemble)
           ((SaveableEnsemble)c).saveResults(preds+"/internalCV_"+sample+".csv",preds+"/internalTestPreds_"+sample+".csv");
        try{              
            c.buildClassifier(data[0]);
            for(int j=0;j<data[1].numInstances();j++)
            {
                act=data[1].instance(j).classValue();
                pred=c.classifyInstance(data[1].instance(j));
                if(act==pred)
                    acc++;
                p.writeLine(act+","+pred);
            }
            acc/=data[1].numInstances();
//            of.writeString(foldAcc[i]+",");

        }catch(Exception e)
        {
                System.out.println(" Error ="+e+" in method simpleExperiment"+e);
                e.printStackTrace();
                System.exit(0);
        }
         return acc;
    }


    public static double singleSampleExperiment(Instances train, Instances test, Classifier c, int sample,String preds){
        Instances[] data=InstanceTools.resampleTrainAndTestInstances(train, test, sample);
        double acc=0;
        OutFile p=new OutFile(preds+"/TestFold"+sample+".csv");
        p.writeLine(c.getClass().getName());
        if(c instanceof SaveCVAccuracy)
            p.writeLine(((SaveCVAccuracy)c).getParameters());
        else
            p.writeLine("NoParameterInfo");

// hack here to save internal CV for furhter ensembling   
        if(c instanceof SaveCVAccuracy)
           ((SaveCVAccuracy)c).setCVPath(preds+"/internalCV_"+sample+".csv");
        
        if(c instanceof SaveableEnsemble)
           ((SaveableEnsemble)c).saveResults(preds+"/internalCV_"+sample+".csv",preds+"/internalTestPreds_"+sample+".csv");
        try{              
            c.buildClassifier(data[0]);
            int[][] predictions=new int[data[1].numInstances()][2];
            for(int j=0;j<data[1].numInstances();j++)
            {
                predictions[j][0]=(int)data[1].instance(j).classValue();
                predictions[j][1]=(int)c.classifyInstance(data[1].instance(j));
                if(predictions[j][0]==predictions[j][1])
                    acc++;
            }
            acc/=data[1].numInstances();
            p.writeLine(acc+"");
            for(int j=0;j<data[1].numInstances();j++)
                p.writeLine(predictions[j][0]+","+predictions[j][1]);
            
//            of.writeString(foldAcc[i]+",");

        }catch(Exception e)
        {
                System.out.println(" Error ="+e+" in method simpleExperiment"+e);
                e.printStackTrace();
                System.out.println(" TRAIN "+train.relationName()+" has "+train.numAttributes()+" attributes and "+train.numInstances()+" instances");
                System.out.println(" TEST "+test.relationName()+" has "+test.numAttributes()+" attributes and "+test.numInstances()+" instances");

                System.exit(0);
        }
         return acc;
    }
    public static void testACC() throws Exception{
//Load a classifier
        String problem="ItalyPowerDemand";
             TSBF t;
            FileInputStream fis = new FileInputStream("C:\\Users\\ajb\\Dropbox\\Big TSC Bake Off\\New Results\\interval\\TSBF\\Predictions\\ItalyPowerDemand\\TSBF0_2TRAIN.ser");
            ObjectInputStream in = new ObjectInputStream(fis);
           t =(TSBF)in.readObject();
           in.close();
           Instances test=ClassifierTools.loadData(DataSets.problemPath+problem+"\\"+problem+"_TEST");
           double a=ClassifierTools.accuracy(test, t);
                System.out.println("ACC ="+a); 
             System.exit(0);
        
    }
        
    public static void clusterRun(String[] args) throws Exception{
        if(args.length>3)   //
            singleClassifierAndFoldAndParameter(args);
        else if(args.length>2)   //
            singleClassifierAndFold(args);
        else
            singleClassifier(args);
            
    }
//Works with the pre-generatedd Shapelet transform filse     
    public static void shapeletRun(String[] args) throws Exception{
        
        
    }
    public static void reconstruct(String classifier, String problem,  int fold, int paras) throws Exception
     {
        String path=DataSets.resultsPath+classifier+"/Predictions/"+problem;
        System.out.println("PATH = "+path);
        boolean oneCompletePara=false;
        double bestAcc=0;
        int bestC=0;
        fold=fold-1;
        for (int j = 1; j <= paras; j++) {
//Check file exists 
            File f=new File(path+"/fold"+fold+"_"+j+"TRAIN.csv");
            if(!f.exists()){//Fold is not complete
                System.out.println("Fold "+fold+" paras "+j+" incomplete on "+path+"/fold"+fold+"_"+j+"TRAIN.csv");
                
            }            
//Find accuracy
            else{
                oneCompletePara=true;
                System.out.println("Parameter values "+j+" is complete");
                InFile inf=new InFile(path+"/fold"+fold+"_"+j+"TRAIN.csv");
                int p=inf.readInt();
                double d=inf.readDouble();
                double acc=inf.readDouble();
                if(acc>bestAcc){
                    bestAcc=acc;
                    bestC=j;
                }
            }
        }
//Load best classifier if all paras have been tested
        Classifier cls=null;
        Instances train=ClassifierTools.loadData(DataSets.problemPath+problem+"/"+problem+"_TRAIN");
        Instances test=ClassifierTools.loadData(DataSets.problemPath+problem+"/"+problem+"_TEST");
        Instances[] data=InstanceTools.resampleTrainAndTestInstances(train, test, fold);
        test=data[1];
        train=data[0];
        if(oneCompletePara){
            System.out.println("Best Train Accuracy="+bestAcc+" with param setting"+bestC);
            File f=new File(path+"/"+classifier+fold+"_"+bestC+"TRAIN.ser");
            if(f.exists()){
                 FileInputStream fis = new FileInputStream(path+"/"+classifier+fold+"_"+bestC+"TRAIN.ser");
                 ObjectInputStream inS = new ObjectInputStream(fis);
                 cls = (Classifier)inS.readObject();
                 System.out.println("Classifier "+classifier+" loaded");
                 inS.close();
            }
            else{
                System.out.println("SER fILE "+f+" does not exist need to rebuild the classifier");
                cls=setClassifier(classifier);
//need to set the right parameters
                if(cls instanceof ParameterSplittable){
                    ((ParameterSplittable)cls).setPara(bestC);
                    ((ParameterSplittable)cls).setParamSearch(false);
                    cls.buildClassifier(train);
                }
                else{
                    System.out.println("ERROR: "+classifier+" NOT SPLITTABLE");
                    System.exit(0);
                }
            }
             OutFile p=new OutFile(path+"/fold"+fold+".csv");
// hack here to save internal CV for furhter ensembling
             double acc=0;
             for(int j=0;j<test.numInstances();j++)
             {
                 double act=test.instance(j).classValue();
                 double pred=cls.classifyInstance(test.instance(j));
                 p.writeLine(act+","+pred);
                 if(act==pred)
                     acc++;
             }
             acc/=test.numInstances();
             System.out.println(" TEST ACC ="+acc);
        }
//            
    }


public static void main(String[] args) throws Exception{
    String version="RandomIntervalForest";
    try{
        if(args.length>0){ //Cluster run
            for (int i = 0; i < args.length; i++) {
                System.out.println("ARGS ="+i+" = "+args[i]);
            }
            DataSets.resultsPath=DataSets.clusterPath+"Results/"+version+"/";
            File f= new File(DataSets.resultsPath);
            if(!f.exists())
                f.mkdir();
            DataSets.problemPath=DataSets.clusterPath+"TSC Problems/";
//                int prob=Integer.parseInt(args[1])-1;
//                singleClassifierZeroFold(args[0],prob);
            clusterRun(args);
        }
        else{       
//Local threaded run    
            DataSets.resultsPath=DataSets.dropboxPath+"Spectral Interval Experiments/";
            DataSets.problemPath=DataSets.dropboxPath+"TSC Problems/";
//            System.exit(0);
            String problem="ItalyPowerDemand";
            String probID="38";
            
            String[] ar={version,probID};
            clusterRun(ar);
            }
        }catch(Exception e){
            System.out.println("Exception thrown ="+e);
            e.printStackTrace();
        }
    }

    public static void sparseInstanceCheck() throws Exception{
        Instances train=ClassifierTools.loadData(DataSets.problemPath+"ItalyPowerDemand\\ItalyPowerDemand_TRAIN");
        BagOfPatternsFilter bop=new  BagOfPatternsFilter(2,4,8);
        Instances transformedBOP = bop.process(train);
        System.out.println("SAX Instances atts: "+transformedBOP.numAttributes());
        System.out.println("SAX Single Instance atts: "+transformedBOP.instance(0).numAttributes());
        System.exit(0);


//  Check to see if numAttributes() is broken in Instances with SparseInstance
        int numFeatures=4; 
        //Set up instances size and format. 
        FastVector atts=new FastVector();
        String name;
        for(int j=0;j<numFeatures;j++){
                name = "Feature"+j;
                atts.addElement(new Attribute(name));
        }
        //Get the class values as a fast vector			
        Attribute target =new Attribute("classValue");
        atts.addElement(target);
//create blank instances with a class value              
        Instances result = new Instances("Temp",atts,0);
        result.setClassIndex(result.numAttributes()-1);
        for(int i=0;i<20;i++){
            SparseInstance in=new SparseInstance(result.numAttributes());
            result.add(in);
        }
        System.out.println("SPARSE: number of atts ="+result.numAttributes());
        System.out.println("SPARSE: number of atts instance 0 ="+result.instance(0).numAttributes());
        System.out.println("SPARSE: class index ="+result.instance(0).classIndex());
        
        
        
        
    }

}
