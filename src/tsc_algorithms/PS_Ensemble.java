package tsc_algorithms;

import fileIO.OutFile;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.meta.RotationForest;
import weka.classifiers.meta.timeseriesensembles.WeightedEnsemble;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.timeseries.PowerSpectrum;

/**
 *
 * @author ajb
 */
public class PS_Ensemble extends AbstractClassifier implements SaveableEnsemble{
    Classifier baseClassifier;
    Instances format;
    ClassifierType c=ClassifierType.RandF;
    private boolean saveResults=false;
    private String trainCV="";
    private String testPredictions="";
    int[] constantFeatures;
    PowerSpectrum ps=new PowerSpectrum();
    boolean doTransform=true;
    public PowerSpectrum getTransform(){ return ps;}
    protected void saveResults(boolean s){
        saveResults=s;
    }
    public void saveResults(String tr, String te){
        saveResults(true);
        trainCV=tr;
        testPredictions=te;
    }
     public void doTransform(boolean b){
        doTransform=b;
    }
   
    //Power Spectrum
    public enum ClassifierType{
        RandF("RandF",500),RotF("RotF",50),WeightedEnsemble("WE",8);
        String type;
        int numBaseClassifiers;
        ClassifierType(String s, int x){
            type=s;
            numBaseClassifiers=x;
        }
        Classifier createClassifier(){
            switch(type){
                case "RandF":
                   RandomForest randf=new RandomForest();
                   randf.setNumTrees(numBaseClassifiers);
                    return randf;
                case "RotF":
                   RotationForest rotf=new RotationForest();
                   rotf.setNumIterations(numBaseClassifiers);
                   return rotf;
                case "WE":
                   WeightedEnsemble we=new WeightedEnsemble();
                   we.setWeightType("prop");
                   return we;
                default:
                   RandomForest c=new RandomForest();
                   c.setNumTrees(numBaseClassifiers);
                    return c;
            }
        }
    }
    
    public  void setClassifierType(String s){
        s=s.toLowerCase();
        switch(s){
            case "randf": case "randomforest": case "randomf":
                c=ClassifierType.RandF;
                break;
            case "rotf": case "rotationforest": case "rotationf":
                c=ClassifierType.RotF;
                break;
            case "weightedensemble": case "we": case "wens": 
                c=ClassifierType.WeightedEnsemble;
                break;                
        } 
    }
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        
        ps=new PowerSpectrum();
        baseClassifier=c.createClassifier();
        Instances psTrain;
        if(doTransform)
             psTrain=ps.process(data);
        else
             psTrain=data;
        constantFeatures=InstanceTools.removeConstantTrainAttributes(psTrain);
        System.out.println(" Number of constant attributes removed ="+constantFeatures.length);
        
        
        
        if(saveResults && c==ClassifierType.WeightedEnsemble){
//Set up the file space here
            System.out.println("SAVING RESULTS FOR TRAIN to "+trainCV+" "+testPredictions);
            ((WeightedEnsemble) baseClassifier).saveTrainCV(trainCV);
            ((WeightedEnsemble) baseClassifier).saveTestPreds(testPredictions);
        }
        
        baseClassifier.buildClassifier(psTrain);
        format=new Instances(data,0);
    }
    @Override
    public double classifyInstance(Instance ins) throws Exception{
//   
        format.add(ins);    //Should match!
        Instances temp;
        if(doTransform)
            temp=ps.process(format);
        else
            temp=format;
//Delete constants
        for(int del:constantFeatures)
            temp.deleteAttributeAt(del);
        Instance trans=temp.get(0);
        format.remove(0);
        return baseClassifier.classifyInstance(trans);
    }
    
//
 public static void brokenFiles() throws Exception{
//Empty file: ElectricDevices/internalCv_0.csv
     Instances train=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\ElectricDevices\\ElectricDevices_TRAIN");
     Instances test=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\ElectricDevices\\ElectricDevices_TEST");
     PS_Ensemble ps=new PS_Ensemble();
//     preds+"/internalCV_"+sample+".csv",preds+"/internalTestPreds_"+sample+".csv");
//     
//     ps.saveResults(null, null);
//     ps.buildClassifier(train);
//     System.out.println("Build finished");
//     double a = ClassifierTools.accuracy(test, ps);
//     System.out.println("Test finished acc = "+a);
     
     int fold=0;
     Instances[] data=InstanceTools.resampleTrainAndTestInstances(train, test, fold);
     ps=new PS_Ensemble();
     ps.setClassifierType("WE");
//     preds+"/internalCV_"+sample+".csv",preds+"/internalTestPreds_"+sample+".csv");
     String preds="C:\\Users\\ajb\\Dropbox\\Big TSC Bake Off\\New Results\\ensemble\\PS\\Predictions\\ElectricDevices";
     
     ps.saveResults(preds+"\\internalCV_"+fold+".csv", preds+"\\internalTestPreds_"+fold+".csv");
     ps.buildClassifier(data[0]);
     System.out.println("Build finished for fold "+fold);
     OutFile of=new OutFile(preds+"\\fold"+fold+".csv");
     for(Instance ins:data[1]){
         int act=(int)ins.classValue();
         int pred=(int)ps.classifyInstance(ins);
         of.writeLine(act+","+pred);
     }
//     double a = ClassifierTools.accuracy(data[1], ps);
     System.out.println("Test finished ");
     
     //Empty file: Lightning7/internalCv_66.csv
     
////Empty file: ProximalPhalanxOutlineAgeGroup/internalCv_34.csv
////Empty file: ProximalPhalanxOutlineAgeGroup/internalCv_42.csv
//     fold=34;
//     train=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\ProximalPhalanxOutlineAgeGroup\\ProximalPhalanxOutlineAgeGroup_TRAIN");
//     test=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\ProximalPhalanxOutlineAgeGroup\\ProximalPhalanxOutlineAgeGroup_TEST");
//     preds="C:\\Users\\ajb\\Dropbox\\Big TSC Bake Off\\New Results\\ensemble\\PS\\Predictions\\ProximalPhalanxOutlineAgeGroup";
//     data=InstanceTools.resampleTrainAndTestInstances(train, test, fold);
//     ps=new PS_Ensemble();
//     ps.setClassifierType("WE");
////     preds+"/internalCV_"+sample+".csv",preds+"/internalTestPreds_"+sample+".csv");
//     
//     ps.saveResults(preds+"\\internalCV_"+fold+".csv", preds+"\\internalTestPreds_"+fold+".csv");
//     ps.buildClassifier(data[0]);
//     System.out.println("Build finished for fold "+fold);
//     OutFile of=new OutFile(preds+"\\fold"+fold+".csv");
//     for(Instance ins:data[1]){
//         int act=(int)ins.classValue();
//         int pred=(int)ps.classifyInstance(ins);
//         of.writeLine(act+","+pred);
//     }
////     double a = ClassifierTools.accuracy(data[1], ps);
//     System.out.println("Test finished ");
//     fold=42;
//     data=InstanceTools.resampleTrainAndTestInstances(train, test, fold);
//     ps=new PS_Ensemble();
//     ps.setClassifierType("WE");
////     preds+"/internalCV_"+sample+".csv",preds+"/internalTestPreds_"+sample+".csv");
//     
//     ps.saveResults(preds+"\\internalCV_"+fold+".csv", preds+"\\internalTestPreds_"+fold+".csv");
//     ps.buildClassifier(data[0]);
//     System.out.println("Build finished for fold "+fold);
//     of=new OutFile(preds+"\\fold"+fold+".csv");
//     for(Instance ins:data[1]){
//         int act=(int)ins.classValue();
//         int pred=(int)ps.classifyInstance(ins);
//         of.writeLine(act+","+pred);
//     }
////     double a = ClassifierTools.accuracy(data[1], ps);
//     System.out.println("Test finished ");
//


//    data=InstanceTools.resampleTrainAndTestInstances(train, test, 42);
//     
 }
    public static void main(String[] args) throws Exception {
        brokenFiles();
    }
 
    
}
