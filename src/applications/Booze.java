/*
 Exploratory analysis of classifier performance on the IFR 
Alcohol data

Data Path

*/

package applications;

import fileIO.InFile;
import fileIO.OutFile;
import java.util.ArrayList;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import utilities.ClassifierTools;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.Kernel;
import weka.classifiers.lazy.kNN;
import tsc_algorithms.COTE;
import tsc_algorithms.ElasticEnsemble;
import weka.classifiers.meta.timeseriesensembles.WeightedEnsemble;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.elastic_distance_measures.BasicDTW;
import weka.filters.NormalizeCase;
import weka.filters.timeseries.SummaryStats;

/**
 *
 * @author ajb
 */
public class Booze {
      static int nosDistilleries=46;
  
public static void generateHeader(){
    OutFile f = new OutFile("C:\\Data\\IFR Spirits\\Header.csv");
    f.writeString("@attribute distillery {");
    f.writeLine("aberfeldy,aberlour,allmalt,amrut,ancnoc,armorik,arran10,arran14,asyla,auchentoshan,balblair,benromach,bernheim,bladnoch,blairathol,cardhu,dutchsinglemalt,elijahcraig,englishwhisky15,englishwhisky9,exhibition,finlaggan,glencadam,glendeveron,glenfarclas,glenfiddich,glengoyne,glenlivet12,glenlivet15,glenmorangie,glenmoray,glenscotia,greatkingst,highlandpark,laphroaig,mackmyra,nikka,oakcross,organic,peatmonster,scapa,smokehead,speyburn,spicetree,talisker,tyrconnell}");
    
    for(int i=0;i<1748;i++){
        f.writeLine("@attribute wavelength"+(226.0+(i+1)/2.0)+" numeric");
    }
    f.writeLine("@attribute abv {40,44,50,55,60}");
    
}  


public static void transformKateStyle(){
    Instances all = ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\IFR Spirits\\AllSamples");
//1: Restrict to a region 890-1050

//Baseline shift
    
//Standardise area under the curve    
}
static double splitProp=0.7;

/** this just does a random split, irrespective of distillery. Seems likely
 * that instance based classifiers will do well with it!
 * @param all
 * @param rep
 * @throws Exception 
 */
public static double resampleExperimentBasic(Instances all,int rep,String resultPath, Classifier c) throws Exception{
    all.randomize(new Random());
    Instances train = new Instances(all);
    Instances test= new Instances(all,0);
    int testSize=(int)(all.numInstances()*splitProp);
    for(int i=0;i<testSize;i++)
        test.add(train.remove(0));
    System.out.println(" Train size ="+train.numInstances()+" test size ="+test.numInstances());
    c.buildClassifier(train);
    
 //   double[] cvAcc=c.getCVAccs();
    double acc=ClassifierTools.accuracy(test, c);
    System.out.println("acc ="+acc);
    return acc;
//    of.writeString(rep+","+acc+",");
//    for(int i=0;i<cvAcc.length;i++)
//        of.writeString(cvAcc[i]+",");
    
}

/** this just does a leave one distillery out split. Should be harder
 * @param all
 * @param rep
 * @throws Exception 
 */
public static void resampleExperimentSplitByDistillery(Instances all,int rep,String resultPath) throws Exception{
    int nosPerDistillery=all.numClasses()*4;
    if(rep>=nosDistilleries) throw new Exception("Invalid distillery identifier");
    ElasticEnsemble ee=new ElasticEnsemble();
    all.sort(0);
    
    Instances train = new Instances(all);
    Instances test= new Instances(all,0);
    for(int i=0;i<nosPerDistillery;i++)
        test.add(train.remove(nosPerDistillery*rep));
    System.out.println("REP = "+rep+" Train size ="+train.numInstances()+" test size ="+test.numInstances()+" ");
    for(int i=1;i<nosPerDistillery;i++)
        if(test.instance(i).value(0)!=test.instance(i).value(0))
            throw new Exception("INCORRECT SPLIT FOR "+test.instance(i).value(0));
    for(Instance ins:test){
        System.out.print((int)ins.value(0)+",");
    }
    train.deleteAttributeAt(0);
    test.deleteAttributeAt(0);
    ee.buildClassifier(train);
    
    double[] cvAcc=ee.getCVAccs();
    double acc=ClassifierTools.accuracy(test, ee);
    OutFile of = new OutFile(resultPath+"booze"+rep+".csv");
    of.writeString(rep+","+acc+",");
    for(int i=0;i<cvAcc.length;i++)
        of.writeString(cvAcc[i]+",");
   
}


public static void classifyBySummaryStats() throws Exception{
    String resultPath="C:\\Users\\ajb\\Dropbox\\Results\\IFR Spirits\\SummaryStats\\";
    String sourcePath="C:\\Users\\ajb\\Dropbox\\IFR Spirits\\";
    Instances all=ClassifierTools.loadData(sourcePath+"AllSamplesFiveClass");
    
    int nosBottles=46;
    int nosPerBottle=all.numClasses()*4;
    OutFile of=new OutFile(resultPath+"boozeSummaryStatsFiveClassLimited.csv");
//    System.out.println(" All Size ="+all.numInstances()+" num attributes ="+all.numAttributes());
    all.sort(0);
    for(int i=0;i<1300;i++)
        all.deleteAttributeAt(1);
    for(int i=0;i<200;i++)
        all.deleteAttributeAt(all.numAttributes()-2);
    ArrayList<String> names =new ArrayList<>();
    Classifier[] c=ClassifierTools.setDefaultSingleClassifiers(names);
    of.writeString(",");
    for(String s:names)
        of.writeString(s+",");
    of.writeString("\n");
    
    for(int rep=0;rep<nosBottles;rep++){
        Instances train = new Instances(all);
        Instances test= new Instances(all,0);
        for(int i=0;i<nosPerBottle;i++)
            test.add(train.remove(nosPerBottle*rep));
        for(int i=1;i<nosPerBottle;i++)
            if(test.instance(i).value(0)!=test.instance(i).value(0))
                throw new Exception("INCORRECT SPLIT FOR "+test.instance(i).value(0));
        if(rep==0){
                System.out.println(" Test Split ="+test.instance(0).stringValue(0));
                double[] d=test.instance(0).toDoubleArray();
                for(int i=0;i<10;i++)
                    System.out.print(d[i]+",");
                System.out.print("\n");
                    
        }
        train.deleteAttributeAt(0);
        test.deleteAttributeAt(0);
        SummaryStats ss = new SummaryStats();   
        Instances trainStats=ss.process(train);
        Instances testStats=ss.process(test);
        if(rep==0){
                System.out.println(" Summary Stats");
                double[] d=testStats.instance(0).toDoubleArray();
                for(int i=0;i<d.length;i++)
                    System.out.print(d[i]+",");
                System.out.print("\n");
                    
        }
        if(rep==0){
            OutFile tr=new OutFile("C:\\Users\\ajb\\Dropbox\\IFR Spirits\\Debug\\SummaryStats_TRAIN.arff");
            tr.writeString(trainStats+"");
            tr=new OutFile("C:\\Users\\ajb\\Dropbox\\IFR Spirits\\Debug\\SummaryStats_TEST.arff");
            tr.writeString(testStats+"");
            
        }
        names =new ArrayList<>();
        c=ClassifierTools.setDefaultSingleClassifiers(names);
  //      WeightedEnsemble we = new WeightedEnsemble();
        of.writeString(rep+",");
        System.out.println(" Rep ="+rep);
        for(Classifier cl:c){
            double a=ClassifierTools.singleTrainTestSplitAccuracy(cl, trainStats, testStats);
            of.writeString(a+",");
        }
//        double a=ClassifierTools.singleTrainTestSplitAccuracy(we, trainStats, testStats);
        
        of.writeString("\n");
    }
    
}




public static void classifyOnNormalizedRange(Instances all) throws Exception{
    String resultPath="C:\\Users\\ajb\\Dropbox\\Results\\IFR Spirits\\GlobalShape\\";
    String sourcePath="C:\\Users\\ajb\\Dropbox\\IFR Spirits\\";
//    =ClassifierTools.loadData(sourcePath+"AllSamplesFiveClass");
    
    int nosBottles=44;
    int nosPerBottle=all.numClasses()*4;
    OutFile of=new OutFile(resultPath+"globalShapeFiveSampleDTW.csv");
//    System.out.println(" All Size ="+all.numInstances()+" num attributes ="+all.numAttributes());
    ArrayList<String> names =new ArrayList<>();
/*    Classifier[] c=ClassifierTools.setDefaultSingleClassifiers(names);
    of.writeString(",");
    for(String s:names)
        of.writeString(s+",");
*/        
    of.writeString(",DTW\n");
    all.sort(0);
    for(int i=0;i<1300;i++)
        all.deleteAttributeAt(1);
    NormalizeCase nc = new NormalizeCase();
    all=nc.process(all);
    for(int rep=0;rep<nosBottles;rep++){
        kNN knn=new kNN();
        knn.setDistanceFunction(new BasicDTW());
        knn.normalise(false);
        Instances train = new Instances(all);
        Instances test= new Instances(all,0);
        for(int i=0;i<nosPerBottle;i++)
            test.add(train.remove(nosPerBottle*rep));
        for(int i=1;i<nosPerBottle;i++)
            if(test.instance(i).value(0)!=test.instance(i).value(0))
                throw new Exception("INCORRECT SPLIT FOR "+test.instance(i).value(0));
        train.deleteAttributeAt(0);
        test.deleteAttributeAt(0);

  //      WeightedEnsemble we = new WeightedEnsemble();
//        names =new ArrayList<>();
//        c=ClassifierTools.setDefaultSingleClassifiers(names);
  //      WeightedEnsemble we = new WeightedEnsemble();
        of.writeString(rep+",");
        System.out.println(" Rep ="+rep);
  //      for(Classifier cl:c){
            double a=ClassifierTools.singleTrainTestSplitAccuracy(knn, train, test);
            of.writeString(a+",");
  //      }
        
        of.writeString("\n");
    }
    
}



public static Instances tidyUp(Instances all) throws Exception{
//Remove irrelevant features     
            //1: Distillery name
 //           all.deleteAttributeAt(0);
            //2: Restrict to a region 890-1050: Attributes
            System.out.println(" Number of attributes ="+all.numAttributes());
            System.out.println(" Number of instances ="+all.numInstances());
            System.out.println(" Number of classes ="+all.numClasses());
            System.out.println(" Class index ="+all.classIndex());
            int start=1300;
            for(int i=0;i<start;i++)
                all.deleteAttributeAt(1);
//Normalise
            Attribute att=all.attribute(0);
            if(att.isNominal())
                System.out.println("Att "+att+" is Nominal ");
            NormalizeCase nc = new NormalizeCase();
            Instances newInst=nc.process(all);
            System.out.println(" Number of attributes ="+all.numAttributes());
            System.out.println(" Number of instances ="+all.numInstances());
            System.out.println(" Number of classes ="+all.numClasses());
            System.out.println(" Class index ="+all.classIndex());
//Save to file
            return newInst;
            
         }
public static void debugRun(String path, String sourcePath, String filename)throws Exception{
 //   String path="Results/IFR/TwoClassDistillerySplits/";
 //   String sourcePath="IFRProblems/";
    Instances all=ClassifierTools.loadData(sourcePath+filename);
//"FiveClassV1"
        for(int i=0;i<46;i++)
        resampleExperimentSplitByDistillery(all,i,path);    
    
}

public static void desktopRun(int rep) throws Exception{
//Two class
//Five Class    
    String path="C:\\Users\\ajb\\Dropbox\\Results\\IFR Spirits\\GlobalShape\\WeightedEnsemble\\";
    String sourcePath="C:\\Users\\ajb\\Dropbox\\IFR Spirits\\FiveClassV1";
    Instances all=ClassifierTools.loadData(sourcePath);
    resampleExperimentSplitByDistillery(all,rep,path);    
}



public static void clusterRun(String[] args) throws Exception{
    int rep=Integer.parseInt(args[0])-1;
//Two class
//    String path="Results/IFR/TwoClassDistillerySplits/";
//    String sourcePath="IFRProblems/"+"TwoClassV1";
//Five Class    
    String path="Results/IFR/FiveClassDistillerySplits/";
    String sourcePath="IFRProblems/"+"FiveClassV1";
    Instances all=ClassifierTools.loadData(sourcePath);
    resampleExperimentSplitByDistillery(all,rep,path);    
}

public static void mergeBoozeFiles(String path,int nosBottles){
    OutFile of = new OutFile(path+"combinedBooze.csv");
    for(int i=0;i<nosBottles;i++){
        InFile f =new InFile(path+"booze"+i+".csv");
        of.writeLine(f.readLine());   }
}
   public static void firstExperiment(){
//   generateHeader();
    Instances all = ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\IFR Spirits\\TwoClassSpirits");
    J48 rf = new J48();
    all.deleteAttributeAt(0);
    double[][] acc;    
//    double[][] acc=ClassifierTools.crossValidationWithStats(rf, all, 10);
//   System.out.println(" C4.5 10 fold acc ="+acc[0][0]);
//Normalise cases
    NormalizeCase nc= new NormalizeCase();
    try {
 //       all=nc.process(all);
    } catch (Exception ex) {
        Logger.getLogger(Booze.class.getName()).log(Level.SEVERE, null, ex);
    }
    kNN k=new kNN(1);
    k.setCrossValidate(false);
    k.normalise(true);
    acc=ClassifierTools.crossValidationWithStats(k, all, 10);
   System.out.println(" 1-NN 10 fold acc ="+acc[0][0]);
    k=new kNN(100);
    k.setCrossValidate(true);
    k.normalise(true);
    acc=ClassifierTools.crossValidationWithStats(k, all, 10);
   System.out.println(" kNN 10 fold acc ="+acc[0][0]);
//    acc=ClassifierTools.crossValidationWithStats(new RandomForest(), all, 10);
//   System.out.println(" Rand forest 10 fold acc ="+acc[0][0]);
          
   } 
public static void main(String[] args) throws Exception{
 //   mergeBoozeFiles("C:\\Users\\ajb\\Dropbox\\Results\\IFR Spirits\\GlobalShape\\ElasticEnsemble\\",46);
    Instances all=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\IFR Spirits\\James Large\\Ethanol4Class");
    all=tidyUp(all);
    double a=resampleExperimentBasic(all,1,"C:\\Users\\ajb\\Dropbox\\IFR Spirits\\James Large\\Results", new kNN());
    
    System.exit(0);
    
    
//    classifyBySummaryStats();
 //   classifyOnNormalizedRange();
 //  System.exit(0);
//    debugRun();
 //   clusterRun(args);
 //   desktopRun(2);
}
 
}
