/*
 Class to work with precalculated shapelet transforms in order to tidy it up

 */
package bakeOffExperiments;

import static bakeOffExperiments.Experiments.singleSampleExperiment;
import development.DataSets;
import fileIO.InFile;
import fileIO.OutFile;
import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import tsc_algorithms.ST_Ensemble;
import tsc_algorithms.SaveableEnsemble;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import weka.classifiers.Classifier;
import weka.classifiers.meta.timeseriesensembles.WeightedEnsemble;
import weka.core.Instances;
import weka.core.shapelet.QualityMeasures;
import weka.filters.timeseries.shapelet_transforms.BalancedClassShapeletTransform;
import weka.filters.timeseries.shapelet_transforms.classValue.BinarisedClassValue;

/**
 *
 * @author ajb
 */
public class ShapeletExperiments {
    
    public static void rename()throws Exception{
        String[] oldNames ={"fish","MALLAT","fiftywords","SonyAIBORobotSurface","SonyAIBORobotSurfaceII","wafer","yoga"};
        String[] newNames ={"Fish","Mallat","FiftyWords","SonyAIBORobotSurface1","SonyAIBORobotSurface2","Wafer","Yoga"};
//        String path="c:/Temp/";
        String path="ShapeletTransforms/BalancedClassShapeletTransform/";
        for(int i=0;i<oldNames.length;i++){
// Rename directory
            File f=new File(path+oldNames[i]);
            if(f.exists()){
                File f2=new File(path+newNames[i]);
                f.renameTo(f2);
                f=new File(path+newNames[i]);
    //Get 
                File[] list=f.listFiles();
                for(File temp:list){
                    String s=temp.getName();
                    String newS=s.replaceAll(oldNames[i], newNames[i]);
                    System.out.println("old name = "+s+" path ="+temp.getPath()+"  new name = "+newS);
                    f2=new File(path+newNames[i]+"/"+newS);
//                    f2.createNewFile();
                    boolean t=temp.renameTo(f2);
                    System.out.println("Rename? ="+t+" path ="+f.getName()+"New file "+temp.getName()+" path "+temp.getPath());
                    temp=null;
                }
            }
            else
                System.out.println("Directory  "+path+newNames[i]+" does not exist");
        }
        
        
    }
    
    public static void clusterRun(String[] args) throws Exception{
        if(args.length>1)   //
            singleSTClassifierAndFold(args);
        else
            singleSTClassifier(args);
            
    }
    public static void singleSTClassifierAndFold(String[] args) throws Exception{
//first gives the problem file  
        String classifier="ST";
        String s=args[0];        
        int fold=Integer.parseInt(args[1])-1;
//        String problem=unfinished[Integer.parseInt(args[1])-1];
//        System.out.println("Classifier ="+classifier+" problem ="+s);
        Classifier c=new ST_Ensemble();
        ((ST_Ensemble)c).doSTransform(false);
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
        //Only do it if foldfile doesnt already exist!!
        f=new File(predictions+"/fold"+fold+".csv");
        if(f.exists()==false || f.length()==0){
            OutFile p=new OutFile(predictions+"/fold"+fold+".csv");
            Instances data=ClassifierTools.loadData(DataSets.clusterPath+"ShapeletTransforms/BalancedClassShapeletTransform/"+s+"/"+s+fold+"_TRAIN"); 
            double acc=0;
            double act,pred;
    // hack here to save internal CV for furhter ensembling         
            if(c instanceof SaveableEnsemble)
               ((SaveableEnsemble)c).saveResults(predictions+"/internalCV_"+fold+".csv",predictions+"/internalTestPreds_"+fold+".csv");
            try{              
                c.buildClassifier(data);
                data=null;
                System.gc();
                data=ClassifierTools.loadData(DataSets.clusterPath+"ShapeletTransforms/BalancedClassShapeletTransform/"+s+"/"+s+fold+"_TEST"); 

                for(int j=0;j<data.numInstances();j++)
                {
                    act=data.instance(j).classValue();
                    pred=c.classifyInstance(data.instance(j));
                    if(act==pred)
                        acc++;
                    p.writeLine(act+","+pred);
                }
                acc/=data.numInstances();
                System.out.println("Problem "+s+" fold ="+fold+" acc= "+acc);
    //            of.writeString(foldAcc[i]+",");

            }catch(Exception e)
            {
                    System.out.println(" Error ="+e+" in method simpleExperiment"+e);
                    e.printStackTrace();
                    System.exit(0);
            }
        }else{
            System.out.println(" Fold ="+fold+" is already complete");            
        }
    }
    public static void countFiles(){
        String path=DataSets.clusterPath+"/ShapeletTransforms/BalancedClassShapeletTransform/";
        String resultsPath=DataSets.clusterPath+"/Results/ST/Predictions/";
        OutFile out=new OutFile(DataSets.clusterPath+"/ShapeletTransforms/STComplete.csv");
        for(String s: DataSets.fileNames){
            ArrayList<Integer> counts=new ArrayList<>();
            ArrayList<Integer> missing=new ArrayList<>();
            int complete=0;            
            long maxTrainLength=0;
            long maxTestLength=0;
            for(int i=0;i<100;i++){
                File f= new File(path+"/"+s+"/"+s+i+"_TRAIN.arff");
                File f2= new File(path+"/"+s+"/"+s+i+"_TEST.arff");
                if(f.exists() && f.length()>0 && f2.exists() && f2.length()>0){
                    counts.add(i);
                    if(f.length()>maxTrainLength)
                        maxTrainLength=f.length();
                    if(f2.length()>maxTestLength)
                        maxTestLength=f2.length();
                    File f3= new File(resultsPath+s+"/fold"+i+".csv");
                    if(f3.exists() && f3.length()>0){
                            complete++;
                    }
                }
            }
            for(int i=0;i<100;i++){
                File f= new File(path+"/"+s+"/"+s+i+"_TRAIN.arff");
                File f2= new File(path+"/"+s+"/"+s+i+"_TEST.arff");
                if(f.exists() && f.length()>0 && f2.exists() && f2.length()>0){
                    File f3= new File(resultsPath+s+"/fold"+i+".csv");
                    if(f3.exists() && f3.length()==0){
                        if(i<complete)
                            missing.add(i);
                    }
                }

                
            }            
            out.writeString(s+","+counts.size()+","+complete+","+(maxTrainLength/1000)+","+(maxTestLength/1000)+",");
            for(Integer in:missing) //Internal missing folds
                out.writeString(in+",");
            out.writeString("\n");
        }
    }
    public static void generateScripts(){
        InFile f= new InFile("C:\\Users\\ajb\\Dropbox\\Big TSC Bake Off\\New Results\\shapelet\\ST\\STComplete.csv");
        class Data{
            String name;
            int nos;
            int comp;
            int trSize;
            int teSize;
            String[] folds;
            boolean missingInner;
        }
        ArrayList<Data> ds=new ArrayList<>();
        for(int i=0;i<DataSets.fileNames.length;i++){
            Data d=new Data();
            d.name=f.readString();
            d.nos=f.readInt();
            d.comp=f.readInt();
            d.trSize=f.readInt();
            d.teSize=f.readInt();
            d.folds=f.readLine().split(",");
            if((d.nos-d.comp)> 0)
                ds.add(d);
            if(!d.folds[0].equals(""))
               d.missingInner=true; 
        }
        for(Data d:ds){
            if(d.missingInner){
                System.out.println("STRAGGLERS: length = "+d.folds.length+" first one ="+d.folds[0]+" Problem ="+d.name+" splits ="+d.nos+" complete ="+d.comp+" mem ="+d.trSize); 
                for(String str:d.folds)
                    System.out.println("MISSING : "+str);
            }
        }
//  Sort by train set max size
        Collections.sort(ds, new Comparator<Data>(){
            public int compare(Data a, Data b){
                if(a.missingInner && !b.missingInner)
                    return -1;
                if(!a.missingInner && b.missingInner)
                    return 1;
                return a.trSize-b.trSize;
            }
        });
        File dir=new File("C:\\Users\\ajb\\Dropbox\\Big TSC Bake Off\\New Results\\Scripts\\ST");
        if(!dir.exists())
            dir.mkdir();
        OutFile all=new OutFile("C:\\Users\\ajb\\Dropbox\\Big TSC Bake Off\\New Results\\Scripts\\ST\\Shapelet.txt"); 
        String queue="long";
        String javaModule="java/jdk/1.8.0_31";
//        String javaModule="java/jdk1.8.0_51";
        int MAXMEM=20000;
        int MINMEM=8000;
        CollateResults.deleteDirectory(dir);
        for(Data d:ds){
            if(d.missingInner)  //Separate job for each
            {
                for(String str:d.folds){
                    all.writeLine("bsub < Scripts/ST/"+d.name+str+".bsub"); 
                    OutFile of=new OutFile("C:\\Users\\ajb\\Dropbox\\Big TSC Bake Off\\New Results\\Scripts\\ST\\"+d.name+str+".bsub"); 
                    of.writeLine("#!/bin/csh");
                    of.writeLine("#BSUB -q "+queue);
                    of.writeLine("#BSUB -J "+d.name+"ST"+str);
                    of.writeLine("#BSUB -oo output/"+d.name+"ST"+str+".out");
                    of.writeLine("#BSUB -eo error/"+d.name+"ST"+str+".err");
                    int memUsage=4*d.trSize;
                    if(memUsage<MINMEM)
                        memUsage=MINMEM;
                    if(memUsage>MAXMEM)
                        memUsage=MAXMEM;
                    of.writeLine("#BSUB -R rusage[mem="+memUsage+"]");
                    of.writeLine("#BSUB -M "+(memUsage+1000));
                    of.writeLine("\n module add "+javaModule);
                    int s2=Integer.parseInt(str)+1;
                    of.writeLine("java -jar ST.jar "+d.name+" "+s2);
                }
            }
            else if(d.comp<d.nos){
                all.writeLine("bsub < Scripts/ST/"+d.name+"ST.bsub"); 
                OutFile of=new OutFile("C:\\Users\\ajb\\Dropbox\\Big TSC Bake Off\\New Results\\Scripts\\ST\\"+d.name+"ST.bsub"); 
                of.writeLine("#!/bin/csh");
                of.writeLine("#BSUB -q "+queue);    
                of.writeLine("#BSUB -J "+d.name+"ST["+(d.comp+1)+"-"+d.nos+"]");
                of.writeLine("#BSUB -oo output/"+d.name+"ST%I.out");
                of.writeLine("#BSUB -eo error/"+d.name+"ST%I.err");
                int memUsage=2000*(d.trSize/1000);
                if(memUsage<MINMEM)
                    memUsage=MINMEM;
                if(memUsage>MAXMEM)
                    memUsage=MAXMEM;
                of.writeLine("#BSUB -R rusage[mem="+memUsage+"]");
                of.writeLine("#BSUB -M "+(memUsage+1000));
                of.writeLine("\n module add "+javaModule);
                of.writeLine("java -jar ST.jar "+d.name+" $LSB_JOBINDEX");
            }
            
        }
    }
    
    public static void singleSTClassifier(String[] args) throws Exception{
//first gives the problem file  
        String classifier="ST";
        String s=DataSets.fileNames[Integer.parseInt(args[0])-1];        
//        String problem=unfinished[Integer.parseInt(args[1])-1];
        System.out.println("Classifier ="+classifier+" problem ="+s);
        Classifier c=new ST_Ensemble();
        ((ST_Ensemble)c).doSTransform(false);
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
        folds=resampleSTExperiment(s,c,100,of,predictions);
        of.writeString("\n");
    }
    public static double[] resampleSTExperiment(String s,Classifier c, int resamples,OutFile of,String preds){
        Instances train,test;
       double[] foldAcc=new double[resamples];
        for(int i=0;i<resamples;i++){
            File f=new File(preds+"/fold"+i+".csv");
            if(!f.exists() || f.length()==0){
                train=ClassifierTools.loadData(DataSets.clusterPath+"ShapeletTransforms/BalancedClassShapeletTransform/"+s+"/"+s+i+"_TRAIN"); 
                test=ClassifierTools.loadData(DataSets.clusterPath+"ShapeletTransforms/BalancedClassShapeletTransform/"+s+"/"+s+i+"_TEST"); 
                foldAcc[i]=singleSampleSTExperiment(train,test,c,i,preds);
                of.writeString(foldAcc[i]+",");
            }
            else
                of.writeString(",");
            train=null;
            test=null;
            System.gc();
        }            
         return foldAcc;
    }
    public static double singleSampleSTExperiment(Instances train, Instances test, Classifier c, int sample,String preds){
        Instances[] data={train,test};
        double acc=0;
        double act,pred;
        OutFile p=new OutFile(preds+"/fold"+sample+".csv");
// hack here to save internal CV for furhter ensembling         
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
                System.out.println(" TRAIN "+train.relationName()+" has "+train.numAttributes()+" attributes and "+train.numInstances()+" instances");
                System.out.println(" TEST "+test.relationName()+" has "+test.numAttributes()+" attributes"+test.numInstances()+" instances");

                System.exit(0);
        }
         return acc;
    }
     
    public static void main(String[] args) throws Exception{
        singleWorms();
        System.exit(0);
//        rename();
//        basicTest();
//        generateScripts();
//        countFiles();
        try{
            if(args.length>0){ //Cluster run
            System.out.println("ARGS[0] ="+args[0]+" ARGS LENGTH ="+args.length);
            DataSets.resultsPath=DataSets.clusterPath+"Results/";
            DataSets.problemPath=DataSets.clusterPath+"TSC Problems/";
            clusterRun(args);
         }
         else{    
                
            }   
        }catch(Exception e){
            System.out.println("Exception thrown, caught in ShapeletExperiments main="+e);
            System.exit(0);
        }
    }
    public static void basicTest() throws Exception{
//        Instances train=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\Temp\\BeetleFly\\BeetleFly0_TRAIN");
//        Instances test=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\Temp\\BeetleFly\\BeetleFly0_TEST");
//        WeightedEnsemble we=new WeightedEnsemble();
//        we.setWeightType("prop");
//        double acc=ClassifierTools.singleTrainTestSplitAccuracy(we, train, test);
//        System.out.println(" FROM FILE ACC ="+acc);
  
        Instances train2=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\Temp\\BeetleFly_TRAIN");
        Instances test2=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\Temp\\BeetleFly_TEST");
//        BalancedClassShapeletTransform transform=new BalancedClassShapeletTransform();
//        transform = new BalancedClassShapeletTransform();
        BalancedClassShapeletTransform transform=new BalancedClassShapeletTransform();
        transform.setClassValue(new BinarisedClassValue());
        transform.useCandidatePruning();
        transform.setNumberOfShapelets(train2.numInstances() * 10);
        transform.setShapeletMinAndMax(3, train2.numAttributes() - 1);
        transform.setQualityMeasure(QualityMeasures.ShapeletQualityChoice.INFORMATION_GAIN);
        transform.turnOffLog();
        Instances temp=transform.process(train2);
        OutFile out1=new OutFile("C:\\Users\\ajb\\Dropbox\\Temp\\TrainNonNormed1.arff");
        out1.writeLine(temp.toString());
        temp=transform.process(test2);
        OutFile out2=new OutFile("C:\\Users\\ajb\\Dropbox\\Temp\\TrainNonNormed1.arff");
        out2.writeLine(temp.toString());

        
        ST_Ensemble st=new ST_Ensemble();
        st.doSTransform(true);

        double acc2=ClassifierTools.singleTrainTestSplitAccuracy(st, train2, test2);
        System.out.println(" WITH TRANSFORM ACC ="+acc2);
        
    }
    public static void singleWorms(){
        Instances train=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\Temp\\Worms7_TRAIN");
        Instances test=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\Temp\\Worms7_TEST");
        WeightedEnsemble we=new WeightedEnsemble();
        we.setWeightType("prop");
        we.saveTrainCV("C:\\Users\\ajb\\Dropbox\\Big TSC Bake Off\\New Results\\shapelet\\ST\\Predictions\\Worms\\"+"internalCV_7.csv");
        we.saveTestPreds("C:\\Users\\ajb\\Dropbox\\Big TSC Bake Off\\New Results\\shapelet\\ST\\Predictions\\Worms\\"+"internalTestPreds_7.csv");
        double acc=ClassifierTools.singleTrainTestSplitAccuracy(we, train, test);
        System.out.println(" FROM FILE ACC ="+acc);
        
    }
    
}
