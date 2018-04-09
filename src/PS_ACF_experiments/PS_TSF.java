package PS_ACF_experiments;
/** * Implementation of Deng's Time Series Forest but with (truncated) PS rather 
 * than  raw data

 **/ 

import tsc_algorithms.*;
import java.util.ArrayList;
import java.util.Random;
import utilities.ClassifierTools;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.RandomTree;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;

/*
For each tree
    for each randomly selected interval (root m)
        perform PS transform
        Find summary stats
        Concatenate onto new features

Should generalise this to any transform. 
 */
public class PS_TSF extends AbstractClassifier{
    RandomTree[] trees;
    int numTrees=500;
    int numFeatures;
    int[][][] intervals;
    Random rand;
    Instances testHolder;
    public PS_TSF(){
        rand=new Random();
    }
    public PS_TSF(int seed){
        rand=new Random();
        rand.setSeed(seed);
    }
//<editor-fold defaultstate="collapsed" desc="results reported in Info Sciences paper">        
    static double[] reportedResults={
        0.2659,
        0.2302,
        0.2333,
        0.0256,
        0.2537,
        0.0391,
        0.0357,
        0.2897,
        0.2,
        0.2436,
        0.049,
        0.08,
        0.0557,
        0.2325,
        0.0227,
        0.101,
        0.1543,
        0.0467,
        0.552,
        0.6818,
        0.0301,
        0.1803,
        0.2603,
        0.0448,
        0.2237,
        0.119,
        0.0987,
        0.0865,
        0.0667,
        0.4339,
        0.233,
        0.1868,
        0.0357,
        0.1056,
        0.1116,
        0.0267,
        0.02,
        0.1177,
        0.0543,
        0.2102,
        0.2876,
        0.2624,
        0.0054,
        0.3793,
        0.1513
    };
      //</editor-fold>  
    
//<editor-fold defaultstate="collapsed" desc="problems used in Info Sciences paper">   
    static String[] problems={
            "FiftyWords",
            "Adiac",
            "Beef",
            "CBF",
            "ChlorineConcentration",
            "CinCECGtorso",
            "Coffee",
            "CricketX",
            "CricketY",
            "CricketZ",
            "DiatomSizeReduction",
            "ECG",
            "ECGFiveDays",
            "FaceAll",
            "FaceFour",
            "FacesUCR",
            "Fish",
            "GunPoint",
            "Haptics",
            "InlineSkate",
            "ItalyPowerDemand",
            "Lightning2",
            "Lightning7",
            "Mallat",
            "MedicalImages",
            "MoteStrain",
            "NonInvasiveFetalECGThorax1",
            "NonInvasiveFetalECGThorax2",
            "OliveOil",
            "OSULeaf",
            "SonyAIBORobotSurface1",
            "SonyAIBORobot Surface2",
            "StarLightCurves",
            "SwedishLeaf",
            "Symbols",
            "Synthetic Control",
            "Trace",
            "TwoLeadECG",
            "TwoPatterns",
            "UWaveGestureLibraryX",
            "UWaveGestureLibraryY",
            "UWaveGestureLibraryZ",
            "Wafer",
            "WordsSynonyms",
            "Yoga"
        };
      //</editor-fold>  
              
    public TechnicalInformation getTechnicalInformation() {
    TechnicalInformation 	result;
    result = new TechnicalInformation(TechnicalInformation.Type.ARTICLE);
    result.setValue(TechnicalInformation.Field.AUTHOR, "H. Deng, G. Runger, E. Tuv and M. Vladimir");
    result.setValue(TechnicalInformation.Field.YEAR, "2013");
    result.setValue(TechnicalInformation.Field.TITLE, "A time series forest for classification and feature extraction");
    result.setValue(TechnicalInformation.Field.JOURNAL, "Information Sciences");
    result.setValue(TechnicalInformation.Field.VOLUME, "239");
    result.setValue(TechnicalInformation.Field.PAGES, "142-153");
    
    return result;
  }

    
    
      
    @Override
    public void buildClassifier(Instances data) throws Exception {
        numFeatures=(int)Math.sqrt(data.numAttributes()-1);
        intervals =new int[numTrees][][];
        trees=new RandomTree[numTrees];
        //Set up instances size and format. 
        FastVector atts=new FastVector();
        String name;
        for(int j=0;j<numFeatures*3;j++){
                name = "F"+j;
                atts.addElement(new Attribute(name));
        }
        //Get the class values as a fast vector			
        Attribute target =data.attribute(data.classIndex());

        FastVector vals=new FastVector(target.numValues());
        for(int j=0;j<target.numValues();j++)
                vals.addElement(target.value(j));
        atts.addElement(new Attribute(data.attribute(data.classIndex()).name(),vals));
//create blank instances with the correct class value                
        Instances result = new Instances("Tree",atts,data.numInstances());
        result.setClassIndex(result.numAttributes()-1);
        for(int i=0;i<data.numInstances();i++){
            DenseInstance in=new DenseInstance(result.numAttributes());
            in.setValue(result.numAttributes()-1,data.instance(i).classValue());
            result.add(in);
        }
        
        testHolder =new Instances(result,0);       
        DenseInstance in=new DenseInstance(result.numAttributes());
        testHolder.add(in);
//For each tree         
        for(int i=0;i<numTrees;i++){
        //1. Select random intervals for tree i
                //TO DO: this may not be as published
//IN CODE:        inx = randsample(size(X,1),ceil(size(X,1)*2/2),1);%1: with replacement; 0: without replacement  

            intervals[i]=new int[numFeatures][2];  //Start and end
            for(int j=0;j<numFeatures;j++){
               intervals[i][j][0]=rand.nextInt(data.numAttributes()-1);       //Start point
               int length=rand.nextInt(data.numAttributes()-1-intervals[i][j][0]);//Min length 3
               intervals[i][j][1]=intervals[i][j][0]+length;
            }
        //2. Generate and store random attributes            
            for(int j=0;j<numFeatures;j++){
                //For each instance
                for(int k=0;k<data.numInstances();k++){
                    //extract the interval
                    double[] series=data.instance(k).toDoubleArray();
                    FeatureSet f= new FeatureSet();
                    f.setFeatures(series, intervals[i][j][0], intervals[i][j][1]);
                    result.instance(k).setValue(j*3, f.mean);
                    result.instance(k).setValue(j*3+1, f.stDev);
                    result.instance(k).setValue(j*3+2, f.slope);
                }
            }
//Set features
/*Create and build tree using all the features. Feature selection
  has already occurred
        */
            trees[i]=new RandomTree();   
            trees[i].setKValue(numFeatures);
            trees[i].buildClassifier(result);
        }
    }

    @Override
    public double classifyInstance(Instance data) throws Exception {
        int[] votes=new int[data.numClasses()];
//Build instance
        double[] series=data.toDoubleArray();
        for(int i=0;i<trees.length;i++){
            for(int j=0;j<numFeatures;j++){
                    //extract the interval
                    FeatureSet f= new FeatureSet();
                    f.setFeatures(series, intervals[i][j][0], intervals[i][j][1]);
                    testHolder.instance(0).setValue(j*3, f.mean);
                    testHolder.instance(0).setValue(j*3+1, f.stDev);
                    testHolder.instance(0).setValue(j*3+2, f.slope);
                }
            int c=(int)trees[i].classifyInstance(testHolder.instance(0));
            votes[c]++;
        }
//Return majority vote            
       int maxVote=0;
       for(int i=1;i<votes.length;i++)
           if(votes[i]>votes[maxVote])
               maxVote=i;
        return maxVote;
    }
    
    public static class DengTree extends AbstractClassifier{
        int numIntervals=20;
        int[][] boundaries; 
        @Override
        public void buildClassifier(Instances data) throws Exception {
            boundaries=new int[data.numAttributes()-1][numIntervals];
        }
    
        
        
    }
    
    public static class FeatureSet{
        double mean;
        double stDev;
        double slope;
        RandomForest r; 
        public void setFeatures(double[] data, int start, int end){
            double sumX=0,sumYY=0;
            double sumY=0,sumXY=0,sumXX=0;
            int length=end-start+1;
            for(int i=start;i<=end;i++){
                sumY+=data[i];
                sumYY+=data[i]*data[i];
                sumX+=(i-start);
                sumXX+=(i-start)*(i-start);
                sumXY+=data[i]*(i-start);
            }
            mean=sumY/length;
            stDev=sumYY-(sumY*sumY)/length;
            slope=(sumXY-(sumX*sumY)/length);
            if(sumXX-(sumX*sumX)/length!=0)
                slope/=sumXX-(sumX*sumX)/length;
            else
                slope=0;
            stDev/=length;
            if(stDev==0)    //Flat line
                slope=0;
//            else
//                stDev=Math.sqrt(stDev);
            if(slope==0)
                stDev=0;
        }
        public void setFeatures(double[] data){
            setFeatures(data,0,data.length-1);
        }
        @Override
        public String toString(){
            return "mean="+mean+" stdev = "+stDev+" slope ="+slope;
        }
    } 
    public static void main(String[] arg) throws Exception{
        FeatureSet f=new FeatureSet();
        double[] y={0,4,8,12,16};
        f.setFeatures(y);

        System.out.println(f+"");
                //Set up instances size and format. 
        FastVector atts=new FastVector();
        Instances train=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\ItalyPowerDemand\\ItalyPowerDemand_TRAIN");
        Instances test=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\ItalyPowerDemand\\ItalyPowerDemand_TEST");
        PS_TSF tsf = new PS_TSF();
        tsf.buildClassifier(train);
        System.out.println("build ok: original atts="+train.numAttributes()+" new atts ="+tsf.testHolder.numAttributes());
        double a=ClassifierTools.accuracy(test, tsf);
        System.out.println(" Accuracy ="+a);
/*
        //Get the class values as a fast vector			
        Attribute target =data.attribute(data.classIndex());

        FastVector vals=new FastVector(target.numValues());
        for(int j=0;j<target.numValues();j++)
                vals.addElement(target.value(j));
        atts.addElement(new Attribute(data.attribute(data.classIndex()).name(),vals));
//Does this create the actual instances?                
        Instances result = new Instances("Tree",atts,data.numInstances());
        for(int i=0;i<data.numInstances();i++){
            DenseInstance in=new DenseInstance(result.numAttributes());
            result.add(in);
        }
        result.setClassIndex(result.numAttributes()-1);
        Instances testHolder =new Instances(result,10);       
//For each tree   
        System.out.println("Train size "+result.numInstances());
        System.out.println("Test size "+testHolder.numInstances());
*/
    }
}
