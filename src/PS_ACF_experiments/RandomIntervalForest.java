package PS_ACF_experiments;
/**
 * Development code for PowerSpectrum Forest
 * 1. set number of trees to max(500,m)
 * 2. Set the first tree to the full interval
 * 2. Randomly select the interval length and start point for each other tree * 3. Build each tree.

 **/ 

import utilities.SaveCVAccuracy;
import fileIO.OutFile;
import java.util.ArrayList;
import java.util.Random;
import utilities.ClassifierTools;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.RandomTree;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import weka.filters.SimpleBatchFilter;
import weka.filters.timeseries.ACF;
import weka.filters.timeseries.FFT;
import weka.filters.timeseries.PowerSpectrum;

/*
To save:
1. Folds in RIF/Predictions/<problem>/foldi.csv 
    actual, predicted
2. CV in RIF/Predictions/<problem>/InternalCVi.csv



 */
public class RandomIntervalForest extends AbstractClassifier implements SaveCVAccuracy{
    Classifier[] trees;
    int numTrees=500;
//INTERVAL BOUNDS ARE INCLUSIVE    
    int[] startPoints;
    int[] endPoints;
    public static int MIN_INTERVAL=4;
    Random rand;
    SimpleBatchFilter filter;
    boolean saveResults=false;//Cannot do this unless the strings below are set
/* Train results are overwritten with each call to buildClassifier
    File opened on this path.   */    
    String trainCV;
    
    public void setCVPath(String train){
        saveResults=true;
        trainCV=train;
    }
    
    enum Filter{PS,ACF,FFT};
    Filter f=Filter.PS;
    public void setFilter(String s){
        String str=s.toUpperCase();
        switch(str){
            case "FFT": case "DFT": case "FOURIER":
              f=Filter.FFT;
                break;
            case "ACF": case "AFC": case "AUTOCORRELATION":
              f=Filter.ACF;                
                break;
            case "PS": case "POWERSPECTRUM":
              f=Filter.PS;
                break;
        }
    }
    Instances[] testHolders;
    public RandomIntervalForest(){
        rand=new Random();
    }
    public RandomIntervalForest(int seed){
        rand=new Random();
        rand.setSeed(seed);
    }
    public TechnicalInformation getTechnicalInformation() {
    TechnicalInformation 	result;
    result = new TechnicalInformation(TechnicalInformation.Type.ARTICLE);
    result.setValue(TechnicalInformation.Field.AUTHOR, "A. Bagnall");
    result.setValue(TechnicalInformation.Field.YEAR, "2016");
    result.setValue(TechnicalInformation.Field.TITLE, "Not published");
    result.setValue(TechnicalInformation.Field.JOURNAL, "NA");
    result.setValue(TechnicalInformation.Field.VOLUME, "NA");
    result.setValue(TechnicalInformation.Field.PAGES, "NA");
    
    return result;
  }

    public String getParameters(){
        return "numTrees,"+numTrees+","+"MinInterval"+MIN_INTERVAL;
    }
    
    
      
    @Override
    public void buildClassifier(Instances data) throws Exception {
        if(saveResults){
            int folds=setNumberOfFolds(data);
            OutFile of=new OutFile(trainCV);
           of.writeLine("RIF");
    //Estimate train accuracy HERE
            RandomIntervalForest rif=new RandomIntervalForest();
            Evaluation eval=new Evaluation(data);
            eval.crossValidateModel(rif,data,folds,rand);            
            double acc=1-eval.errorRate();
            System.out.println("CV acc ="+acc);
            of.writeLine(acc+"");
        }
        
//Determine the number of trees
        if(data.numAttributes()-1>numTrees)
            numTrees=data.numAttributes()-1;
//Set series length
        int m=data.numAttributes()-1;
        startPoints =new int[numTrees];
        endPoints =new int[numTrees];
        trees=new Classifier[numTrees];
        testHolders=new Instances[numTrees];
        //1. Select random intervals for each tree
        for(int i=0;i<numTrees;i++){
            if(i==0){//Do whole series
                startPoints[i]=0;
                endPoints[i]=m-1;
            }
            else{
                startPoints[i]=rand.nextInt(m-MIN_INTERVAL);
                if(startPoints[i]==m-1-MIN_INTERVAL) 
//Interval at the end, need to avoid calling nextInt with argument 0
                    endPoints[i]=m-1;
                else{    
                    endPoints[i]=rand.nextInt(m-startPoints[i]);
                    if(endPoints[i]<MIN_INTERVAL)
                        endPoints[i]=MIN_INTERVAL;
                    endPoints[i]+=startPoints[i];
                }
            }
//            System.out.println("START = "+startPoints[i]+" END ="+endPoints[i]);
//Set up train instances and save format for testing. 
            int numFeatures=endPoints[i]-startPoints[i];
            String name;
            FastVector atts=new FastVector();
            for(int j=0;j<numFeatures;j++){
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
            for(int j=0;j<data.numInstances();j++){
                DenseInstance in=new DenseInstance(result.numAttributes());
                double[] v=data.instance(j).toDoubleArray();
                for(int k=0;k<numFeatures;k++)
                    in.setValue(k,v[startPoints[i]+k]);
//Set interval features                
                in.setValue(result.numAttributes()-1,data.instance(j).classValue());
                result.add(in);
            }
            testHolders[i] =new Instances(result,0);       
            DenseInstance in=new DenseInstance(result.numAttributes());
            testHolders[i].add(in);
//Perform the transform
            Instances newTrain;
            switch(f){
                case ACF:
                    filter=new ACF();
                    newTrain=ACF.formChangeCombo(result);
                    break;
                case FFT:
                    filter=new FFT();
                newTrain=filter.process(result);
                case PS: default:
                    filter=new PowerSpectrum();
                newTrain=filter.process(result);
                    break;

            }             
//Build Classifier
            trees[i]=new RandomTree();   
            ((RandomTree)trees[i]).setKValue(numFeatures);
            trees[i]=new J48();   
            trees[i].buildClassifier(newTrain);
        }
    }

    @Override
    public double classifyInstance(Instance ins) throws Exception {
        int[] votes=new int[ins.numClasses()];
////Build instance
        double[] series=ins.toDoubleArray();
        for(int i=0;i<trees.length;i++){
            int numFeatures=endPoints[i]-startPoints[i];
        //extract the interval
            for(int j=0;j<numFeatures;j++){
                testHolders[i].instance(0).setValue(j, ins.value(j+startPoints[i]));
            }
//Do the transform
            Instances temp;
            if(f==Filter.PS||f==Filter.FFT)
                temp=filter.process(testHolders[i]);
            else
                temp=ACF.formChangeCombo(testHolders[i]);
            int c=(int)trees[i].classifyInstance(temp.instance(0));
            votes[c]++;
        }
//Return majority vote            
       int maxVote=0;
       for(int i=1;i<votes.length;i++)
           if(votes[i]>votes[maxVote])
               maxVote=i;
       return maxVote;
    }
    
    public static void intervalGenerationTest(){
        int m=500;
        int numTrees=500;
        int[] startPoints =new int[numTrees];
        int[] endPoints =new int[numTrees];
        Random rand=new Random();
        for(int i=0;i<numTrees;i++){
            if(i==0){//Do whole series
                startPoints[i]=0;
                endPoints[i]=m-1;
            }
            else{
                startPoints[i]=rand.nextInt(m-MIN_INTERVAL);
                if(startPoints[i]==m-1-MIN_INTERVAL) 
//Interval at the end, need to avoid calling nextInt with argument 0
                    endPoints[i]=m-1;
                else{    
                    endPoints[i]=rand.nextInt(m-startPoints[i]);
                    if(endPoints[i]<MIN_INTERVAL)
                        endPoints[i]=MIN_INTERVAL;
                    endPoints[i]+=startPoints[i];
                }
            }
            System.out.println("START = "+startPoints[i]+" END ="+endPoints[i]+ " LENGTH ="+(endPoints[i]-startPoints[i]));
        }
    }
    public static void main(String[] arg) throws Exception{
//        intervalGenerationTest();
//        System.exit(0);
        
        Instances train=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\ItalyPowerDemand\\ItalyPowerDemand_TRAIN");
        Instances test=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\ItalyPowerDemand\\ItalyPowerDemand_TEST");
        RandomIntervalForest rif = new RandomIntervalForest();
        rif.setCVPath("C:\\Users\\ajb\\Dropbox\\Spectral Interval Experiments\\RIF\\Predictions\\InternalCV0.csv");

        
        rif.buildClassifier(train);
        System.out.println("build ok:");
        double a=ClassifierTools.accuracy(test, rif);
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
