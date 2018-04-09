package PS_ACF_experiments;
/**
 * Development code for PowerSpectrum Forest
 * 
 * 1.Fix window size by problem to a power of 2. Minimum 16, default 256 
 *      could do this at varying randomised intervals
 * 
 * 2.Randomly select at most root m start and end points. Do a PS transform, send
 * the instances to a tree. 
 * 
 * Could you the DFT coefficients. 
 * Could truncate like most do
 * 
 * 3. Build each tree.
 * 
 * Just use a random tree
 **/ 

import java.util.ArrayList;
import java.util.Random;
import utilities.ClassifierTools;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
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


 */
public class FixedIntervalForest extends AbstractClassifier{
    RandomTree[] trees;
    int numTrees=500;
    int numFeatures=256;
    public boolean setIntervalSizeThroughCV=false;
    int[] startPoints;
    Random rand;
    SimpleBatchFilter filter;
    enum Filter{PS,ACF,FFT};
    Filter f=Filter.PS;
    
    Instances testHolder;
    public FixedIntervalForest(){
        rand=new Random();
    }
    public FixedIntervalForest(int seed){
        rand=new Random();
        rand.setSeed(seed);
    }
    public void useCV(boolean f){
        setIntervalSizeThroughCV=f;
    }
    public void setWindowSize(int w){
        numFeatures=w;
        useCV(false);
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

    
    
      
    @Override
    public void buildClassifier(Instances data) throws Exception {
//Determine the window size.
        int folds=10;
        if(setIntervalSizeThroughCV){
//Clone so we can randomize the train data
            Instances temp = new Instances(data);
            temp.randomize(new Random());
            int maxSize=(data.numAttributes()-1);
//In powers of 2 from 8 to m?
            double bestAcc=0;
            int bestWindow=0;
            int incrementSize=(maxSize-4)/25;
            if(incrementSize<1)
                incrementSize=1;
            for(int i=4;i<=maxSize;i+=incrementSize){
                FixedIntervalForest f=new FixedIntervalForest();
                f.setWindowSize(i);
                    Evaluation eval=new Evaluation(data);
                    eval.crossValidateModel(f, temp, folds,new Random());
                    double e=eval.errorRate();                
                double acc=1-e;
                if(acc>bestAcc){
                    bestAcc=acc;
                    bestWindow=i;
                }
            }
            numFeatures=bestWindow;
                System.out.println("Best window size ="+numFeatures+" has CV acc "+bestAcc);
        }
        else{
            if(numFeatures>(data.numAttributes()-2))
                numFeatures=(data.numAttributes()-2);
        }            
      
//Check there are enough intervals for all the trees, otherwise its pointless
        int numIntervals=data.numAttributes()-numFeatures;
        if(numTrees>numIntervals)   //Could set up here so it enumerates them all
            numTrees=numIntervals;
        
        startPoints =new int[numTrees];
        trees=new RandomTree[numTrees];
        //Set up instances size and format. 
        FastVector atts=new FastVector();
        String name;
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
        //1. Select random start interval for tree i
            startPoints[i]=rand.nextInt(data.numAttributes()-numFeatures);
        //2. Generate and store random attributes            
                //For each instance
                for(int k=0;k<data.numInstances();k++){
                    //extract the interval
                    for(int j=0;j<numFeatures;j++){
                        result.instance(k).setValue(j, data.instance(k).value(j+startPoints[i]));
                }
        //3. Perform the PS transform
            }
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


            ACF.formChangeCombo(data);
            trees[i]=new RandomTree();   
            trees[i].setKValue(numFeatures);
            trees[i].buildClassifier(newTrain);
        }
    }

    @Override
    public double classifyInstance(Instance data) throws Exception {
        int[] votes=new int[data.numClasses()];
//Build instance
        double[] series=data.toDoubleArray();
        for(int i=0;i<trees.length;i++){
                    //extract the interval
            for(int j=0;j<numFeatures;j++){
                testHolder.instance(0).setValue(j, data.value(j+startPoints[i]));
            }
//Do the transform
            Instances temp;
            if(f==Filter.PS||f==Filter.FFT)
                temp=filter.process(testHolder);
            else
                temp=ACF.formChangeCombo(testHolder);
//            PowerSpectrum ps=new PowerSpectrum();
//            Instances temp=ps.process(testHolder);
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
    
    public static void main(String[] arg) throws Exception{
        Instances train=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\ItalyPowerDemand\\ItalyPowerDemand_TRAIN");
        Instances test=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\ItalyPowerDemand\\ItalyPowerDemand_TEST");
        FixedIntervalForest tsf = new FixedIntervalForest();
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
