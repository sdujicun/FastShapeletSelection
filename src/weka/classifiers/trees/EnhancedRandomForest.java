/*
This classifier is enhanced so that classifier builds a random forest with the 
facility to build by forward selection addition of trees to minimize OOB error.    
Further enhanced to include OOB error estimates and predictions
 */
package weka.classifiers.trees;

import development.DataSets;
import java.text.DecimalFormat;
import java.util.Random;
import tsc_algorithms.TSBF;
import utilities.ClassifierTools;
import weka.classifiers.meta.Bagging;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

/**
 *
 * @author ajb
 */
public class EnhancedRandomForest extends RandomForest{

    public EnhancedRandomForest(){
        super();
        m_numTrees=50;
        
    }
    double OOBError;
    double[][] OOBPredictions;
/*This 
    */    
    @Override
    public void buildClassifier(Instances data) throws Exception{
    // can classifier handle the data?
        getCapabilities().testWithFail(data);

        // remove instances with missing class
        data = new Instances(data);
        data.deleteWithMissingClass();
        m_bagger = new EnhancedBagging();
        RandomTree rTree = new RandomTree();

        // set up the random tree options
        m_KValue = m_numFeatures;
        if (m_KValue < 1) m_KValue = (int) Utils.log2(data.numAttributes())+1;
        rTree.setKValue(m_KValue);
        rTree.setMaxDepth(getMaxDepth());

        // set up the bagger and build the forest
        m_bagger.setClassifier(rTree);
        m_bagger.setSeed(m_randomSeed);
        m_bagger.setNumIterations(m_numTrees);
        m_bagger.setCalcOutOfBag(true);
        m_bagger.setNumExecutionSlots(m_numExecutionSlots);
        m_bagger.buildClassifier(data);
        ((EnhancedBagging)m_bagger).findOOBProbabilities();
        
    }

    public void addTrees(int n, Instances data) throws Exception{
        EnhancedBagging newTrees =new EnhancedBagging();
        RandomTree rTree = new RandomTree();
        // set up the random tree options
        m_KValue = m_numFeatures;
        rTree.setKValue(m_KValue);
        rTree.setMaxDepth(getMaxDepth());
//Change this
        Random r= new Random();
        newTrees.setSeed(r.nextInt());
        newTrees.setClassifier(rTree);
        newTrees.setNumIterations(n);
        newTrees.setCalcOutOfBag(true);
        newTrees.setNumExecutionSlots(m_numExecutionSlots);
        newTrees.buildClassifier(data);
        newTrees.findOOBProbabilities();
//Merge with previous
        m_bagger.aggregate(newTrees);
        m_bagger.finalizeAggregation();
//Update OOB Error, as this is seemingly not done in the bagger
        
        m_numTrees+=n;
        m_bagger.setNumIterations(m_numTrees); 
        
        ((EnhancedBagging)m_bagger).mergeBaggers(newTrees);
    }
    public double getBaggingPercent(){
      return m_bagger.getBagSizePercent();
    }
    private class EnhancedBagging extends Bagging{
// 
        @Override
        public void buildClassifier(Instances data)throws Exception {
            super.buildClassifier(data);
            m_data=data;
//            System.out.println(" RESET BAGGER");

        }
        double[][] OOBProbabilities;
        int[] counts;
        public void mergeBaggers(EnhancedBagging other){
            for (int i = 0; i < m_data.numInstances(); i++) {
                for (int j = 0; j < m_data.numClasses(); j++) {
                      OOBProbabilities[i][j]=counts[i]*OOBProbabilities[i][j]+other.counts[i]*other.OOBProbabilities[i][j];
                      OOBProbabilities[i][j]/=counts[i]+other.counts[i];
                }
                counts[i]=counts[i]+other.counts[i];
            }
//Merge  m_inBags index i is classifier, j the instance
            boolean[][] inBags = new boolean[m_inBag.length+other.m_inBag.length][];
            for(int i=0;i<m_inBag.length;i++)
                inBags[i]=m_inBag[i];
            for(int i=0;i<other.m_inBag.length;i++)
                inBags[m_inBag.length+i]=m_inBag[i];
            m_inBag=inBags;
            findOOBError();
        }
        public void findOOBProbabilities() throws Exception{
            OOBProbabilities=new double[m_data.numInstances()][m_data.numClasses()];
            counts=new int[m_data.numInstances()];
            for (int i = 0; i < m_data.numInstances(); i++) {
                for (int j = 0; j < m_Classifiers.length; j++) {
                    if (m_inBag[j][i])
                      continue;
                    counts[i]++;
                    double[] newProbs = m_Classifiers[j].distributionForInstance(m_data.instance(i));
                // average the probability estimates
                    for (int k = 0; k < m_data.numClasses(); k++) {
                        OOBProbabilities[i][k] += newProbs[k];
                    }
                }
                for (int k = 0; k < m_data.numClasses(); k++) {
                    OOBProbabilities[i][k] /= counts[i];
                }
            }
        }
        
        public double findOOBError(){
            double correct = 0.0;
            for (int i = 0; i < m_data.numInstances(); i++) {
                double[] probs = OOBProbabilities[i];
                int vote =0;
                for (int j = 1; j < probs.length; j++) {
                  if(probs[vote]<probs[j])
                      vote=j;
            }
            if(m_data.instance(i).classValue()==vote) 
                correct++;
            }
            m_OutOfBagError=1- correct/(double)m_data.numInstances();
//            System.out.println(" NEW OOB ERROR ="+m_OutOfBagError);
            return m_OutOfBagError;
        }
        
 //       public double getOOBError
    }
    public double[][] findOOBProbabilities() throws Exception{
        ((EnhancedBagging)m_bagger).findOOBProbabilities();
        return ((EnhancedBagging)m_bagger).OOBProbabilities;
    }
    public double[][] getOBProbabilities() throws Exception{
        return ((EnhancedBagging)m_bagger).OOBProbabilities;
    }
  
  
  
    public static void main(String[] args) {
        
  //      testBinMaker();
  //      System.exit(0);
        DecimalFormat df = new DecimalFormat("##.###");
        try{
                String s="SwedishLeaf";
                System.out.println(" PROBLEM ="+s);
                Instances train=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\"+s+"\\"+s+"_TRAIN");
                Instances test=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\"+s+"\\"+s+"_TEST");
                EnhancedRandomForest rf=new EnhancedRandomForest();
               rf.buildClassifier(train);
                System.out.println(" bag percent ="+rf.getBaggingPercent()+" OOB error "+rf.measureOutOfBagError());
                for(int i=0;i<5;i++){
                    System.out.println(" Number f trees ="+rf.getNumTrees()+" num elements ="+rf.numElements());
                    System.out.println(" bag percent ="+rf.getBaggingPercent()+" OOB error "+rf.measureOutOfBagError());
                    double[][] probs=rf.findOOBProbabilities();
/*s
                    for (int j = 0; j < probs.length; j++) {
                        double[] prob = probs[j];
                        for (int k = 0; k < prob.length; k++) {
                            System.out.print(","+prob[k]);
                        }
                        System.out.println("");
                        
                    }
*/
                    rf.addTrees(50, train);
                }
                int correct=0;
                for(Instance ins:test){
                    double[] pred=rf.distributionForInstance(ins);
                    double cls=rf.classifyInstance(ins);
                    if(cls==ins.classValue())
                        correct++;
                }
                System.out.println(" ACC = "+((double)correct)/test.numInstances());
//                System.out.println(" calc out of bag? ="+rf.m_bagger.m_CalcOutOfBag);
                System.exit(0);
                double a =ClassifierTools.singleTrainTestSplitAccuracy(rf, train, test);
                System.out.println(" error ="+df.format(1-a));
//                tsbf.buildClassifier(train);
 //               double c=tsbf.classifyInstance(test.instance(0));
 //               System.out.println(" Class ="+c);
        }catch(Exception e){
            System.out.println("Exception "+e);
            e.printStackTrace();
            System.exit(0);
        }
    }
  
}
