package weka.classifiers.lazy;
import java.util.HashMap;
import utilities.ClassifierTools;
import weka.classifiers.Classifier;
import weka.core.*;
import weka.core.elastic_distance_measures.*;

/* This class is a specialisation of kNN that can only be used with the efficient DTW distance
 * 
 * The reason for specialising is this class has the option of searching for the optimal window length
 * through a grid search of values.
 * 
 * By default this class does a search. 
 * To search for the window size call
 * optimiseWindow(true);
 * By default, this does a leave one out cross validation on every possible window size, then sets the 
 * proportion to the one with the largest accuracy. This will be slow. Speed it up by
 * 
 * 1. Set the max window size to consider by calling
 * setMaxWindowSize(double r) where r is on range 0..1, with 1 being a full warp, 0 being Euclidean distance.
 * 
 * 2. Set the increment size 
 * setIncrementSize(int s) where s is on range 1...trainSetSize 
 * 
 * This is a basic brute force implementation, not optimised! There are probably ways of 
 * incrementally doing this. It could further be speeded up by using PAA to reduce the dimensionality first.
 * 
 */

public class DTW_1NN implements Classifier {
    private boolean optimiseWindow=false;
    private double windowSize=1;
    private int maxNosWindows=100;
    private Instances train;
    private int trainSize;
    private int bestWarp;
    private int maxWindowSize;
    DTW_DistanceBasic dtw;
    HashMap<Integer,Double> distances;
    double maxR=1;

        public DTW_1NN(){
        dtw=new DTW();
    }
        public double getMaxR(){ return maxR;}
    public void setMaxNosWindows(int a){maxNosWindows=a;}
    public void optimiseWindow(boolean b){ optimiseWindow=b;}
    public void setR(double r){dtw.setR(r);}
    public double getR(){ return dtw.getR();}
    public int getBestWarp(){ return bestWarp;}
    public int getWindowSize(){ return dtw.getWindowSize(train.numAttributes()-1);}

    @Override
    public void buildClassifier(Instances d){
        train=d;
        trainSize=d.numInstances();
        if(optimiseWindow){
            maxR=0;
            double maxAcc=0;
/*Set the maximum warping window: The window size in the r value is range 0..1, 
so the window size= r*(length of series). For implementation reasons, a window size of 1 
is equivalent to Euclidean distance (rather than a window size of 0            
            */
            int dataLength=train.numAttributes()-1;
            if(dataLength<maxNosWindows)
                maxNosWindows=dataLength;

            for(int i=maxNosWindows;i>0;i-=1){
                //Set r for current value sa the precentage of series length.
//                dtw=new DTW();
                dtw.setR(i/100.0);

/*Can do an early abandon inside cross validate? If it cannot be more accurate 
 than maxR even with some left to evaluate then yes
*/                
                double acc=crossValidateAccuracy(maxAcc);
                if(acc>=maxAcc){
                    maxR=i;
                    maxAcc=acc;
               }
//                System.out.println(" r="+i+" warpsize ="+x+" train acc= "+acc+" best acc ="+maxR);
               if(maxWindowSize<(i-1)*dataLength/100){
                   System.out.println("WINDOW SIZE ="+dtw.getWindowSize()+" Can reset downwards at "+i+"% to ="+((int)(100*(maxWindowSize/(double)dataLength))));
                   i=(int)(100*(maxWindowSize/(double)dataLength));
                   i++;
//                   i=Math.round(100*(maxWindowSize/(double)dataLength))/100;
               } 

            }
            bestWarp=(int)(maxR*dataLength/100);
            System.out.println("OPTIMAL WINDOW ="+maxR+" % which gives a warp of"+bestWarp+" data");
  //          dtw=new DTW();
            dtw.setR(maxR/100.0);
        }
    }
    @Override
    public double classifyInstance(Instance d){
/*Basic distance, with early abandon. This is only for 1-nearest neighbour*/
            double minSoFar=Double.MAX_VALUE;
            double dist; int index=0;
            for(int i=0;i<train.numInstances();i++){
                    dist=dtw.distance(train.instance(i),d,minSoFar);
                    if(dist<minSoFar){
                            minSoFar=dist;
                            index=i;
                    }
            }
            return train.instance(index).classValue();
    }
    @Override
    public double[] distributionForInstance(Instance instance){
        double[] dist=new double[instance.numClasses()];
        dist[(int)classifyInstance(instance)]=1;
        return dist;
    }

    
    /**Could do this by calculating the distance matrix, but then 	
 * you cannot use the early abandon. Early abandon about doubles the speed,
 * as will storing the distances. Given the extra n^2 memory, probably better
 * to just use the early abandon. We could store those that were not abandoned?
answer is to store those without the abandon in a hash table indexed by i and j,
*where index i,j == j,i

* @return 
 */
    private  double crossValidateAccuracy(double maxAcc){
        double a=0,d, minDist;
        int nearest;
        Instance inst;
        int bestNosCorrect=(int)(maxAcc*trainSize);
        maxWindowSize=0;
        int w;
        distances=new HashMap<>(trainSize);
        
        
        for(int i=0;i<trainSize;i++){
//Find nearest to element i
            nearest=0;
            minDist=Double.MAX_VALUE;
            inst=train.instance(i);
            for(int j=0;j<trainSize;j++){
                if(i!=j){
//  d=dtw.distance(inst,train.instance(j),minDist);
//Store past distances if not early abandoned 
//Not seen i,j before                    
                  if(j>i){
                        d=dtw.distance(inst,train.instance(j),minDist);
                        //Store if not early abandon
                        if(d!=Double.MAX_VALUE){
//                            System.out.println(" Storing distance "+i+" "+j+" d="+d+" with key "+(i*trainSize+j));
                            distances.put(i*trainSize+j,d);
//                            storeCount++;
                        }
//Else if stored recover                        
                    }else if(distances.containsKey(j*trainSize+i)){
                        d=distances.get(j*trainSize+i);
//                       System.out.println(" Recovering distance "+i+" "+j+" d="+d);
//                        recoverCount++;
                    }
//Else recalculate with new early abandon                    
                    else{
                        d=dtw.distance(inst,train.instance(j),minDist);
                    }        
                    if(d<minDist){
                        nearest=j;
                        minDist=d;
                        w=dtw.findMaxWindow();
                        if(w>maxWindowSize)
                            maxWindowSize=w;
                    }
                }
            }
                //Measure accuracy for nearest to element i			
            if(inst.classValue()==train.instance(nearest).classValue())
                a++;
           //Early abandon if it cannot be better than the best so far. 
            if(a+trainSize-i<bestNosCorrect){
//                    System.out.println(" Early abandon on CV when a="+a+" and i ="+i+" best nos correct = "+bestNosCorrect+" maxAcc ="+maxAcc+" train set size ="+trainSize);
                return 0.0;
            }
        }
//        System.out.println("trainSize ="+trainSize+" stored ="+storeCount+" recovered "+recoverCount);
        return a/(double)trainSize;
    }
    public static void main(String[] args){
            DTW_1NN c = new DTW_1NN();
            String path="C:\\Research\\Data\\Time Series Data\\Time Series Classification\\";

            Instances test=ClassifierTools.loadData(path+"Coffee\\Coffee_TEST.arff");
            Instances train=ClassifierTools.loadData(path+"Coffee\\Coffee_TRAIN.arff");
            train.setClassIndex(train.numAttributes()-1);
            c.buildClassifier(train);

    }

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

}
