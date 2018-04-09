
package weka.core.shapelet;

import java.util.ArrayList;
import java.util.TreeMap;
import weka.core.Instances;
import weka.core.shapelet.QualityMeasures.ShapeletQualityMeasure;

/**
  * Copyright: Anthony Bagnall

 * @author u0318701
 */
public class BinaryShapelet extends Shapelet{

    protected double splitThreshold;
    
    public BinaryShapelet(double[] content) {
        super(content);
        splitThreshold = -1;
    }

    public BinaryShapelet(double[] content, int seriesId, int startPos, ShapeletQualityMeasure qualityChoice) {
        super(content, seriesId, startPos, qualityChoice);
        splitThreshold = -1;
    }

    public BinaryShapelet(double[] content, int seriesId, int startPos,
            ShapeletQualityMeasure qualityChoice, double qualityValue)
    {
        super(content,seriesId,startPos,qualityChoice);
        this.qualityValue = qualityValue;
        this.splitThreshold = -1;
    }

    public double getSplitThreshold() {
        return splitThreshold;
    }
    
    public void calcInfoGainAndThreshold(ArrayList<OrderLineObj> orderline, TreeMap<Double, Integer> classDistribution){
            // for each split point, starting between 0 and 1, ending between end-1 and end
            // addition: track the last threshold that was used, don't bother if it's the same as the last one
            double lastDist = orderline.get(0).getDistance(); // must be initialised as not visited(no point breaking before any data!)
            double thisDist = -1;

            double bsfGain = -1;
            double threshold = -1;

            // check that there is actually a split point
            // for example, if all

            for(int i = 1; i < orderline.size(); i++){
                thisDist = orderline.get(i).getDistance();
                if(i==1 || thisDist != lastDist){ // check that threshold has moved(no point in sampling identical thresholds)- special case - if 0 and 1 are the same dist

                    // count class instances below and above threshold
                    TreeMap<Double, Integer> lessClasses = new TreeMap<Double, Integer>();
                    TreeMap<Double, Integer> greaterClasses = new TreeMap<Double, Integer>();

                    for(double j : classDistribution.keySet()){
                        lessClasses.put(j, 0);
                        greaterClasses.put(j, 0);
                    }

                    int sumOfLessClasses = 0;
                    int sumOfGreaterClasses = 0;

                    //visit those below threshold
                    for(int j = 0; j < i; j++){
                        double thisClassVal = orderline.get(j).getClassVal();
                        int storedTotal = lessClasses.get(thisClassVal);
                        storedTotal++;
                        lessClasses.put(thisClassVal, storedTotal);
                        sumOfLessClasses++;
                    }

                    //visit those above threshold
                    for(int j = i; j < orderline.size(); j++){
                        double thisClassVal = orderline.get(j).getClassVal();
                        int storedTotal = greaterClasses.get(thisClassVal);
                        storedTotal++;
                        greaterClasses.put(thisClassVal, storedTotal);
                        sumOfGreaterClasses++;
                    }

                    int sumOfAllClasses = sumOfLessClasses + sumOfGreaterClasses;

                    double parentEntropy = entropy(classDistribution);

                    // calculate the info gain below the threshold
                    double lessFrac =(double) sumOfLessClasses / sumOfAllClasses;
                    double entropyLess = entropy(lessClasses);
                    // calculate the info gain above the threshold
                    double greaterFrac =(double) sumOfGreaterClasses / sumOfAllClasses;
                    double entropyGreater = entropy(greaterClasses);

                    double gain = parentEntropy - lessFrac * entropyLess - greaterFrac * entropyGreater;
//                    System.out.println(parentEntropy+" - "+lessFrac+" * "+entropyLess+" - "+greaterFrac+" * "+entropyGreater);
//                    System.out.println("gain calc:"+gain);
                    if(gain > bsfGain){
                        bsfGain = gain;
                        threshold =(thisDist - lastDist) / 2 + lastDist;
                    }
                }
                lastDist = thisDist;
            }
            if(bsfGain >= 0){
                this.splitThreshold = threshold;
            }

        }
    
    private static double entropy(TreeMap<Double, Integer> classDistributions){
            if(classDistributions.size() == 1){
                return 0;
            }

            double thisPart;
            double toAdd;
            int total = 0;
            for(Double d : classDistributions.keySet()){
                total += classDistributions.get(d);
            }
            // to avoid NaN calculations, the individual parts of the entropy are calculated and summed.
            // i.e. if there is 0 of a class, then that part would calculate as NaN, but this can be caught and
            // set to 0.
            ArrayList<Double> entropyParts = new ArrayList<Double>();
            for(Double d : classDistributions.keySet()){
                thisPart =(double) classDistributions.get(d) / total;
                toAdd = -thisPart * Math.log10(thisPart) / Math.log10(2);
                if(Double.isNaN(toAdd))
                    toAdd=0;
                entropyParts.add(toAdd);
            }

            double entropy = 0;
            for(int i = 0; i < entropyParts.size(); i++){
                entropy += entropyParts.get(i);
            }
            return entropy;
        }

        
    
}
