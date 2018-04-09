/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package weka.filters.timeseries.shapelet_transforms;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

import development.DataSets;
import utilities.ClassifierTools;
import weka.core.Instances;
import weka.core.shapelet.OrderLineObj;
import weka.core.shapelet.QualityBound;
import weka.core.shapelet.QualityMeasures;
import weka.core.shapelet.Shapelet;
import static weka.filters.timeseries.shapelet_transforms.FullShapeletTransform.removeSelfSimilar;

/**
 *
 * @author raj09hxu
 */
public class ShapeletTransform extends FullShapeletTransform {

    //generate all the 3 length sub sequences for all series normalising and storing.
    @Override
    public ArrayList<Shapelet> findBestKShapeletsCache(Instances data) {
        ArrayList<Shapelet> kShapelets = new ArrayList<>();
        ArrayList<Shapelet> windowShapelets;

        //for all time series
       // outputPrint("Processing data: ");

        int dataSize = data.numInstances();

        //for the shapelet lengths
        for (int length = minShapeletLength; length <= maxShapeletLength; length++) {
            windowShapelets = new ArrayList<>();

            //outputPrint("length : " + length);

            double[][][] normalisedSubSeqs = generateNormalisedSubSequences(data, length);

            //changed to pass in the worst of the K-Shapelets.
            worstShapelet = kShapelets.size() == numShapelets ? kShapelets.get(numShapelets - 1) : null;

            //for all possible time series.
            for (int i = 0; i < normalisedSubSeqs.length; i++) {
                //set the shapelets class value.                   
                classValue.setShapeletValue(data.get(i));

                //for all possible starting positions of that length
                for (int j = 0; j < normalisedSubSeqs[i].length; j++) {
                    double[] candidate = normalisedSubSeqs[i][j];

                    //Initialize bounding algorithm for current candidate
                    //QualityBound.ShapeletQualityBound qualityBound = initializeQualityBound(classValue.getClassDistributions());

                    //Set bound of the bounding algorithm
                    if (qualityBound != null && worstShapelet != null) {
                        qualityBound.setBsfQuality(worstShapelet.qualityValue);
                    }

                    Shapelet candidateShapelet = checkCandidate(data, normalisedSubSeqs, candidate, i, j, qualityBound);

                    if (candidateShapelet != null) {
                        windowShapelets.add(candidateShapelet);
                    }
                }
            }

            //TODO: write the SELF SIMILAR CODE.
            //Collections.sort(windowShapelets, shapeletComparator);
            //windowShapelets = removeSelfSimilar(windowShapelets);
            kShapelets = combine(numShapelets, kShapelets, windowShapelets);
        }

        this.numShapelets = kShapelets.size();
        recordShapelets(kShapelets, this.ouputFileLocation);
        // printShapelets(kShapelets);

        return kShapelets;
    }

    //this generates the set of subsequences and normalises them for a set length sliding window.
    private double[][][] generateNormalisedSubSequences(Instances data, int length) {
        //create a 2D matrix of normalised series.
        //number of attributes in our series - length - 1 is the number of subsequences we'll have. -1 for class val.
        double[][][] normalisedSubSeqs = new double[data.numInstances()][data.numAttributes() - length - 1][];

        //for all possible time series.
        for (int i = 0; i < normalisedSubSeqs.length; i++) {
            //get the series
            double[] wholeCandidate = data.instance(i).toDoubleArray();

            //for all possible starting positions of that length
            for (int j = 0; j < normalisedSubSeqs[i].length; j++) {
                double[] candidate = new double[length];
                System.arraycopy(wholeCandidate, j, candidate, 0, length);

                // znorm candidate here so it's only done once, rather than in each distance calculation
                candidate = subseqDistance.zNormalise(candidate, false);

                //stor the candidate
                normalisedSubSeqs[i][j] = candidate;
            }
        }

        return normalisedSubSeqs;
    }

    private Shapelet checkCandidate(Instances data, double[][][] normalisedSubSeqs, double[] candidate, int seriesId, int startPos, QualityBound.ShapeletQualityBound qualityBound) {

        ArrayList<OrderLineObj> orderline = new ArrayList<>();

        //compare our candidate to the other series and generate the orderline.
        for (int i = 0; i < normalisedSubSeqs.length; i++) {
            //Check if it is possible to prune the candidate
            if (qualityBound != null && qualityBound.pruneCandidate()) {
                return null;
            }

            double distance = 0.0;
            if (i != seriesId) {
                distance = calculateDistance(normalisedSubSeqs[i], candidate);
            }

            //this could be binarised or normal. 
            double classVal = classValue.getClassValue(data.instance(i));

            // without early abandon, it is faster to just add and sort at the end
            orderline.add(new OrderLineObj(distance, classVal));

            if (qualityBound != null) {
                qualityBound.updateOrderLine(orderline.get(orderline.size() - 1));
            }
        }

        // create a shapelet object to store all necessary info, i.e.
        Shapelet shapelet = new Shapelet(candidate, dataSourceIDs[seriesId], startPos, this.qualityMeasure);
        //this class distribution could be binarised or normal.
        shapelet.calculateQuality(orderline, classValue.getClassDistributions());
        shapelet.classValue = classValue.getShapeletValue(); //set classValue of shapelet. (interesing to know).
        return shapelet;
    }

    //pass in the series with the normalised subseqs.
    private double calculateDistance(double[][] normalisedSubSeqs, double[] candidate) {
        double temp;
        double bestSum = Double.MAX_VALUE;

        //compare our candidate to all the candidates in the series, finding the best distance.
        for (double[] normalisedSubSeq : normalisedSubSeqs) {
            double sum = 0;
            //euclidean distance between two subs of the same length.
            for (int j = 0; j < candidate.length; j++) {
                temp = (candidate[j] - normalisedSubSeq[j]);
                sum = sum + (temp * temp);
            }
            if (sum < bestSum) {
                bestSum = sum;
            }
        }

        return (bestSum == 0.0) ? 0.0 : (1.0 / candidate.length * bestSum);
    }

    public static void main(String[] args) {
        final String dotdotSlash = ".." + File.separator;
        String adiacLocation = dotdotSlash + dotdotSlash + "resampled data sets" + File.separator + "ItalyPowerDemand" + File.separator + "ItalyPowerDemand99";

        Instances train = utilities.ClassifierTools.loadData(adiacLocation + "_TRAIN");
//        String s="ItalyPowerDemand";
//		Instances train = ClassifierTools.loadData(DataSets.problemPath + s + "\\" + s + "_TRAIN");
//		Instances test = ClassifierTools.loadData(DataSets.problemPath + s + "\\" + s + "_TEST"); 
        /*ShapeletTransform transform = new ShapeletTransform();
        transform.setNumberOfShapelets(train.numInstances() * 10);
        transform.setShapeletMinAndMax(3, train.numAttributes() - 1);
        transform.supressOutput();
        transform.setQualityMeasure(QualityMeasures.ShapeletQualityChoice.INFORMATION_GAIN);

        long startTime = System.nanoTime();
        transform.process(train);
        long finishTime = System.nanoTime();
        System.out.println((finishTime - startTime));*/

        FullShapeletTransform transform1 = new FullShapeletTransform();
        transform1.setNumberOfShapelets(train.numInstances() * 10);
        transform1.setShapeletMinAndMax(3, train.numAttributes() - 1);
        transform1.setQualityMeasure(QualityMeasures.ShapeletQualityChoice.INFORMATION_GAIN);

        //startTime = System.nanoTime();
        transform1.process(train);
        //finishTime = System.nanoTime();
        //System.out.println((finishTime - startTime));

    }
}
