package weka.core.shapelet;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.TreeMap;
import utilities.class_distributions.ClassDistribution;
import utilities.class_distributions.TreeSetClassDistribution;

/**
 *      * copyright: Anthony Bagnall
 *
 * A class to store shapelet quality measure implementations. This includes an
 * abstract quality measure class, and implementations of each of the four
 * shapelet quality measures used in:
 * <p>
 * Jason Lines , Anthony Bagnall, Alternative quality measures for time series
 * shapelets, Proceedings of the 13th international conference on Intelligent
 * Data Engineering and Automated Learning, August 29-31, 2012, Natal, Brazil
 * <p>
 * and
 * <p>
 * Jason Lines , Luke M. Davis , Jon Hills , Anthony Bagnall, A shapelet
 * transform for time series classification, Proceedings of the 18th ACM SIGKDD
 * international conference on Knowledge discovery and data mining, August
 * 12-16, 2012, Beijing, China
 *
 * @author Jason Lines
 */
public class QualityMeasures implements Serializable
{

    /**
     * An enum for selecting the quality measure to use in the filter for
     * selecting the k best shapelets.
     * <p>
     * The choices include: Information Gain (KDD12), F-Stat, (KDD12),
     * Kruskal-Wallis (IDEAL12), and Mood's Median (IDEAL12)
     */
    public enum ShapeletQualityChoice
    {

        /**
         * Used to specify that the filter will use Information Gain as the
         * shapelet quality measure (introduced in Ye & Keogh 2009)
         */
        INFORMATION_GAIN,
        /**
         * Used to specify that the filter will use F-Stat as the shapelet
         * quality measure (introduced in Lines et. al 2012)
         */
        F_STAT,
        /**
         * Used to specify that the filter will use Kruskal-Wallis as the
         * shapelet quality measure (introduced in Lines and Bagnall 2012)
         */
        KRUSKALL_WALLIS,
        /**
         * Used to specify that the filter will use Mood's Median as the
         * shapelet quality measure (introduced in Lines and Bagnall 2012)
         */
        MOODS_MEDIAN
    }

    public interface ShapeletQualityMeasure 
    {
        public double calculateQuality(ArrayList<OrderLineObj> orderline, ClassDistribution classDistribution);
    }

    /**
     * A class for calculating the information gain of a shapelet, according to
     * the set of distances from the shapelet to a dataset.
     */
    public static class InformationGain implements ShapeletQualityMeasure, Serializable 
    {

        /**
         * A method to calculate the quality of a FullShapeletTransform, given
         * the orderline produced by computing the distance from the shapelet to
         * each element of the dataset.
         *
         * @param orderline the pre-computed set of distances for a dataset to a
         * single shapelet
         * @param classDistribution the distibution of all possible class values
         * in the orderline
         * @return a measure of shapelet quality according to information gain
         */
        @Override
        public double calculateQuality(ArrayList<OrderLineObj> orderline, ClassDistribution classDistribution)
        {
            Collections.sort(orderline);
            // for each split point, starting between 0 and 1, ending between end-1 and end
            // addition: track the last threshold that was used, don't bother if it's the same as the last one
            double lastDist = -1;//orderline.get(0).getDistance(); // must be initialised as not visited(no point breaking before any data!)
            double thisDist = -1;

            double bsfGain = -1;

            // initialise class counts
            ClassDistribution lessClasses = new TreeSetClassDistribution();
            ClassDistribution greaterClasses = new TreeSetClassDistribution();

            // parent entropy will always be the same, so calculate just once
            double parentEntropy = entropy(classDistribution);

            int sumOfAllClasses = 0;
            for (double j : classDistribution.keySet())
            {
                lessClasses.put(j, 0);
                greaterClasses.put(j, classDistribution.get(j));
                sumOfAllClasses += classDistribution.get(j);
            }
            int sumOfLessClasses = 0;
            int sumOfGreaterClasses = sumOfAllClasses;

            double thisClassVal;
            int oldCount;

            for (OrderLineObj ol : orderline)
            {
                thisDist = ol.getDistance();

                //move the threshold along one (effectively by adding this dist to lessClasses
                thisClassVal = ol.getClassVal();
                oldCount = lessClasses.get(thisClassVal) + 1;
                lessClasses.put(thisClassVal, oldCount);
                oldCount = greaterClasses.get(thisClassVal) - 1;
                greaterClasses.put(thisClassVal, oldCount);

                // adjust counts - maybe makes more sense if these are called counts, rather than sums!
                sumOfLessClasses++;
                sumOfGreaterClasses--;

                // check to see if the threshold has moved (ie if thisDist isn't the same as lastDist)
                // important, else gain calculations will be made 'in the middle' of a threshold, resulting in different info gain for
                // the split point, that won't actually be valid as it is 'on' a distances, rather than 'between' them/
                if (thisDist != lastDist)
                {

                    // calculate the info gain below the threshold
                    double lessFrac = (double) sumOfLessClasses / sumOfAllClasses;
                    double entropyLess = entropy(lessClasses);

                    // calculate the info gain above the threshold
                    double greaterFrac = (double) sumOfGreaterClasses / sumOfAllClasses;
                    double entropyGreater = entropy(greaterClasses);

                    double gain = parentEntropy - lessFrac * entropyLess - greaterFrac * entropyGreater;
                    if (gain > bsfGain)
                    {
                        bsfGain = gain;
                    }
                }
                lastDist = thisDist;
            }
            return bsfGain;
        }

        public static double entropy(ClassDistribution classDistributions)
        {
            if (classDistributions.size() == 1)
            {
                return 0;
            }

            double thisPart;
            double toAdd;
            int total = 0;
            //Aaron: should be simpler than iterating using the keySet.
            //Values is backed by the Map so it doesn't need to be constructed.
            Collection<Integer> values = classDistributions.values();
            for (Integer d : values)
            {
                total += d;
            }

            // to avoid NaN calculations, the individual parts of the entropy are calculated and summed.
            // i.e. if there is 0 of a class, then that part would calculate as NaN, but this can be caught and
            // set to 0. 
            //Aaron:  Instead of using the keyset to loop through, use the underlying Array to iterate through, ordering of calculations doesnt matter.
            //just that we do them all. so i think previously it was n log n, now should be just n.
            double entropy = 0;
            for (Integer d : values)
            {
                thisPart = (double) d / total;
                toAdd = -thisPart * Math.log10(thisPart) / Math.log10(2);
                //Aaron: if its not NaN we can add it, if it was NaN we'd just add 0.
                if (!Double.isNaN(toAdd))
                {
                    entropy += toAdd;
                }
            }

            return entropy;
        }

    }

    /**
     * A class for calculating the F-Statistic of a shapelet, according to the
     * set of distances from the shapelet to a dataset.
     */
    public static class FStat implements ShapeletQualityMeasure, Serializable
    {

        /**
         * A method to calculate the quality of a FullShapeletTransform, given
         * the orderline produced by computing the distance from the shapelet to
         * each element of the dataset.
         *
         * @param orderline the pre-computed set of distances for a dataset to a
         * single shapelet
         * @param classDistribution the distibution of all possible class values
         * in the orderline
         * @return a measure of shapelet quality according to f-stat
         */
        @Override
        public double calculateQuality(ArrayList<OrderLineObj> orderline, ClassDistribution classDistribution)
        {
            Collections.sort(orderline);
            int numClasses = classDistribution.size();
            int numInstances = orderline.size();

            double[] sums = new double[numClasses];
            double[] sumsSquared = new double[numClasses];
            double[] sumOfSquares = new double[numClasses];

            for (int i = 0; i < numClasses; i++)
            {
                sums[i] = 0;
                sumsSquared[i] = 0;
                sumOfSquares[i] = 0;
            }

            for (OrderLineObj orderline1 : orderline)
            {
                int c = (int) orderline1.getClassVal();
                double thisDist = orderline1.getDistance();
                sums[c] += thisDist;
                sumOfSquares[c] += thisDist * thisDist;
            }

            for (int i = 0; i < numClasses; i++)
            {
                sumsSquared[i] = sums[i] * sums[i];
            }

            double ssTotal = 0;
            double part1 = 0;
            double part2 = 0;

            for (int i = 0; i < numClasses; i++)
            {
                part1 += sumOfSquares[i];
                part2 += sums[i];
            }

            part2 *= part2;
            part2 /= numInstances;
            ssTotal = part1 - part2;

            double ssAmoung = 0;
            part1 = 0;
            part2 = 0;
            for (int i = 0; i < numClasses; i++)
            {
                part1 += (double) sumsSquared[i] / classDistribution.get((double) i);//.data[i].size();
                part2 += sums[i];
            }
            ssAmoung = part1 - (part2 * part2) / numInstances;
            double ssWithin = ssTotal - ssAmoung;

            int dfAmoung = numClasses - 1;
            int dfWithin = numInstances - numClasses;

            double msAmoung = ssAmoung / dfAmoung;
            double msWithin = ssWithin / dfWithin;

            double f = msAmoung / msWithin;

            return Double.isNaN(f) ? 0.0 : f;
        }

        /**
         *
         * @param orderline
         * @param classDistribution
         * @return a va
         */
        public double calculateQualityNew(ArrayList<OrderLineObj> orderline, TreeMap<Double, Integer> classDistribution)
        {
            Collections.sort(orderline);
            int numClasses = classDistribution.size();
            int numInstances = orderline.size();

            double[] sums = new double[numClasses];
            double[] sumsSquared = new double[numClasses];
            double[] sumOfSquares = new double[numClasses];

            for (int i = 0; i < orderline.size(); i++)
            {
                int c = (int) orderline.get(i).getClassVal();
                double thisDist = orderline.get(i).getDistance();
                sums[c] += thisDist;
                sumOfSquares[c] += thisDist * thisDist;
            }

            double ssTotal = 0;
            double part1 = 0;
            double part2 = 0;

            for (int i = 0; i < numClasses; i++)
            {
                sumsSquared[i] = sums[i] * sums[i];
                part1 += sumOfSquares[i];
                part2 += sums[i];
            }

            part2 *= part2;
            part2 /= numInstances;
            ssTotal = part1 - part2;

            double ssAmoung = 0;
            part1 = 0;
            part2 = 0;
            for (int i = 0; i < numClasses; i++)
            {
                part1 += (double) sumsSquared[i] / classDistribution.get((double) i);//.data[i].size();
                part2 += sums[i];
            }
            ssAmoung = part1 - (part2 * part2) / numInstances;
            double ssWithin = ssTotal - ssAmoung;

            int dfAmoung = numClasses - 1;
            int dfWithin = numInstances - numClasses;

            double msAmoung = ssAmoung / dfAmoung;
            double msWithin = ssWithin / dfWithin;

            double f = msAmoung / msWithin;

            return f;
        }
    }

    /**
     * A class for calculating the Mood's Median statistic of a shapelet,
     * according to the set of distances from the shapelet to a dataset.
     */
    public static class MoodsMedian implements ShapeletQualityMeasure, Serializable
    {

        /**
         * A method to calculate the quality of a FullShapeletTransform, given
         * the orderline produced by computing the distance from the shapelet to
         * each element of the dataset.
         *
         * @param orderline the pre-computed set of distances for a dataset to a
         * single shapelet
         * @param classDistributions the distibution of all possible class
         * values in the orderline
         * @return a measure of shapelet quality according to Mood's Median
         */
        @Override
        public double calculateQuality(ArrayList<OrderLineObj> orderline, ClassDistribution classDistributions)
        {

            //naive implementation as a benchmark for finding median - actually faster than manual quickSelect! Probably due to optimised java implementation
            Collections.sort(orderline);
            int lengthOfOrderline = orderline.size();
            double median;
            if (lengthOfOrderline % 2 == 0)
            {
                median = (orderline.get(lengthOfOrderline / 2 - 1).getDistance() + orderline.get(lengthOfOrderline / 2).getDistance()) / 2;
            }
            else
            {
                median = orderline.get(lengthOfOrderline / 2).getDistance();
            }

            int totalCount = orderline.size();
            int countBelow = 0;
            int countAbove = 0;
            int numClasses = classDistributions.size();
            int[] classCountsBelowMedian = new int[numClasses];
            int[] classCountsAboveMedian = new int[numClasses];

            double distance;
            double classVal;
            int countSoFar;
            for (OrderLineObj orderline1 : orderline)
            {
                distance = orderline1.getDistance();
                classVal = orderline1.getClassVal();
                if (distance < median)
                {
                    countBelow++;
                    classCountsBelowMedian[(int) classVal]++;
                }
                else
                {
                    countAbove++;
                    classCountsAboveMedian[(int) classVal]++;
                }
            }

            double chi = 0;
            double expectedAbove, expectedBelow;
            for (int i = 0; i < numClasses; i++)
            {
                expectedBelow = (double) (countBelow * classDistributions.get((double) i)) / totalCount;
                chi += ((classCountsBelowMedian[i] - expectedBelow) * (classCountsBelowMedian[i] - expectedBelow)) / expectedBelow;

                expectedAbove = (double) (countAbove * classDistributions.get((double) i)) / totalCount;
                chi += ((classCountsAboveMedian[i] - expectedAbove)) * (classCountsAboveMedian[i] - expectedAbove) / expectedAbove;
            }

            if (Double.isNaN(chi))
            {
                chi = 0; // fix for cases where the shapelet is a straight line and chi is calc'd as NaN
            }
            return chi;
        }

    }

    /**
     * A class for calculating the Kruskal-Wallis statistic of a shapelet,
     * according to the set of distances from the shapelet to a dataset.
     */
    public static class KruskalWallis implements ShapeletQualityMeasure, Serializable
    {

        /**
         * A method to calculate the quality of a FullShapeletTransform, given
         * the orderline produced by computing the distance from the shapelet to
         * each element of the dataset.
         *
         * @param orderline the pre-computed set of distances for a dataset to a
         * single shapelet
         * @param classDistribution the distibution of all possible class values
         * in the orderline
         * @return a measure of shapelet quality according to Kruskal-Wallis
         */
        @Override
        public double calculateQuality(ArrayList<OrderLineObj> orderline, ClassDistribution classDistribution)
        {
            // sort
            Collections.sort(orderline);
            int numClasses = classDistribution.size();
            int[] classRankCounts = new int[numClasses];
            double[] classRankMeans = new double[numClasses];

            double lastDistance = orderline.get(0).getDistance();
            double thisDistance = lastDistance;
            double classVal = orderline.get(0).getClassVal();
            classRankCounts[(int) classVal] += 1;

            int duplicateCount = 0;

            for (int i = 1; i < orderline.size(); i++)
            {
                thisDistance = orderline.get(i).getDistance();
                if (duplicateCount == 0 && thisDistance != lastDistance)
                { // standard entry
                    classRankCounts[(int) orderline.get(i).getClassVal()] += i + 1;

                }
                else if (duplicateCount > 0 && thisDistance != lastDistance)
                { // non-duplicate following duplicates
                    // set ranks for dupicates

                    double minRank = i - duplicateCount;
                    double maxRank = i;
                    double avgRank = (minRank + maxRank) / 2;

                    for (int j = i - duplicateCount - 1; j < i; j++)
                    {
                        classRankCounts[(int) orderline.get(j).getClassVal()] += avgRank;
                    }

                    duplicateCount = 0;
                    // then set this rank
                    classRankCounts[(int) orderline.get(i).getClassVal()] += i + 1;
                }
                else
                {// thisDistance==lastDistance
                    if (i == orderline.size() - 1)
                    { // last one so must do the avg ranks here (basically copied from above, BUT includes this element too now)

                        double minRank = i - duplicateCount;
                        double maxRank = i + 1;
                        double avgRank = (minRank + maxRank) / 2;

                        for (int j = i - duplicateCount - 1; j <= i; j++)
                        {
                            classRankCounts[(int) orderline.get(j).getClassVal()] += avgRank;
                        }
                    }
                    duplicateCount++;
                }
                lastDistance = thisDistance;
            }

            //3) overall mean rank
            double overallMeanRank = (1.0 + orderline.size()) / 2;

            //4) sum of squared deviations from the overall mean rank
            double s = 0;
            for (int i = 0; i < numClasses; i++)
            {
                classRankMeans[i] = (double) classRankCounts[i] / classDistribution.get((double) i);
                s += classDistribution.get((double) i) * (classRankMeans[i] - overallMeanRank) * (classRankMeans[i] - overallMeanRank);
            }

            //5) weight s with the scale factor
            double h = 12.0 / (orderline.size() * (orderline.size() + 1)) * s;

            return h;
        }
    }
}
