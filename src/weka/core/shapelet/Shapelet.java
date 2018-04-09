/*
 * A legacy shapelet class to recreate the results from the following paper:
 * Classification of Time Series by Shapelet Transformation,
 * Hills, J., Lines, J., Baranauskas, E., Mapp, J., and Bagnall, A.
 * Data Mining and Knowledge Discovery (2013)
 * 
 */
package weka.core.shapelet;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Comparator;
import utilities.class_distributions.ClassDistribution;
import utilities.class_distributions.TreeSetClassDistribution;

/**
 *
 * @author Jon Hills, j.hills@uea.ac.uk
 */
public class Shapelet implements Comparable<Shapelet>, Serializable
{

    public double separationGap;
    /*It is optional whether to store the whole shapelet or not. It is much more 
     memory efficient not to, but it does make life a little easier. 
     */
    public double[] content;
    public int length;
    public int seriesId;
    public int startPos;
    public QualityMeasures.ShapeletQualityMeasure qualityType;
    public double qualityValue;
    public boolean hasContent = true;
    boolean useSeparationGap = false;
    public double classValue;

    public void setUseSeparationGap(boolean b)
    {
        useSeparationGap = true;
    }

    public double[] getContent()
    {
        return content;
    }

    public double getQualityValue()
    {
        return qualityValue;
    }

    public int getSeriesId()
    {
        return seriesId;
    }

    public int getStartPos()
    {
        return startPos;
    }

    public void setSeriesID(int a)
    {
        seriesId = a;
    }

    public Shapelet(double[] content)
    {
        this.content = content;
        length = content.length;
    }

    public Shapelet(int seriesId, int startPos, QualityMeasures.ShapeletQualityMeasure qualityChoice)
    {
        this.seriesId = seriesId;
        this.startPos = startPos;
        this.qualityType = qualityChoice;
        this.content = null;
        length = 0;
        this.hasContent = false;
    }

    public Shapelet(double[] content, double qualValue, int seriesId, int startPos)
    {
        this.content = content;
        length = content.length;
        this.seriesId = seriesId;
        this.startPos = startPos;
        this.qualityValue = qualValue;
    }

    public Shapelet(double[] content, double qualValue, int seriesId, int startPos, double sepGap)
    {
        this.content = content;
        length = content.length;
        this.seriesId = seriesId;
        this.startPos = startPos;
        this.qualityValue = qualValue;
        this.separationGap = sepGap;
    }

    public Shapelet(double[] content, int seriesId, int startPos, QualityMeasures.ShapeletQualityMeasure qualityChoice)
    {
        this.content = content;
        length = content.length;
        this.seriesId = seriesId;
        this.startPos = startPos;
        this.qualityType = qualityChoice;
    }

    public void clearContent()
    {
        this.length = content.length;
        this.content = null;
        this.hasContent = false;
    }

    public void calculateQuality(ArrayList<OrderLineObj> orderline, ClassDistribution classDistribution)
    {
        this.qualityValue = this.qualityType.calculateQuality(orderline, classDistribution);
    }
//This also calculates the the separation gap used in the sampling routine    

    public void calcInfoGainAndThreshold(ArrayList<OrderLineObj> orderline, ClassDistribution classDistribution)
    {
            // for each split point, starting between 0 and 1, ending between end-1 and end
        // addition: track the last threshold that was used, don't bother if it's the same as the last one
        double lastDist = orderline.get(0).getDistance(); // must be initialised as not visited(no point breaking before any data!)
        double thisDist = -1;

        double bsfGain = -1;
        double threshold = -1;
        
        int numClasses = classDistribution.size();

        for (int i = 1; i < orderline.size(); i++)
        {
            thisDist = orderline.get(i).getDistance();
            if (i == 1 || thisDist != lastDist)
            { // check that threshold has moved(no point in sampling identical thresholds)- special case - if 0 and 1 are the same dist

                // count class instances below and above threshold
                
                
                ClassDistribution lessClasses = new TreeSetClassDistribution();
                ClassDistribution greaterClasses = new TreeSetClassDistribution();

                int sumOfLessClasses = 0;
                int sumOfGreaterClasses = 0;

                //visit those below threshold
                for (int j = 0; j < i; j++)
                {
                    double thisClassVal = orderline.get(j).getClassVal();
                    int storedTotal = lessClasses.get(thisClassVal);
                    storedTotal++;
                    lessClasses.put(thisClassVal, storedTotal);
                    sumOfLessClasses++;
                }

                //visit those above threshold
                for (int j = i; j < orderline.size(); j++)
                {
                    double thisClassVal = orderline.get(j).getClassVal();
                    int storedTotal = greaterClasses.get(thisClassVal);
                    storedTotal++;
                    greaterClasses.put(thisClassVal, storedTotal);
                    sumOfGreaterClasses++;
                }

                int sumOfAllClasses = sumOfLessClasses + sumOfGreaterClasses;

                double parentEntropy = QualityMeasures.InformationGain.entropy(classDistribution);

                // calculate the info gain below the threshold
                double lessFrac = (double) sumOfLessClasses / sumOfAllClasses;
                double entropyLess = QualityMeasures.InformationGain.entropy(lessClasses);
                // calculate the info gain above the threshold
                double greaterFrac = (double) sumOfGreaterClasses / sumOfAllClasses;
                double entropyGreater = QualityMeasures.InformationGain.entropy(greaterClasses);

                double gain = parentEntropy - lessFrac * entropyLess - greaterFrac * entropyGreater;

                if (gain > bsfGain)
                {
                    bsfGain = gain;
                    threshold = (thisDist - lastDist) / 2 + lastDist;
                }
            }
            lastDist = thisDist;
        }
        if (bsfGain >= 0)
        {
//                this.informationGain = bsfGain;
//                this.splitThreshold = threshold;
            this.separationGap = calculateSeparationGap(orderline, threshold);
        }
    }
    

    private double calculateSeparationGap(ArrayList<OrderLineObj> orderline, double distanceThreshold)
    {
        double sumLeft = 0;
        double leftSize = 0;
        double sumRight = 0;
        double rightSize = 0;

        for (OrderLineObj orderline1 : orderline) {
            if (orderline1.getDistance() < distanceThreshold) {
                sumLeft += orderline1.getDistance();
                leftSize++;
            } else {
                sumRight += orderline1.getDistance();
                rightSize++;
            }
        }

        if (rightSize == 0 || leftSize == 0)
        {
            return -1;
        }
        
        double thisSeparationGap = 1 / rightSize * sumRight - 1 / leftSize * sumLeft; //!!!! they don't divide by 1 in orderLine::minGap(int j)
        return thisSeparationGap;
    }

    @Override
    public int compareTo(Shapelet shapelet) {
        //compare by quality, if there quality is equal we sort by the shorter shapelets.
        int compare1 = Double.compare(qualityValue, shapelet.qualityValue);
        int compare2 = Double.compare(content.length, shapelet.content.length);
        return compare1 != 0 ? compare1 : compare2;
    }

    @Override
    public String toString()
    {
        String str = seriesId + "," + startPos + "," + length;

        return str;
    }

    public static class ReverseOrder implements Comparator<Shapelet>, Serializable
    {

        @Override
        public int compare(Shapelet s1, Shapelet s2)
        {
            //invert our compare.
            return -s1.compareTo(s2);
        }
    }

    public static class ReverseSeparationGap implements Comparator<Shapelet>, Serializable
    {

        @Override
        public int compare(Shapelet s1, Shapelet s2)
        {
            return -(new SeparationGap().compare(s1, s2));
        }
    }

    public static class SeparationGap implements Comparator<Shapelet>, Serializable
    {
        @Override
        public int compare(Shapelet s1, Shapelet s2)
        {
            int compare1 = Double.compare(s1.qualityValue, s2.qualityValue);
            if(compare1 != 0) return compare1;
            
            int compare2 = Double.compare(s1.separationGap, s2.separationGap);
            if(compare2 != 0) return compare2;
            
            int compare3 = Double.compare(s1.length, s2.length);
            return compare3;
        }

    }

}
