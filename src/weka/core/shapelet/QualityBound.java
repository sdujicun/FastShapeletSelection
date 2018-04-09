package weka.core.shapelet;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.ListIterator;
import java.util.Map;
import java.util.TreeMap;
import utilities.class_distributions.ClassDistribution;
import utilities.class_distributions.SimpleClassDistribution;
import utilities.class_distributions.TreeSetClassDistribution;

/**
 * A class to store shapelet quality measure bounding implementations.
 * 
 * @author Edgaras Baranauskas
 */
public class QualityBound{
    
    /**
     * Abstract parent class for the shapelet quality measure bound implementations
     */
    public abstract static class ShapeletQualityBound implements Serializable{

        /**
         * Best quality observed so far, which is used for determining if the 
         * candidate can be pruned
         */
        protected double bsfQuality;
        /**
         * Orderline of the observed distance, class pairs
         */
        protected ArrayList<OrderLineObj> orderLine;
        /**
         * Class distribution of the observed distance, class pairs
         */
        protected ClassDistribution orderLineClassDist;
        /**
         * Class distribution of the dataset, which currently being processed
         */
        protected ClassDistribution parentClassDist;
        /**
         * Number of instances in the dataset, which is currently being processed
         */
        protected int numInstances;
        /**
         * The percentage of data point that must be in the observed orderline 
         * before the bounding mechanism is can be invoked
         */
        protected int percentage;
        
        /**
         * 
         * @param classDist
         * @param percentage
         */
        protected void initParentFields(ClassDistribution classDist, int percentage){
            //Initialize the fields
            bsfQuality = Double.MAX_VALUE;
            orderLine = new ArrayList<>();
            orderLineClassDist = new TreeSetClassDistribution();
            parentClassDist = classDist;
            this.percentage = percentage;
            
            //Initialize orderline class distribution
            numInstances = 0;
            for(Double key : parentClassDist.keySet()){
                orderLineClassDist.put(key, 0);
                numInstances += parentClassDist.get(key);
            }
        }
        
        
        /**
         * Method to set the best quality so far of the shapelet
         * @param quality quality of the best so far quality observed
         */
        public void setBsfQuality(double quality){
            bsfQuality = quality;
        }
                
        /**
         * Method to update the ShapeletQualityBound with newly observed OrderLineObj
         * @param orderLineObj newly observed OrderLineObj
         */
        public void updateOrderLine(OrderLineObj orderLineObj){
            //Update classDistribution of unprocessed elements
            orderLineClassDist.put(orderLineObj.getClassVal(), orderLineClassDist.get(orderLineObj.getClassVal()) + 1);  
            
            //Update sorted orderline
            if(orderLine.isEmpty()){
                orderLine.add(orderLineObj);
            }else{
                boolean added = false;
                ListIterator<OrderLineObj> iterator = orderLine.listIterator();
                while(iterator.hasNext() && !added){
                    OrderLineObj current = iterator.next();
                    if(orderLineObj.compareTo(current) < 0 || orderLineObj.compareTo(current) == 0){
                        iterator.previous();
                        iterator.add(orderLineObj);
                        added = true;
                    }
                }
                
                if(!added){
                    orderLine.add(orderLineObj);
                }
            }
        }
        
        /**
         * Method to calculate the quality bound for the current orderline
         * @return quality bound for the current orderline
         */
        protected abstract double calculateBestQuality();
                
        /**
         * Method to check if the current candidate can be pruned
         * @return true if candidate can be pruned otherwise false
         */
        public boolean pruneCandidate(){
            //Check if the required amont of data has been observed and
            //best quality so far set
            if(bsfQuality == Double.MAX_VALUE || orderLine.size() * 100 / numInstances <= percentage){
                return false;
            }
 
            //The precondition is met, so quality bound can be computed
            return calculateBestQuality() <= bsfQuality;
        }                
    }
    
    /**
     * A class for calculating the information gain bound of a shapelet, according to 
     * the set of distances from the shapelet to a dataset.
     */
    public static class InformationGainBound extends QualityBound.ShapeletQualityBound{
        private double parentEntropy;
        boolean isExact;
        
        /**
         * Constructor to construct InformationGainBound
         * @param classDist class distribution of the data currently being processed
         * @param percentage percentage of data required to be processed before
         *                   bounding mechanism is used.
         *
         * @param isExact
         * */
        public InformationGainBound(ClassDistribution classDist, int percentage, boolean isExact){
            initParentFields(classDist, percentage);
            this.isExact = isExact;
            parentEntropy = QualityMeasures.InformationGain.entropy(parentClassDist);
        }
        public InformationGainBound(ClassDistribution classDist, int percentage){
            this(classDist,percentage,false);
        }
           
        /**
         * Method to calculate the quality bound for the current orderline
         * @return information gain bound
         */
        @Override
        protected double calculateBestQuality(){
            TreeMap<Double, Boolean> perms = new TreeMap<>();
            double bsfGain = -1;
                        
            //Cycle through all permutations
            if(isExact){
                //Initialise perms
                for(Double key : orderLineClassDist.keySet()){
                    perms.put(key, Boolean.TRUE);
                }
            
                for(int totalCycles = perms.keySet().size(); totalCycles > 1; totalCycles--){
                    for(int cycle = 0; cycle < totalCycles; cycle++){
                        int start = 0, count = 0;
                        for(Double key : perms.keySet()){
                            Boolean val = Boolean.TRUE;
                            if(cycle == start){
                                val = Boolean.FALSE;
                                int size = perms.keySet().size();
                                if(totalCycles <  size && count < (size - totalCycles)){
                                    count++;
                                    start--;
                                }
                            }
                            perms.put(key, val);
                            start++;
                        }
                        //Check quality of current permutation
                        double currentGain = computeIG(perms);

                        if(currentGain > bsfGain){
                            bsfGain = currentGain;
                        }

                        if(bsfGain > bsfQuality){
                            System.out.println("cycles left: " + (totalCycles - cycle));
                            break;
                        }

                    }
                }
            }else{
                double currentGain = computeIG(null);
                    
                if(currentGain > bsfGain){
                    bsfGain = currentGain;
                }
            }
        
            return bsfGain;
        }
        
        private double computeIG(Map<Double, Boolean> perm){
            //Initialise class counts
            TreeSetClassDistribution lessClasses = new TreeSetClassDistribution();
            TreeSetClassDistribution greaterClasses = new TreeSetClassDistribution();
            TreeMap<Double, Boolean> isShifted = new TreeMap<>();
            
            int countOfAllClasses = 0;
            int countOfLessClasses = 0;
            int countOfGreaterClasses = 0;
            
            for(double j : parentClassDist.keySet()){
                int lessVal =0;
                int greaterVal = parentClassDist.get(j);
                
                if(perm != null){
                    if(perm.get(j) != null && perm.get(j)){
                        lessVal = parentClassDist.get(j) - orderLineClassDist.get(j);
                        greaterVal = orderLineClassDist.get(j);
                    }
                    countOfLessClasses += lessClasses.get(j);
                
               //Assign everything to the right for fast bound
                }else{
                    isShifted.put(j, Boolean.FALSE);
                }
                
                lessClasses.put(j, lessVal);
                greaterClasses.put(j, greaterVal);
                countOfGreaterClasses += greaterClasses.get(j);
                
                
                countOfAllClasses += parentClassDist.get(j);
            }
           

            double bsfGain = -1;
            double lastDist = -1;
            double thisDist;
            double thisClassVal;
            int oldCount;
                   
            for(int i = 0; i < orderLine.size()-1; i++){ 
                thisDist = orderLine.get(i).getDistance();
                thisClassVal = orderLine.get(i).getClassVal();

                 //move the threshold along one (effectively by adding this dist to lessClasses
                oldCount = lessClasses.get(thisClassVal)+1;
                lessClasses.put(thisClassVal,oldCount);                
                oldCount = greaterClasses.get(thisClassVal)-1;
                greaterClasses.put(thisClassVal,oldCount);
                
                // adjust counts - maybe makes more sense if these are called counts, rather than sums!
                countOfLessClasses++;
                countOfGreaterClasses--;

                //For fast bound dynamically shift the unassigned objects when majority side changes
                if(!isExact){
                    //Check if shift has not already happened
                    if(!isShifted.get(thisClassVal)){
                        int greaterCount = greaterClasses.get(thisClassVal) - (parentClassDist.get(thisClassVal) - orderLineClassDist.get(thisClassVal));
                        int lessCount = lessClasses.get(thisClassVal);
                        
                        //Check if shift has happened
                        if(lessCount - greaterCount > 0){
                            greaterClasses.put(thisClassVal, greaterClasses.get(thisClassVal) - (parentClassDist.get(thisClassVal) - orderLineClassDist.get(thisClassVal)));
                            countOfGreaterClasses -= parentClassDist.get(thisClassVal) - orderLineClassDist.get(thisClassVal);
                            lessClasses.put(thisClassVal, lessClasses.get(thisClassVal) + (parentClassDist.get(thisClassVal) - orderLineClassDist.get(thisClassVal)));
                            countOfLessClasses += parentClassDist.get(thisClassVal) - orderLineClassDist.get(thisClassVal);
                            isShifted.put(thisClassVal, Boolean.TRUE);
                        }
                    }
                }
                
                // check to see if the threshold has moved (ie if thisDist isn't the same as lastDist)
                // important, else gain calculations will be made 'in the middle' of a threshold, resulting in different info gain for
                // the split point, that won't actually be valid as it is 'on' a distances, rather than 'between' them/
                if(thisDist != lastDist){

                    // calculate the info gain below the threshold
                    double lessFrac =(double) countOfLessClasses / countOfAllClasses;
                    double entropyLess = QualityMeasures.InformationGain.entropy(lessClasses);

                    // calculate the info gain above the threshold
                    double greaterFrac =(double) countOfGreaterClasses / countOfAllClasses;
                    double entropyGreater = QualityMeasures.InformationGain.entropy(greaterClasses);

                    double gain = parentEntropy - lessFrac * entropyLess - greaterFrac * entropyGreater;
                    if(gain > bsfGain){
                        bsfGain = gain;
                    }
                }
                
                lastDist = thisDist;
            }
            
            return bsfGain;
        }
        
        
        @Override
        public boolean pruneCandidate(){
            //Check if we at least have observed an object from each class
            if(orderLine.size() % parentClassDist.size() != 0){
                return false;
            }else{
                return super.pruneCandidate();
            }
        }          
    }
    
    /**
     * A class for calculating the moods median statistic bound of a shapelet, according to 
     * the set of distances from the shapelet to a dataset.
     */
    public static class MoodsMedianBound extends QualityBound.ShapeletQualityBound{  
        
        /**
         * Constructor to construct MoodsMedianBound
         * @param classDist class distribution of the data currently being processed
         * @param percentage percentage of data required to be processed before
         *                   bounding mechanism is used.
         */
        public MoodsMedianBound(ClassDistribution classDist, int percentage){
            initParentFields(classDist, percentage);
        }
                
        /**
         * Method to calculate the quality bound for the current orderline
         * @return Moods Median statistic bound
         */
        @Override
        protected double calculateBestQuality(){
            int lengthOfOrderline = orderLine.size();
            double median;
            if(lengthOfOrderline%2==0){
                median = (orderLine.get(lengthOfOrderline/2-1).getDistance()+orderLine.get(lengthOfOrderline/2).getDistance())/2;
            }else{
                median = orderLine.get(lengthOfOrderline/2).getDistance();
            }

            int totalCount = orderLine.size();
            int countBelow = 0;
            int countAbove = 0;
            int numClasses = parentClassDist.size();
            
            ClassDistribution classCountsBelowMedian = new SimpleClassDistribution(numClasses);
            ClassDistribution classCountsAboveMedian = new SimpleClassDistribution(numClasses);

            double distance;
            double classVal;
            
            // Count observed class distributions above and below the median
            for (OrderLineObj orderLine1 : orderLine) {
                distance = orderLine1.getDistance();
                classVal = orderLine1.getClassVal();
                if(distance < median){
                    countBelow++;
                    
                    classCountsBelowMedian.addTo(classVal, 1); //increment by 1
                }else{
                    countAbove++;
                    classCountsAboveMedian.addTo(classVal, 1);
                }
            }
            
            // Add count of predicted class distributions above and below the median
            for(double key : orderLineClassDist.keySet()){
                int predictedCount = parentClassDist.get(key) - orderLineClassDist.get(key);
                if(classCountsBelowMedian.get(key) <= classCountsAboveMedian.get(key)){
                    classCountsAboveMedian.addTo(key, predictedCount);
                    countAbove += predictedCount;
                }else{
                    classCountsBelowMedian.addTo(key, predictedCount);
                    countBelow += predictedCount;
                }
                totalCount += predictedCount;
            }
            
            double chi = 0;
            double expectedAbove = 0, expectedBelow;
            for(int i = 0; i < numClasses; i++){
                expectedBelow = (double)(countBelow*parentClassDist.get((double)i))/totalCount;
                double classCountsBelow = classCountsBelowMedian.get(i) - expectedBelow;
                double classCountsAbove = classCountsAboveMedian.get(i) - expectedAbove;
                chi += (classCountsBelow*classCountsBelow)/expectedBelow;

                expectedAbove = (double)(countAbove*parentClassDist.get((double)i))/totalCount;
                chi += (classCountsAbove*classCountsAbove)/expectedAbove;
            }

            if(Double.isNaN(chi)){
                chi = 0; // fix for cases where the shapelet is a straight line and chi is calc'd as NaN
            }
            return chi;
        }
        
        @Override
        public boolean pruneCandidate(){
            if(orderLine.size() % parentClassDist.size() != 0){//if(orderLine.size() < parentClassDist.size()){
                return false;
            }else{
                return super.pruneCandidate();
            }
        }    
    }   
    
    
    /**
     * A class for calculating the f-stat statistic bound of a shapelet, according to 
     * the set of distances from the shapelet to a dataset.
     */
    public static class FStatBound extends QualityBound.ShapeletQualityBound{  
        
        private double[] sums;        
        private double[] sumsSquared;
        private double[] sumOfSquares;
        private ArrayList<OrderLineObj> meanDistOrderLine;
       
        private double minDistance;
        private double maxDistance;
        
        /**
         * Constructor to construct FStatBound
         * @param classDist class distribution of the data currently being processed
         * @param percentage percentage of data required to be processed before
         *                   bounding mechanism is used.
         */
        public FStatBound(ClassDistribution classDist, int percentage){
            initParentFields(classDist, percentage);
          
            int numClasses = parentClassDist.size();
            sums = new double[numClasses];
            sumsSquared = new double[numClasses];
            sumOfSquares = new double[numClasses];
            meanDistOrderLine = new ArrayList<OrderLineObj>(numClasses);
            minDistance = -1.0;
            maxDistance = -1.0;
        }
        
        @Override
        public void updateOrderLine(OrderLineObj orderLineObj){
            super.updateOrderLine(orderLineObj);
            
            int c = (int) orderLineObj.getClassVal();
            double thisDist = orderLineObj.getDistance();
            sums[c] += thisDist;
            sumOfSquares[c] += thisDist * thisDist;
            sumsSquared[c] = sums[c] * sums[c];
            
            //Update min/max distance observed so far
            if(orderLineObj.getDistance() != 0.0){
                if(minDistance == -1 || minDistance > orderLineObj.getDistance()){
                    minDistance = orderLineObj.getDistance();
                }
            
                if(maxDistance == -1 || maxDistance < orderLineObj.getDistance()){
                    maxDistance = orderLineObj.getDistance();
                }
            }
            
            //Update mean distance orderline
            boolean isUpdated = false;
            for(int i = 0; i < meanDistOrderLine.size(); i++){
                if(meanDistOrderLine.get(i).getClassVal() == orderLineObj.getClassVal()){
                    meanDistOrderLine.get(i).setDistance(sums[(int)orderLineObj.getClassVal()] / orderLineClassDist.get(orderLineObj.getClassVal()));
                    isUpdated = true;
                    break;
                }
            }
            
            if(!isUpdated){
                meanDistOrderLine.add(new OrderLineObj(sums[(int)orderLineObj.getClassVal()] / orderLineClassDist.get(orderLineObj.getClassVal()), orderLineObj.getClassVal()));
            }
        }

        
        /**
         * Method to calculate the quality bound for the current orderline
         * @return F-stat statistic bound
         */
        @Override
        public double calculateBestQuality() {
            int numClasses = parentClassDist.size();
            
            //Sort the mean distance orderline
            Collections.sort(meanDistOrderLine);
            
            //Find approximate minimum orderline objects
            OrderLineObj min = new OrderLineObj(-1.0, 0.0);
            for(Double d : parentClassDist.keySet()){
                int unassignedObjs = parentClassDist.get(d) - orderLineClassDist.get(d);
                double distMin = (sums[(int)d.doubleValue()] + (unassignedObjs * minDistance)) / parentClassDist.get(d);
                if(min.getDistance() == -1.0 || distMin < min.getDistance()){
                    min.setDistance(distMin);
                    min.setClassVal(d);
                }
            }
            
            //Find approximate maximum orderline objects
            OrderLineObj max = new OrderLineObj(-1.0, 0.0);
            for(Double d : parentClassDist.keySet()){
                int unassignedObjs = parentClassDist.get(d) - orderLineClassDist.get(d);
                double distMax = (sums[(int)d.doubleValue()] + (unassignedObjs * maxDistance)) / parentClassDist.get(d); 
                if(d != min.getClassVal() && (max.getDistance() == -1.0 || distMax > max.getDistance())){
                    max.setDistance(distMax);
                    max.setClassVal(d);
                }
            }
            
            //Adjust running sums
            double increment = (max.getDistance() - min.getDistance()) / (numClasses-1);
            int multiplyer = 1;
            for(int i = 0; i < meanDistOrderLine.size(); i++){
                OrderLineObj currentObj = meanDistOrderLine.get(i);
                double thisDist;
                int unassignedObjs = parentClassDist.get(currentObj.getClassVal()) - orderLineClassDist.get(currentObj.getClassVal());
                
                if(currentObj.getClassVal() == min.getClassVal()){
                    thisDist = minDistance;
                }else if(currentObj.getClassVal() == max.getClassVal()){
                    thisDist = maxDistance;
                }else{
                    thisDist = minDistance + (increment * multiplyer);
                    multiplyer++;        
                }
                sums[(int)currentObj.getClassVal()] += thisDist * unassignedObjs;
                sumOfSquares[(int)currentObj.getClassVal()] += thisDist * thisDist * unassignedObjs;
                sumsSquared[(int)currentObj.getClassVal()] = sums[(int)currentObj.getClassVal()] * sums[(int)currentObj.getClassVal()];
            }
            
            double ssTotal;
            double part1 = 0;
            double part2 = 0;

            for (int i = 0; i < numClasses; i++) {
                part1 += sumOfSquares[i];
                part2 += sums[i];
            }

            part2 *= part2;
            part2 /= numInstances;
            ssTotal = part1 - part2;

            double ssAmoung;
            part1 = 0;
            part2 = 0;
            for (int i = 0; i < numClasses; i++) {
                part1 += (double) sumsSquared[i] / parentClassDist.get((double) i);//.data[i].size();
                part2 += sums[i];
            }
            ssAmoung = part1 - (part2 * part2) / numInstances;
            double ssWithin = ssTotal - ssAmoung;

            int dfAmoung = numClasses - 1;
            int dfWithin = numInstances - numClasses;

            double msAmoung = ssAmoung / dfAmoung;
            double msWithin = ssWithin / dfWithin;

            double f = msAmoung / msWithin;
            
            //Reset running sums
            multiplyer = 1;
            for(int i = 0; i < meanDistOrderLine.size(); i++){
                OrderLineObj currentObj = meanDistOrderLine.get(i);
                double thisDist;
                int unassignedObjs = parentClassDist.get(currentObj.getClassVal()) - orderLineClassDist.get(currentObj.getClassVal());
                
                if(currentObj.getClassVal() == min.getClassVal()){
                    thisDist = minDistance;
                }else if(currentObj.getClassVal() == max.getClassVal()){
                    thisDist = maxDistance;
                }else{
                    thisDist = minDistance + (increment * multiplyer);
                    multiplyer++;        
                }
                sums[(int)currentObj.getClassVal()] -= thisDist * unassignedObjs;
                sumOfSquares[(int)currentObj.getClassVal()] -= thisDist * thisDist * unassignedObjs;
                sumsSquared[(int)currentObj.getClassVal()] = sums[(int)currentObj.getClassVal()] * sums[(int)currentObj.getClassVal()];
            }
            
            return Double.isNaN(f) ? 0.0 : f;
        }    
        
        @Override
        public boolean pruneCandidate(){
            if(orderLine.size() % parentClassDist.size() != 0){
                return false;
            }else{
                return super.pruneCandidate();
            }
        }  
                
    }     
    
    /**
     * A class for calculating the Kruskal Wallis statistic bound of a shapelet, according to 
     * the set of distances from the shapelet to a dataset.
     */
    public static class KruskalWallisBound extends QualityBound.ShapeletQualityBound{
        
        /**
         * Constructor to construct KruskalWallisBound
         * @param classDist class distribution of the data currently being processed
         * @param percentage percentage of data required to be processed before
         *                   bounding mechanism is used.
         */
        public KruskalWallisBound(ClassDistribution classDist, int percentage){
            initParentFields(classDist, percentage);
        }
               
        @Override
        public void updateOrderLine(OrderLineObj orderLineObj){
            super.updateOrderLine(orderLineObj);
            numInstances--;
        }
               
        /**
         * Method to calculate the quality bound for the current orderline
         * @return Kruskal Wallis statistic bound
         */
        @Override
        protected double calculateBestQuality() {
             
            //1) Find sums of ranks for the observed orderline objects
            int numClasses = parentClassDist.size();
            int[] classRankCounts = new int[numClasses];
            double minimumRank = -1.0;
            double maximumRank = -1.0;
            double lastDistance = orderLine.get(0).getDistance();
            double thisDistance;
            double classVal = orderLine.get(0).getClassVal();
            classRankCounts[(int)classVal]+=1;

            int duplicateCount = 0;

            for(int i=1; i< orderLine.size(); i++){
                thisDistance = orderLine.get(i).getDistance();
                if(duplicateCount == 0 && thisDistance!=lastDistance){ // standard entry
                    classRankCounts[(int)orderLine.get(i).getClassVal()]+=i+1;
                    
                    //Set min/max ranks
                    if(thisDistance > 0.0 && minimumRank == -1.0){
                        minimumRank = i+1;
                    }
                    maximumRank = i+1;
                }else if(duplicateCount > 0 && thisDistance!=lastDistance){ // non-duplicate following duplicates
                    // set ranks for dupicates

                    double minRank = i-duplicateCount;
                    double maxRank = i;
                    double avgRank = (minRank+maxRank)/2;

                    for(int j = i-duplicateCount-1; j < i; j++){
                        classRankCounts[(int)orderLine.get(j).getClassVal()]+=avgRank;
                    }


                    duplicateCount = 0;
                    // then set this rank
                    classRankCounts[(int)orderLine.get(i).getClassVal()]+=i+1;
                   
                    //Set min/max ranks
                    if(thisDistance > 0.0 && minimumRank == -1.0){
                        minimumRank = i+1;
                    }
                    maximumRank = i+1;
                } else{// thisDistance==lastDistance
                    if(i == orderLine.size() - 1){ // last one so must do the avg ranks here (basically copied from above, BUT includes this element too now)

                        double minRank = i-duplicateCount;
                        double maxRank = i+1;
                        double avgRank = (minRank+maxRank)/2;

                        for(int j = i-duplicateCount-1; j <= i; j++){
                            classRankCounts[(int)orderLine.get(j).getClassVal()]+=avgRank;
                        }
                        
                        //Set min/max ranks
                        if(thisDistance > 0.0 && minimumRank == -1.0){
                            minimumRank = avgRank;
                        }
                        maximumRank = avgRank;  
                    }
                    duplicateCount++;
                }
                lastDistance = thisDistance;
            }

            // 2) Compute mean rank for the obsereved objects 
            ArrayList<OrderLineObj> meanRankOrderLine = new ArrayList<OrderLineObj>();
            for(int i = 0; i < numClasses; i++){
                meanRankOrderLine.add(new OrderLineObj((double)classRankCounts[i]/orderLineClassDist.get((double)i), (double)i));
            }
            Collections.sort(meanRankOrderLine);
            
            //Find approximate minimum orderline objects
            OrderLineObj min = new OrderLineObj(-1.0, 0.0);
            for(int i = 0; i < meanRankOrderLine.size(); i++){
                classVal = meanRankOrderLine.get(i).getClassVal();
                int unassignedObjs = parentClassDist.get(classVal) - orderLineClassDist.get(classVal);
                double observed = classRankCounts[(int)classVal];
                double predicted = minimumRank * unassignedObjs;
                double approximateRank = (observed + predicted) / parentClassDist.get(classVal);
                if(min.getDistance() == -1.0 || approximateRank < min.getDistance()){
                    min.setDistance(approximateRank);
                    min.setClassVal(classVal);
                }
            }
            
            //Find approximate maximum orderline objects
            OrderLineObj max = new OrderLineObj(-1.0, 0.0);
            for(int i = 0; i < meanRankOrderLine.size(); i++){
                classVal = meanRankOrderLine.get(i).getClassVal();
                int unassignedObjs = parentClassDist.get(classVal) - orderLineClassDist.get(classVal);
                double observed = classRankCounts[(int)classVal];
                double predicted = maximumRank * unassignedObjs;
                double approximateRank = (observed + predicted) / parentClassDist.get(classVal); 
                if(classVal != min.getClassVal() && (max.getDistance() == -1.0 || approximateRank > max.getDistance())){
                    max.setDistance(approximateRank);
                    max.setClassVal(classVal);
                }
            }
            
            //3) overall mean rank
            double overallMeanRank = (1.0+ orderLine.size() + numInstances)/2;
    
            //4) Interpolate mean ranks
            double increment = (max.getDistance() - min.getDistance()) / (numClasses-1);
            int multiplyer = 1;
            for(int i = 0; i < meanRankOrderLine.size(); i++){
                OrderLineObj currentObj = meanRankOrderLine.get(i);
                int unassignedObjs = parentClassDist.get(currentObj.getClassVal()) - orderLineClassDist.get(currentObj.getClassVal());
                
                if(currentObj.getClassVal() == min.getClassVal()){
                    currentObj.setDistance(min.getDistance());
                }else if(currentObj.getClassVal() == max.getClassVal()){
                    currentObj.setDistance(max.getDistance());
                }else{
                    classVal = currentObj.getClassVal();
                    double observed = classRankCounts[(int)classVal];
                    double predicted = (minimumRank + (increment * multiplyer)) * unassignedObjs;
                    double approximateRank = (observed + predicted) / parentClassDist.get(classVal); 
                    currentObj.setDistance(approximateRank);
                    multiplyer++;        
                }
                
            }
            
            //5) sum of squared deviations from the overall mean rank
            double s = 0;
            for(int i = 0; i < numClasses; i++){
                s+= parentClassDist.get((double)i)*(meanRankOrderLine.get(i).getDistance() -overallMeanRank)*(meanRankOrderLine.get(i).getDistance() -overallMeanRank);
            }

            //6) weight s with the scale factor
            int totalInstances = orderLine.size() + numInstances;
            double h = 12.0/(totalInstances*(totalInstances+1))*s;

            return h;
        }
    
        @Override
        public boolean pruneCandidate(){
            if(orderLine.size() % parentClassDist.size() != 0){
                return false;
            }else{
                return super.pruneCandidate();
            }
        }  
                
    }
}
