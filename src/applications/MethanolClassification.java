/**
 * @author Jason Lines (j.lines@uea.ac.uk)
 * 
 * Class in progress; no helpful commenting/javadoc!
 */
package applications;


import java.util.ArrayList;
import java.util.TreeMap;
import java.text.DecimalFormat;
import java.util.Set;
import utilities.ClassifierTools;
import weka.core.Instance;
import weka.core.Instances;
import weka.classifiers.Classifier;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.lazy.DTW_1NN;

import weka.filters.timeseries.SummaryStats;

import tsc_algorithms.ElasticEnsemble;
import weka.filters.NormalizeCase;
import weka.filters.timeseries.PowerSpectrum;
import weka.filters.timeseries.shapelet_transforms.ApproximateShapeletTransform;

public class MethanolClassification {
    
    private static final String PATH = "C:\\Temp\\Dropbox\\IFR Spirits\\MethanolDetection\\ProblemFolder\\SecondGen\\";
    
    public static TreeMap<String, Instances> getInstancesByMethanolBottle(Instances instances) throws Exception{
        
        TreeMap<String, Instances> bottles = new TreeMap<>();
        Instances bottleInstances;
        String bottleName;
        for(Instance instance:instances){
            bottleName = instance.stringValue(0);
            
            int positionId = -1;
            if(bottleName.charAt(bottleName.length()-10)=='P'){
                positionId = Integer.parseInt(bottleName.substring(bottleName.length()-9,bottleName.length()-8));
            }else{
                positionId = Integer.parseInt(bottleName.substring(bottleName.length()-4,bottleName.length()-3));
            }
            
//            if(positionId==2){ // used when investigating position 2, which seems to make everything worse!
//                continue;
//            }
            
            bottleName = bottleName.substring(0, bottleName.length()-11); // -11 to remove run, position and channel information
            
            
            
            if(bottles.containsKey(bottleName)){
                bottles.get(bottleName).add(instance);
            }else{
                bottleInstances = new Instances(instances, 0);
                bottleInstances.add(instance);
                bottles.put(bottleName, bottleInstances);
            }
        }
        
        for(String bottle: bottles.keySet()){
//            System.out.println(bottle+"\t"+bottles.get(bottle).size());
            bottles.get(bottle).deleteAttributeAt(0); // remove now redundant bottle name
        }
        
        return bottles;
    }
    
    public static ArrayList<TreeMap<String, Instances>> getPositionBottles(Instances instances) throws Exception{
        ArrayList<TreeMap<String, Instances>> byPosition = new ArrayList<>();
        
        for(int i = 0; i < 5; i++){
            byPosition.add(new TreeMap<String, Instances>());
        }
        TreeMap<String, Instances> bottles;// = new TreeMap<>();
        Instances bottleInstances;
        String bottleName;
        for(Instance instance:instances){
            bottleName = instance.stringValue(0);
            
            int positionId = -1;
            if(bottleName.charAt(bottleName.length()-10)=='P'){
                positionId = Integer.parseInt(bottleName.substring(bottleName.length()-9,bottleName.length()-8));
            }else{
                positionId = Integer.parseInt(bottleName.substring(bottleName.length()-4,bottleName.length()-3));
            }
            
//            if(positionId==2){ // used when investigating position 2, which seems to make everything worse!
//                continue;
//            }
            
            bottleName = bottleName.substring(0, bottleName.length()-11); // -11 to remove run, position and channel information
            
            bottles = byPosition.get(positionId-1);
            
            if(bottles.containsKey(bottleName)){
                bottles.get(bottleName).add(instance);
            }else{
                bottleInstances = new Instances(instances, 0);
                bottleInstances.add(instance);
                bottles.put(bottleName, bottleInstances);
            }
        }
        
        for(int i = 0; i < 5; i++){
            bottles = byPosition.get(i);
            for(String bottle: bottles.keySet()){
//            System.out.println(bottle+"\t"+bottles.get(bottle).size());
                bottles.get(bottle).deleteAttributeAt(0); // remove now redundant bottle name
            }
        }
        
        return byPosition;
    }
    
    public static TreeMap<String, Instances> getInstancesByMethanolBottleAndPostition(Instances instances, int position) throws Exception{
        
        TreeMap<String, Instances> bottles = new TreeMap<>();
        Instances bottleInstances;
        String bottleName;
        for(Instance instance:instances){
            bottleName = instance.stringValue(0);
            
            int positionId = -1;
            if(bottleName.charAt(bottleName.length()-10)=='P'){
                positionId = Integer.parseInt(bottleName.substring(bottleName.length()-9,bottleName.length()-8));
            }else{
                positionId = Integer.parseInt(bottleName.substring(bottleName.length()-4,bottleName.length()-3));
            }
            
            if(position!=positionId){
                continue;
            }
            
            bottleName = bottleName.substring(0, bottleName.length()-11); // -11 to remove run, position and channel information
            
            if(bottles.containsKey(bottleName)){
                bottles.get(bottleName).add(instance);
            }else{
                bottleInstances = new Instances(instances, 0);
                bottleInstances.add(instance);
                bottles.put(bottleName, bottleInstances);
            }
        }
        
        for(String bottle: bottles.keySet()){
//            System.out.println(bottle+"\t"+bottles.get(bottle).size());
            bottles.get(bottle).deleteAttributeAt(0); // remove now redundant bottle name
        }
        
        return bottles;
    }
    
   
    public static TreeMap<String, Instances> normaliseBottles(TreeMap<String, Instances> bottles) throws Exception{
        TreeMap<String, Instances> outBottles = new TreeMap<>();
        NormalizeCase nc = new NormalizeCase();
        for(String bottle: bottles.keySet()){
            outBottles.put(bottle, nc.process(bottles.get(bottle)));
        }
        return outBottles;
    }
    
    public static TreeMap<String, Instances> summariseBottles(TreeMap<String, Instances> bottles) throws Exception{
        TreeMap<String, Instances> outBottles = new TreeMap<>();
        SummaryStats ss = new SummaryStats();
        for(String bottle: bottles.keySet()){
            outBottles.put(bottle, ss.process(bottles.get(bottle)));
        }
        return outBottles;
    }
    
    public enum ClassificationType{STATS,EUCLIDEAN, SHAPELETS, DTW5};
    public enum IndividualPrinting {ON,OFF};
    public enum SinglePositionClassification {OFF,P1,P2,P3,P4,P5};
    
    public static void leaveOneBottleOut(Instances instances, ClassificationType classificationType, IndividualPrinting individualPrinting, SinglePositionClassification singlePos) throws Exception{
        
        TreeMap<String, Instances> readIn;
        if(singlePos==SinglePositionClassification.OFF){
            readIn = getInstancesByMethanolBottle(instances);
        }else{
            readIn = getInstancesByMethanolBottleAndPostition(instances,singlePos.ordinal());
        }
        TreeMap<String, Instances> bottles = normaliseBottles(readIn);
        
        TreeMap<String, TreeMap<String, Integer>> correctByBottle = new TreeMap<>();
        
        if(classificationType == ClassificationType.STATS){
            bottles = summariseBottles(bottles);
        }
        
        
        //debugging: check bottle names and instances:
//        for(String bottle: bottles.keySet()){
//            System.out.println(bottle+", "+bottles.get(bottle).size());
//        }
        
        
        Instances train, test;
        Classifier[] classifiers;
        ElasticEnsemble ee;
        ApproximateShapeletTransform st;
        
        
        int[] correct = null;
        double prediction;
        int total = 0;
        DecimalFormat df = new DecimalFormat("#.###");
        int bottleCorrect = 0;
        String classifierName;
        
        ArrayList<String> names = null;
        for(String testBottle:bottles.keySet()){
            
            test = bottles.get(testBottle);
            
            total+=test.numInstances();
            train = new Instances(test,0);
            for(String trainBottle:bottles.keySet()){
                if(!trainBottle.equalsIgnoreCase(testBottle)){
                    train.addAll(bottles.get(trainBottle));
                }
            }
            names = new ArrayList<>();
            
            if(classificationType==ClassificationType.STATS){
                classifiers = ClassifierTools.setDefaultSingleClassifiers(names); 
            }else if(classificationType==ClassificationType.SHAPELETS){
                classifiers = ClassifierTools.setDefaultSingleClassifiers(names);
                st = new ApproximateShapeletTransform(50, 250, 250);
                
                train = st.process(train);
                test = st.process(test);
                
            }else if(classificationType==ClassificationType.EUCLIDEAN){
                ee = new ElasticEnsemble();
                ee.removeAllClassifiersFromEnsemble();
                ee.addClassifierToEnsemble(ElasticEnsemble.ClassifierVariants.Euclidean_1NN);
                classifiers = new Classifier[1];
                classifiers[0] = ee;
                names.add("ED");
            }else if(classificationType==ClassificationType.DTW5){
                classifiers = new Classifier[1];
                DTW_1NN dtw = new DTW_1NN();
                dtw.setR(0.05);
                classifiers[0] = dtw;
                names.add("DTW_5");
            }else{
                classifiers = null;
            }
            
            
            
            if(correct==null){
                correct = new int[classifiers.length];
            }
            
            for(int c = 0; c < classifiers.length; c++){
                classifiers[c].buildClassifier(train);
                bottleCorrect = 0;
                for(int i = 0; i < test.numInstances();i++){
                    prediction = classifiers[c].classifyInstance(test.instance(i));
                    
                    if(prediction==test.instance(i).classValue()){
                        correct[c]++;
                        bottleCorrect++;
                    }
//                    if(c==3){ // print out svmL
//                        System.out.println(prediction+"    "+ test.instance(i).classValue());
//                    }
                }
                classifierName = names.get(c);
                if(!correctByBottle.containsKey(classifierName)){
                    correctByBottle.put(classifierName, new TreeMap<String, Integer>());
                }
                correctByBottle.get(classifierName).put(testBottle, bottleCorrect); 
            }
        }
        
        for(int c = 0; c < names.size(); c++){
            System.out.println(names.get(c)+": "+correct[c]+"/"+total+" "+df.format(100.0/total*correct[c])+"%");
        }
        
        if(individualPrinting==IndividualPrinting.ON){
            for(String classifier:correctByBottle.keySet()){
                System.out.println(classifier);
                for(String bottle:correctByBottle.get(classifier).keySet()){
                    System.out.printf("%s \t %d \n", bottle, correctByBottle.get(classifier).get(bottle));            
                }
            }
        }
        
        
    }

    public static void debugMethanolClassification() throws Exception{
        
        Instances train = ClassifierTools.loadData(PATH+"boarderlineSafeVsAdvised_debug_train");
        Instances test = ClassifierTools.loadData(PATH+"boarderlineSafeVsAdvised_debug_test");
        train.deleteAttributeAt(0);
        test.deleteAttributeAt(0);
        NormalizeCase nc = new NormalizeCase();
        train = nc.process(train);
        test = nc.process(test);
        
        SummaryStats ss = new SummaryStats();
        train = ss.process(train);
        test = ss.process(test);
        
//        weka.classifiers.functions.SMO smo = new SMO();
//        smo.buildClassifier(train);
        
        SMO smo =new SMO();
        PolyKernel kernel = new PolyKernel();
        kernel.setExponent(1);
        ((SMO)smo).setKernel(kernel);
        smo.buildClassifier(train);
        
        int correct = 0;
        int total = 0;
        
        for(int i = 0;i < test.numInstances(); i++){
            if(smo.classifyInstance(test.instance(i))==test.instance(i).classValue()){
                correct++;
            }
            total++;
        }
        System.out.println(correct+"/"+total);
        
    }
    
    public static void positionEnsemble(Instances input) throws Exception{
        ArrayList<TreeMap<String, Instances>> positions = getPositionBottles(input);
        
        Set<String> bottleNames = positions.get(0).keySet();
        
        int total = 0;
        int correct = 0;
        
        for(String testBottle:bottleNames){

            ElasticEnsemble[] positionEEs = new ElasticEnsemble[5];
            ElasticEnsemble ee;
            Instances[] trainPositions = new Instances[5];
            Instances[] testPositions = new Instances[5];
        
            // get instances ready for each position and train classifier
            for(int p = 0; p < 5; p++){
                testPositions[p] = positions.get(p).get(testBottle);
                trainPositions[p] = new Instances(testPositions[p],0);
                for(String bottle:bottleNames){
                    if(!bottle.equalsIgnoreCase(testBottle)){
                        trainPositions[p].addAll(positions.get(p).get(bottle));
                    }
                }
                
                // train classifiers
                ee = new ElasticEnsemble();
                ee.removeAllClassifiersFromEnsemble();
                ee.addClassifierToEnsemble(ElasticEnsemble.ClassifierVariants.Euclidean_1NN);
                ee.buildClassifier(trainPositions[p]);
                positionEEs[p] = ee;
            }
            
            int numInstances = testPositions[0].numInstances();
            
            // for each instance, get the 5 predictions
            double[] predictions = new double[5];
            for(int i = 0; i < numInstances; i++){
                // naive, but a quick implementation
                int[] votes = new int[2];
                double pred = -1;
                for(int p = 0; p < 5; p++){
                    System.out.print(positionEEs[p].classifyInstance(testPositions[p].instance(i))+",");
                    predictions[p] = positionEEs[p].classifyInstance(testPositions[p].instance(i));
                    if(predictions[p]==0){
                        votes[0]++;
                    }else{
                        votes[1]++;
                    }
                }
                
                if(votes[0] > votes[1]){
                    pred = 0;
                }else{
                    pred = 1;
                }
                
                System.out.println("\t"+pred+"\t"+testPositions[0].instance(i).classValue());
                if(pred==testPositions[0].instance(i).classValue()){
                    correct++;
                }
                total++;
            }
            
        }
        DecimalFormat df = new DecimalFormat("#.###");
        System.out.println(correct+"/"+total);
        System.out.println(df.format(100.0/total*correct)+"%");
        
    }
    
    
    public static void main(String[] args) throws Exception{ 
        
        // Experiemnt 1: 13.3% methanol vs. the rest
        
//        Instances veryHigh = ClassifierTools.loadData(PATH+"veryHigh_vs_rest.arff");
//        leaveOneBottleOut(veryHigh, MethanolClassification.ClassificationType.EUCLIDEAN, IndividualPrinting.OFF, SinglePositionClassification.OFF);
//        leaveOneBottleOut(veryHigh, MethanolClassification.ClassificationType.EUCLIDEAN, IndividualPrinting.OFF, SinglePositionClassification.P1);
//        leaveOneBottleOut(veryHigh, MethanolClassification.ClassificationType.EUCLIDEAN, IndividualPrinting.OFF, SinglePositionClassification.P2);
//        leaveOneBottleOut(veryHigh, MethanolClassification.ClassificationType.EUCLIDEAN, IndividualPrinting.OFF, SinglePositionClassification.P3);
//        leaveOneBottleOut(veryHigh, MethanolClassification.ClassificationType.EUCLIDEAN, IndividualPrinting.OFF, SinglePositionClassification.P4);
//        leaveOneBottleOut(veryHigh, MethanolClassification.ClassificationType.EUCLIDEAN, IndividualPrinting.OFF, SinglePositionClassification.P5);
//        positionEnsemble(veryHigh);
        
//         Experiemnt 2: 13.3% methanol and 1.3 vs. the rest
        
//        Instances high = ClassifierTools.loadData(PATH+"greaterThan40vsAll.arff");
//        leaveOneBottleOut(high, MethanolClassification.ClassificationType.EUCLIDEAN, IndividualPrinting.OFF,SinglePositionClassification.OFF);
//        leaveOneBottleOut(high, MethanolClassification.ClassificationType.EUCLIDEAN, IndividualPrinting.OFF,SinglePositionClassification.P1);
//        leaveOneBottleOut(high, MethanolClassification.ClassificationType.EUCLIDEAN, IndividualPrinting.OFF,SinglePositionClassification.P2);
//        leaveOneBottleOut(high, MethanolClassification.ClassificationType.EUCLIDEAN, IndividualPrinting.OFF,SinglePositionClassification.P3);
//        leaveOneBottleOut(high, MethanolClassification.ClassificationType.EUCLIDEAN, IndividualPrinting.OFF,SinglePositionClassification.P4);
//        leaveOneBottleOut(high, MethanolClassification.ClassificationType.EUCLIDEAN, IndividualPrinting.OFF,SinglePositionClassification.P5);
//        positionEnsemble(high);
        
        // Experiemnt 3: 
        
//        Instances boarderline = ClassifierTools.loadData(PATH+"boarderlineSafeVsAdvised.arff");
//        leaveOneBottleOut(boarderline, MethanolClassification.ClassificationType.STATS, IndividualPrinting.OFF);
//        leaveOneBottleOut(boarderline, MethanolClassification.ClassificationType.EUCLIDEAN, IndividualPrinting.OFF);
        
        // Experiment 4: (paired with 1) 0% methanol vs rest (1. is the highest class vs rest, this is lowest vs rest)
        Instances noneVsRest = ClassifierTools.loadData(PATH+"noneVsRest.arff");
        leaveOneBottleOut(noneVsRest, MethanolClassification.ClassificationType.EUCLIDEAN, IndividualPrinting.OFF,SinglePositionClassification.OFF);
        leaveOneBottleOut(noneVsRest, MethanolClassification.ClassificationType.EUCLIDEAN, IndividualPrinting.OFF,SinglePositionClassification.P1);
        leaveOneBottleOut(noneVsRest, MethanolClassification.ClassificationType.EUCLIDEAN, IndividualPrinting.OFF,SinglePositionClassification.P2);
        leaveOneBottleOut(noneVsRest, MethanolClassification.ClassificationType.EUCLIDEAN, IndividualPrinting.OFF,SinglePositionClassification.P3);
        leaveOneBottleOut(noneVsRest, MethanolClassification.ClassificationType.EUCLIDEAN, IndividualPrinting.OFF,SinglePositionClassification.P4);
        leaveOneBottleOut(noneVsRest, MethanolClassification.ClassificationType.EUCLIDEAN, IndividualPrinting.OFF,SinglePositionClassification.P5);
        positionEnsemble(noneVsRest);

        
    
    }
}