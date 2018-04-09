package applications;

/**
 *
 * @author Jason Lines (j.lines@uea.ac.uk)
 * 
 * Continuation of the work by AJB in Booze.java. Uses the two problems created there:
 *      FiveClassV1.arff and TwoClassV1.arff
 * 
 * These instances are normalised and cropped to wavelengths 876-1100
 * 
 */
import java.io.File;
import java.util.ArrayList;
import java.util.TreeSet;
import java.util.Scanner;
import java.util.TreeMap;
import java.text.DecimalFormat;

import utilities.ClassifierTools;

import weka.classifiers.Classifier;
import tsc_algorithms.ElasticEnsemble;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.timeseries.SummaryStats;

public class AlcoholLevel {

    private static final String PATH = "C:/Temp/Dropbox/IFR Spirits/AlcoholLevelClassification/";
    
    public static TreeSet<String> listWhiteStandardName() throws Exception{
        TreeSet<String> whiteList = new TreeSet<>();
        File input = new File(PATH+"white_glass_standard.txt");
        Scanner scan = new Scanner(input);
        scan.useDelimiter("\n");
        
        while(scan.hasNext()){
            whiteList.add(scan.next().trim());
        }
        
        return whiteList;
    }
    
    public static Instances onlyWhiteStandardBottles(Instances all) throws Exception{
        Instances whiteStandard = new Instances(all,0);
        
        // read in white list
        TreeSet<String> whiteList = listWhiteStandardName();
        for(Instance i:all){
            if(whiteList.contains(i.stringValue(0).trim())){
                whiteStandard.add(i);
            }
        }
        return whiteStandard;
    }
    
    public static TreeMap<String, Instances> getInstancesByBottle(Instances instances) throws Exception{
        
        TreeMap<String, Instances> bottles = new TreeMap<>();
        Instances bottleInstances;
        String bottleName;
        for(Instance instance:instances){
            bottleName = instance.stringValue(0);
            
            if(bottles.containsKey(bottleName)){
                bottles.get(bottleName).add(instance);
            }else{
                bottleInstances = new Instances(instances, 0);
                bottleInstances.add(instance);
                bottles.put(bottleName, bottleInstances);
            }
        }
        return bottles;
    }
    
    public static void summaryClassification(String inputFileName, boolean standardWhiteOnly) throws Exception{
        SummaryStats ss = new SummaryStats();
        DecimalFormat df = new DecimalFormat("#.###");
        Instances input = utilities.ClassifierTools.loadData(PATH+inputFileName);
        
        // if only looking at white standard, remove bottles that are irregular or non-clear
        if(standardWhiteOnly){
            input = onlyWhiteStandardBottles(input);
        }
        
        // split by bottle form leave-one-bottle-out (lobo)
        TreeMap<String, Instances> byBottle = getInstancesByBottle(input);
        TreeMap<String, Instances> transformedBottles = new TreeMap<>();
        
        //remove text attributes and transform
        for(String bottle:byBottle.keySet()){
            byBottle.get(bottle).deleteAttributeAt(0);
            transformedBottles.put(bottle, ss.process(byBottle.get(bottle)));
        }
        
        Instances train, test;
        Classifier[] classifiers;
        int[] correct = null;
        double prediction;
        int total = 0;
        
        ArrayList<String> names = null;
        for(String testBottle:transformedBottles.keySet()){
            test = transformedBottles.get(testBottle);
            total+=test.numInstances();
            train = new Instances(test,0);
            for(String trainBottle:transformedBottles.keySet()){
                if(!trainBottle.equalsIgnoreCase(testBottle)){
                    train.addAll(transformedBottles.get(trainBottle));
                }
            }
            names = new ArrayList<>();
            classifiers = ClassifierTools.setDefaultSingleClassifiers(names);
            
            if(correct==null){
                correct = new int[classifiers.length];
            }
            
            for(int c = 0; c < classifiers.length; c++){
                classifiers[c].buildClassifier(train);
                for(int i = 0; i < test.numInstances();i++){
                    prediction = classifiers[c].classifyInstance(test.instance(i));
                    if(prediction==test.instance(i).classValue()){
                        correct[c]++;
                    }
                }
            }
        }
        
        for(int c = 0; c < names.size(); c++){
            System.out.println(names.get(c)+": "+correct[c]+"/"+total+" "+df.format(100.0/total*correct[c])+"%");
        }
    }
    
    public static void euclideanClassification(String inputFileName, boolean standardWhiteOnly) throws Exception{
        Instances input = utilities.ClassifierTools.loadData(PATH+inputFileName);
//        Instances input = utilities.ClassifierTools.loadData(inputFileName);
        DecimalFormat df = new DecimalFormat("#.###");
        // if only looking at white standard, remove bottles that are irregular or non-clear
        if(standardWhiteOnly){
            input = onlyWhiteStandardBottles(input);
        }
        
        // split by bottle form leave-one-bottle-out (lobo)
        TreeMap<String, Instances> byBottle = getInstancesByBottle(input);
        
        //remove text attributes
        for(String bottle:byBottle.keySet()){
            byBottle.get(bottle).deleteAttributeAt(0);
        }
        
        Instances train, test;
        ElasticEnsemble ee;
        int correct = 0;
        double prediction;
        int total = 0;
        
        
        for(String testBottle:byBottle.keySet()){
            test = byBottle.get(testBottle);
            total+=test.numInstances();
            train = new Instances(test,0);
            for(String trainBottle:byBottle.keySet()){
                if(!trainBottle.equalsIgnoreCase(testBottle)){
                    train.addAll(byBottle.get(trainBottle));
                }
            }
            ee = new ElasticEnsemble();
            ee.removeAllClassifiersFromEnsemble();
            ee.addClassifierToEnsemble(ElasticEnsemble.ClassifierVariants.Euclidean_1NN);
                                    
            
            ee.buildClassifier(train);
            for(int i = 0; i < test.numInstances();i++){
                prediction = ee.classifyInstance(test.instance(i));
                if(prediction==test.instance(i).classValue()){
                    correct++;
                }else{
//                    System.out.println("Wrong: "+testBottle);
                }
            }
            
        }
        
        System.out.println("Euclidean EE: "+correct+"/"+total+" "+df.format(100.0/total*correct)+"%");
        
    }
    
    public static void main(String[] args) throws Exception{
        
        System.out.println("Summary Stats and Euclidean 1NN");
        System.out.println("Two Class, all bottles:");
        System.out.println("=============================================");
        summaryClassification("AlcoholLevelClassification/TwoClassV1",false);
        euclideanClassification("AlcoholLevelClassification/TwoClassV1",false);
        System.out.println("=============================================\n");

        System.out.println("Two Class, white and standard only:");
        System.out.println("=============================================");
        summaryClassification("AlcoholLevelClassification/TwoClassV1",true);
        euclideanClassification("AlcoholLevelClassification/TwoClassV1",true);
        System.out.println("=============================================\n");

        System.out.println("Five Class, all bottles:");
        System.out.println("=============================================");
        summaryClassification("AlcoholLevelClassification/FiveClassV1",false);
        euclideanClassification("AlcoholLevelClassification/FiveClassV1",false);
        System.out.println("=============================================\n");

        System.out.println("Five Class, white and standard only:");
        System.out.println("=============================================");
        summaryClassification("AlcoholLevelClassification/FiveClassV1",true);
        euclideanClassification("AlcoholLevelClassification/FiveClassV1",true);
        System.out.println("=============================================\n");
   
        
    }
}
