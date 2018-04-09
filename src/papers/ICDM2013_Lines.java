package papers;
/**
 *
 * @author Author 1 and Author 2
 */


import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Random;
import java.util.Scanner;
import java.util.TreeMap;

import weka.classifiers.lazy.kNN;
import weka.core.elastic_distance_measures.BasicDTW;
import weka.core.elastic_distance_measures.SakoeChibaDTW;
import weka.core.elastic_distance_measures.WeightedDTW;
import weka.core.elastic_distance_measures.LCSSDistance;

import weka.filters.timeseries.DerivativeFilter;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.EuclideanDistance;

public class ICDM2013_Lines {

    /* DATA_DIR is the location of the arff instance files. Each dataset should be included in this folder witin a subfolder of the dataName,
     * which contains the training and test data in the form 'dataName/dataName_TRAIN.arff' and dataName/dataName_TEST.arff'
     * 
     * OUTPUT_DIR is the location where experimental results will be written. The CV and TRAIN_TEST subfolders are used to keep the different
     * experiments seperate and are relative to the overall OUTPUT_DIR String.
     */
    public static final String DATA_DIR = "TSC Problems";           
    public static final String OUTPUT_DIR = "Results";                              
    public static final String OUTPUT_DIR_CV = OUTPUT_DIR+"/cv";
    public static final String OUTPUT_DIR_TRAIN_TEST = OUTPUT_DIR+"/trainTest";

    //<editor-fold defaultstate="collapsed" desc="Initialisation Methods">
    public static void initCv(String dataName) throws Exception{
        // create outputDir (if it doesn't exist)
        File outputDir = new File(OUTPUT_DIR);
        File outputDirCv = new File(OUTPUT_DIR_CV);
        outputDir.mkdir();
        outputDirCv.mkdir();
        
        // write decision log
        //if dataset doesn't have a dir in the cv dir yet, make one. Else, throw exception to alert user
        //to possiblility of overwriting results
        File datasetDir = new File(OUTPUT_DIR_CV+"/"+dataName);
        if(datasetDir.exists()){
            throw new Exception("WARNING! Cross-validation results already exist for dataset '"+dataName+"'. Please remove these (or rename), then re-run if you wish to continue.");
        }
        datasetDir.mkdir();
    }
    
    public static void initTrainTest(String dataName) throws Exception{
        // create results outputDir (if it doesn't exist)
        File outputDirTrainTest = new File(OUTPUT_DIR_TRAIN_TEST);
        outputDirTrainTest.mkdir();
        
        // make specific data dir
        File datasetDir = new File(OUTPUT_DIR_TRAIN_TEST+"/"+dataName);
        datasetDir.mkdir();
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="Cross-Validation Methods">
    
    // generic cross-validation method
    public static double crossValidate(Instances data, int k, EuclideanDistance distanceMetric, StringBuilder st) throws Exception {

        Instances trainLoocv;
        Instance testInstance;

        kNN knn;

        int correct = 0;
        int total = 0;
        double decision, classValue;

        for (int i = 0; i < data.numInstances(); i++) {
            testInstance = data.instance(i);
            trainLoocv = new Instances(data, data.numInstances() - 1);
            classValue = testInstance.classValue();

            // add all instances to trainLoocv EXCEPT instance[i]
            for (int j = 0; j < data.numInstances(); j++) {
                if (j != i) {
                    trainLoocv.add(data.instance(j));
                }
            }

            if (trainLoocv.numInstances() != data.numInstances() - 1) {
                throw new Exception("Incorrect initialisation of instances!");
            }

            // build classifier and classify
            knn = new kNN(k);
            knn.setDistanceFunction(distanceMetric);
            knn.buildClassifier(trainLoocv);
            decision = knn.classifyInstance(testInstance);

            if (decision == classValue) {
                correct++;
            }
            total++;
            if(st!=null){
                st.append(decision).append(",").append(classValue).append("\n");
            }
        }

        return 100.0/total*correct;
    }
    
    
    // start of experiment-specific cross-validation methods
    
    public static double cv_01_Euclidean_1NN(String dataName, Instances data) throws Exception{
        EuclideanDistance euclid = new EuclideanDistance();
        euclid.setDontNormalize(true);
        
        StringBuilder st = new StringBuilder();
        double euclidean_1nn = crossValidate(data, 1, euclid,st);
        
        // write log
        FileWriter log = new FileWriter(OUTPUT_DIR_CV+"/"+dataName+"/cv_01_Euclidean_1NN.txt");
        log.append(euclidean_1nn+"\n");
        log.append(st);
        log.close();
        return euclidean_1nn;
    }
    
    public static double cv_02_DTW_fullWindow_1NN(String dataName, Instances data) throws Exception{
        BasicDTW fullWindowDtw = new BasicDTW();
        StringBuilder st = new StringBuilder();
        
        double dtw_fullWindow_1nn = crossValidate(data, 1, fullWindowDtw,st);
        
        FileWriter log = new FileWriter(OUTPUT_DIR_CV+"/"+dataName+"/cv_02_DTW_fullWindow_1NN.txt");
        log.append(dtw_fullWindow_1nn+"\n");
        log.append(st);
        log.close();
        
        return dtw_fullWindow_1nn;
    }
    
    public static double[] cv_03_DTW_bestWindow_1NN(String dataName, Instances data) throws Exception{
        double r;
        double thisAcc;
        double bsfAcc = -1;
        double bsfR = -1;
        
        StringBuilder bsfSt = null;
        StringBuilder thisSt;
        
        for(int window = 0; window <= 100; window++){
            thisSt = new StringBuilder();
            r = (double)window/100; // to avoid double imprecision
            thisAcc = crossValidate(data, 1, new SakoeChibaDTW(r),thisSt);
            
            if(thisAcc > bsfAcc){
                bsfAcc = thisAcc;
                bsfR = r;
                bsfSt = thisSt;
            }
        }
        
        double[] dtw_cvWindow_1nn = {bsfAcc,bsfR};
        FileWriter log = new FileWriter(OUTPUT_DIR_CV+"/"+dataName+"/cv_03_DTW_bestWindow_1NN.txt");
        log.append(bsfAcc+","+bsfR+"\n");
        log.append(bsfSt);
        log.close();
        
        return dtw_cvWindow_1nn;
    }
    
    public static double[] cv_04_WDTW_1NN(String dataName, Instances data) throws Exception{
        double g;
        double thisAcc = - 1;
        double bsfAcc = -1;
        double bsfG = -1;
        
        StringBuilder bsfSt = null;
        StringBuilder thisSt;
        
        for(int weight = 0; weight <= 100; weight++){
            thisSt = new StringBuilder();
            g = (double)weight/100; // to avoid double imprecision
            thisAcc = crossValidate(data, 1, new WeightedDTW(g),thisSt);
            
            if(thisAcc > bsfAcc){
                bsfAcc = thisAcc;
                bsfG = g;
                bsfSt = thisSt;
            }
        }
        double[] wdtw_1nn = {bsfAcc,bsfG};
        FileWriter log = new FileWriter(OUTPUT_DIR_CV+"/"+dataName+"/cv_04_WDTW_1NN.txt");
        log.append(bsfAcc+","+bsfG+"\n");
        log.append(bsfSt);
        log.close();
        return wdtw_1nn;
    }
    
    public static double[] cv_05_Euclidean_kNN(String dataName, Instances data) throws Exception{
        int bsfK = -1;
        double thisAcc = - 1;
        double bsfAcc = -1;
        EuclideanDistance euclid;
        
        StringBuilder bsfSt = null;
        StringBuilder thisSt;
        
        for(int k = 1; k <= 100; k++){
            thisSt = new StringBuilder();
            euclid = new EuclideanDistance();
            euclid.setDontNormalize(true);
            thisAcc = crossValidate(data, k, euclid,thisSt);
            
            if(thisAcc > bsfAcc){
                bsfAcc = thisAcc;
                bsfK = k;
                bsfSt = thisSt;
            }
        }
        double[] euclidean_knn = {bsfAcc,bsfK};
        FileWriter log = new FileWriter(OUTPUT_DIR_CV+"/"+dataName+"/cv_05_Euclidean_kNN.txt");
        log.append(bsfAcc+","+bsfK+"\n");
        log.append(bsfSt);
        log.close();
        return euclidean_knn;
    }
    
    public static double[] cv_06_DTW_fullWindow_kNN(String dataName, Instances data) throws Exception{
        int bsfK = -1;
        double thisAcc = - 1;
        double bsfAcc = -1;
        BasicDTW dtw;
        
        StringBuilder bsfSt = null;
        StringBuilder thisSt;
        
        for(int k = 1; k <= 100; k++){
            thisSt = new StringBuilder();
            dtw = new BasicDTW();
            thisAcc = crossValidate(data, k, dtw,thisSt);
            
            if(thisAcc > bsfAcc){
                bsfAcc = thisAcc;
                bsfK = k;
                bsfSt = thisSt;
            }
        }
        double[] dtw_fullWindow_knn = {bsfAcc,bsfK};
        FileWriter log = new FileWriter(OUTPUT_DIR_CV+"/"+dataName+"/cv_06_DTW_fullWindow_kNN.txt");
        log.append(bsfAcc+","+bsfK+"\n");
        log.append(bsfSt);
        log.close();
        return dtw_fullWindow_knn;
    }
    
    public static double[] cv_11_DTW_optimalWindow_kNN(String dataName, Instances data) throws Exception{
        int bsfK = -1;
        double bsfR = -1;
        
        double thisAcc = - 1;
        double bsfAcc = -1;
        
        SakoeChibaDTW dtw;
        
        StringBuilder bsfSt = null;
        StringBuilder thisSt;
        
        for(int window = 0; window <= 100; window++){
            double r = (double)window/100; // avoid double imprecision
            //            System.out.println("r: "+r);
            for(int k = 1; k <=100; k++){
                thisSt = new StringBuilder();
                dtw = new SakoeChibaDTW(r);
                thisAcc = crossValidate(data, k, dtw, thisSt);
                
                if(thisAcc > bsfAcc){
                    bsfAcc = thisAcc;
                    bsfK = k;
                    bsfR = r;
                    bsfSt = thisSt;
                }
            }
        }
        double [] dtw_rn_knn = {bsfAcc,bsfK,bsfR};
        FileWriter log = new FileWriter(OUTPUT_DIR_CV+"/"+dataName+"/cv_11_DTW_optimalWindow_kNN.txt");
        log.append(bsfAcc+","+bsfK+","+bsfR+"\n");
        log.append(bsfSt);
        log.close();
        return dtw_rn_knn;
    }
    
    public static double[] cv_12_WDTW_kNN(String dataName, Instances data) throws Exception{
        int bsfK = -1;
        double bsfG = -1;
        
        double thisAcc = - 1;
        double bsfAcc = -1;
        
        WeightedDTW wdtw;
        StringBuilder bsfSt = null;
        StringBuilder thisSt;
        
        for(int weight = 0; weight <= 100; weight++){
            double g = (double)weight/100; // avoid double imprecision
            //            System.out.println("g: "+g);
            for(int k = 1; k <=100; k++){
                thisSt = new StringBuilder();
                wdtw = new WeightedDTW(g);
                thisAcc = crossValidate(data, k, wdtw,thisSt);
                
                if(thisAcc > bsfAcc){
                    bsfAcc = thisAcc;
                    bsfK = k;
                    bsfG = g;
                    bsfSt = thisSt;
                }
            }
        }
        double [] wdtw_knn = {bsfAcc,bsfK,bsfG};
        FileWriter log = new FileWriter(OUTPUT_DIR_CV+"/"+dataName+"/cv_12_WDTW_kNN.txt");
        log.append(bsfAcc+","+bsfK+","+bsfG+"\n");
        log.append(bsfSt);
        log.close();
        return wdtw_knn;
    }
    
    public static double cv_21_DDTW_fullWindow_1NN(String dataName, Instances data) throws Exception{
        if(!data.relationName().contains("derivative")){
            throw new Exception("WARNING! Instances object does not include derivative in the relation name! If this is intentional, please either update relation name, or remove this exception from the code.");
        }
        BasicDTW fullWindowDtw = new BasicDTW();
        StringBuilder st = new StringBuilder();
        
        double cv_21_ddtw_fullWindow_1nn = crossValidate(data, 1, fullWindowDtw, st);
        FileWriter log = new FileWriter(OUTPUT_DIR_CV+"/"+dataName+"/cv_21_DDTW_fullWindow_1NN.txt");
        log.append(cv_21_ddtw_fullWindow_1nn+"\n");
        log.append(st);
        log.close();
        return cv_21_ddtw_fullWindow_1nn;
    }
    
    public static double[] cv_22_DDTW_bestWindow_1NN(String dataName, Instances data) throws Exception{
        
        if(!data.relationName().contains("derivative")){
            throw new Exception("WARNING! Instances object does not include derivative in the relation name! If this is intentional, please either update relation name, or remove this exception from the code.");
        }
        
        double r;
        double thisAcc;
        double bsfAcc = -1;
        double bsfR = -1;
        StringBuilder bsfSt = null;
        StringBuilder thisSt;
        
        for(int window = 0; window <= 100; window++){
            thisSt = new StringBuilder();
            r = (double)window/100; // to avoid double imprecision
            thisAcc = crossValidate(data, 1, new SakoeChibaDTW(r),thisSt);
            
            if(thisAcc > bsfAcc){
                bsfAcc = thisAcc;
                bsfR = r;
                bsfSt = thisSt;
            }
        }
        double[] ddtw_cvWindow_1nn = {bsfAcc,bsfR};
        FileWriter log = new FileWriter(OUTPUT_DIR_CV+"/"+dataName+"/cv_22_DDTW_bestWindow_1NN.txt");
        log.append(bsfAcc+","+bsfR+"\n");
        log.append(bsfSt);
        log.close();
        return ddtw_cvWindow_1nn;
    }
    
    public static double[] cv_23_WDDTW_1NN(String dataName, Instances data) throws Exception{
        if(!data.relationName().contains("derivative")){
            throw new Exception("WARNING! Instances object does not include derivative in the relation name! If this is intentional, please either update relation name, or remove this exception from the code.");
        }
        
        
        double bsfG = -1;
        
        double thisAcc = - 1;
        double bsfAcc = -1;
        
        WeightedDTW wdtw;
        StringBuilder bsfSt = null;
        StringBuilder thisSt;
        
        for(int weight = 0; weight <= 100; weight++){
            thisSt = new StringBuilder();
            double g = (double)weight/100; // avoid double imprecision
            wdtw = new WeightedDTW(g);
            thisAcc = crossValidate(data, 1, wdtw,thisSt);
            
            if(thisAcc > bsfAcc){
                bsfAcc = thisAcc;
                bsfG = g;
                bsfSt = thisSt;
            }
        }
        double [] wddtw_knn = {bsfAcc,bsfG};
        FileWriter log = new FileWriter(OUTPUT_DIR_CV+"/"+dataName+"/cv_23_WDDTW_1NN.txt");
        log.append(bsfAcc+","+bsfG+"\n");
        log.append(bsfSt);
        log.close();
        return wddtw_knn;
    }
    
    public static double[] cv_24_DDTW_fullWindow_kNN(String dataName, Instances data) throws Exception{
        if(!data.relationName().contains("derivative")){
            throw new Exception("WARNING! Instances object does not include derivative in the relation name! If this is intentional, please either update relation name, or remove this exception from the code.");
        }
        
        int bsfK = -1;
        double thisAcc = - 1;
        double bsfAcc = -1;
        BasicDTW dtw;
        StringBuilder bsfSt = null;
        StringBuilder thisSt;
        for(int k = 1; k <= 100; k++){
            thisSt = new StringBuilder();
            dtw = new BasicDTW();
            thisAcc = crossValidate(data, k, dtw,thisSt);
            
            if(thisAcc > bsfAcc){
                bsfAcc = thisAcc;
                bsfK = k;
                bsfSt = thisSt;
            }
        }
        double[] ddtw_fullWindow_knn = {bsfAcc,bsfK};
        FileWriter log = new FileWriter(OUTPUT_DIR_CV+"/"+dataName+"/cv_24_DDTW_fullWindow_kNN.txt");
        log.append(bsfAcc+","+bsfK+"\n");
        log.append(bsfSt);
        log.close();
        return ddtw_fullWindow_knn;
    }
    
    public static double[] cv_25_DDTW_optimalWindow_kNN(String dataName, Instances data) throws Exception{
        
        if(!data.relationName().contains("derivative")){
            throw new Exception("WARNING! Instances object does not include derivative in the relation name! If this is intentional, please either update relation name, or remove this exception from the code.");
        }
        
        int bsfK = -1;
        double bsfR = -1;
        
        double thisAcc = - 1;
        double bsfAcc = -1;
        
        SakoeChibaDTW dtw;
        StringBuilder bsfSt = null;
        StringBuilder thisSt;
        
        for(int window = 0; window <= 100; window++){
            double r = (double)window/100; // avoid double imprecision
            for(int k = 1; k <=100; k++){
                thisSt = new StringBuilder();
                dtw = new SakoeChibaDTW(r);
                thisAcc = crossValidate(data, k, dtw,thisSt);
                
                if(thisAcc > bsfAcc){
                    bsfAcc = thisAcc;
                    bsfK = k;
                    bsfR = r;
                    bsfSt = thisSt;
                }
            }
        }
        double [] ddtw_rn_knn = {bsfAcc,bsfK,bsfR};
        FileWriter log = new FileWriter(OUTPUT_DIR_CV+"/"+dataName+"/cv_25_DDTW_optimalWindow_kNN.txt");
        log.append(bsfAcc+","+bsfK+","+bsfR+"\n");
        log.append(bsfSt);
        log.close();
        return ddtw_rn_knn;
    }
    
    public static double[] cv_26_WDTW_kNN(String dataName, Instances data) throws Exception{
        
        if(!data.relationName().contains("derivative")){
            throw new Exception("WARNING! Instances object does not include derivative in the relation name! If this is intentional, please either update relation name, or remove this exception from the code.");
        }
        
        int bsfK = -1;
        double bsfG = -1;
        
        double thisAcc = - 1;
        double bsfAcc = -1;
        
        WeightedDTW wdtw;
        StringBuilder bsfSt = null;
        StringBuilder thisSt;
        
        for(int weight = 0; weight <= 100; weight++){
            double g = (double)weight/100; // avoid double imprecision
            for(int k = 1; k <=100; k++){
                thisSt = new StringBuilder();
                wdtw = new WeightedDTW(g);
                thisAcc = crossValidate(data, k, wdtw,thisSt);
                
                if(thisAcc > bsfAcc){
                    bsfAcc = thisAcc;
                    bsfK = k;
                    bsfG = g;
                    bsfSt = thisSt;
                }
            }
        }
        double [] wddtw_knn = {bsfAcc,bsfK,bsfG};
        FileWriter log = new FileWriter(OUTPUT_DIR_CV+"/"+dataName+"/cv_26_WDTW_kNN.txt");
        log.append(bsfAcc+","+bsfK+","+bsfG+"\n");
        log.append(bsfSt);
        log.close();
        return wddtw_knn;
    }
    
    
    public static double[] cv_31_LCSS_1NN(String dataName, Instances data) throws Exception{
        // get the 10 delta and 10 epsilon readings between the ranges specified by <citation>
        int seriesLength = data.numAttributes()-1; // -1 to remove class value
        double dataStdv = LCSSDistance.stdv_p(data);
        
        //compute delta params
        int[] deltas = LCSSDistance.getInclusive10(0, seriesLength/4);
        // compute epsilon params
        double stdvFloor = dataStdv*0.2;
        double[] epsilons = LCSSDistance.getInclusive10(stdvFloor, dataStdv);
        
        int thisDelta;
        double thisEpsilon;
        double thisAcc;
        
        int bsfDelta = -1;
        double bsfEpsilon = 1;
        double bsfAcc = -1;
        LCSSDistance lcss;
        StringBuilder bsfSt = null;
        StringBuilder thisSt;
        
        for(int d = 0; d < deltas.length; d++){
            for(int e = 0; e < epsilons.length; e++){
                thisSt = new StringBuilder();
                thisDelta = deltas[d];
                thisEpsilon = epsilons[e];
                
                lcss = new LCSSDistance(thisDelta, thisEpsilon);
                thisAcc = crossValidate(data, 1, lcss,thisSt);
                
                if(thisAcc > bsfAcc){
                    bsfAcc = thisAcc;
                    bsfDelta = thisDelta;
                    bsfEpsilon=thisEpsilon;
                    bsfSt = thisSt;
                }else if(thisAcc == bsfAcc && thisDelta <= bsfDelta && thisEpsilon <= bsfEpsilon){
                    bsfAcc = thisAcc;
                    bsfDelta = thisDelta;
                    bsfEpsilon=thisEpsilon;
                    bsfSt = thisSt;
                }
            }
        }
        
        double[] lcss_1nn = {bsfAcc,bsfDelta,bsfEpsilon};
        FileWriter log = new FileWriter(OUTPUT_DIR_CV+"/"+dataName+"/cv_31_LCSS_1NN.txt");
        log.append(bsfAcc+","+bsfDelta+","+bsfEpsilon+"\n");
        log.append(bsfSt);
        log.close();
        return lcss_1nn;// accuracy, delta, epsilon
    }
    
    
    /*
     * Due to time constratints, we could not investigate all possible paramater options for
     * k = 1, 2, ..., 100. As a comprmise, we investigate 100 k values using the best LCSS
     * parameters found in the LCSS 1NN cross-validation.
     */
    public static double[] cv_32_LCSS_kNN(String dataName, Instances data, int delta, double epsilon) throws Exception{
        // get the 10 delta and 10 epsilon readings between the ranges specified by <citation>
        
        int bsfK = -1;
        double thisAcc = - 1;
        double bsfAcc = -1;
        LCSSDistance lcss;
        StringBuilder bsfSt = null;
        StringBuilder thisSt;
        
        for(int k = 1; k <= 100; k++){
            thisSt = new StringBuilder();
            lcss = new LCSSDistance(delta, epsilon);
            thisAcc = crossValidate(data, k, lcss,thisSt);
            
            if(thisAcc > bsfAcc){
                bsfAcc = thisAcc;
                bsfK = k;
                bsfSt = thisSt;
            }
        }
        double[] lcss_knn = {bsfAcc,delta,epsilon,bsfK};
        FileWriter log = new FileWriter(OUTPUT_DIR_CV+"/"+dataName+"/cv_32_LCSS_kNN.txt");
        log.append(bsfAcc+","+delta+","+epsilon+","+bsfK+"\n");
        log.append(bsfSt);
        log.close();
        return lcss_knn;
    }
    
    //</editor-fold>
        
    //<editor-fold defaultstate="collapsed" desc="Train/Test Methods">
    
    // generic train/test code
    public static double trainTest(Instances train, Instances test, int k, EuclideanDistance distanceMetric, StringBuilder st) throws Exception{

        kNN knn = new kNN(k);
        knn.setDistanceFunction(distanceMetric);
        knn.buildClassifier(train);
        
        int correct = 0;
        double decision, classValue;
        
        for(int i = 0; i < test.numInstances(); i++){
            classValue = test.instance(i).classValue();
            decision = knn.classifyInstance(test.instance(i));
            st.append(decision).append(",").append(classValue).append("\n");
            if(classValue==decision){
                correct++;
            }
        }
        
        return 100.0/test.numInstances()*correct;
    }
    
    // start of experiment-specific train/test code
    public static double trainTest_01_Euclidean_1NN(String dataName, Instances train, Instances test) throws Exception{
        StringBuilder st = new StringBuilder();
        EuclideanDistance euclid = new EuclideanDistance();
        euclid.setDontNormalize(true);
        
        double trainTest_01_euclidean_1nn = trainTest(train, test, 1, euclid, st);
        
        FileWriter log = new FileWriter(OUTPUT_DIR_TRAIN_TEST+"/"+dataName+"/trainTest_01_Euclidean_1NN.txt");
        log.append(trainTest_01_euclidean_1nn+"\n");
        log.append(st);
        log.close();
        return trainTest_01_euclidean_1nn;
    }
    
    public static double trainTest_02_DTW_fullWindow_1NN(String dataName, Instances train, Instances test) throws Exception{
        StringBuilder st = new StringBuilder();
        BasicDTW dtw = new BasicDTW();
        
        double trainTest_02_dtw_fullwindow_1nn = trainTest(train, test, 1, dtw, st);
        
        FileWriter log = new FileWriter(OUTPUT_DIR_TRAIN_TEST+"/"+dataName+"/trainTest_02_DTW_fullWindow_1NN.txt");
        log.append(trainTest_02_dtw_fullwindow_1nn+"\n");
        log.append(st);
        log.close();
        return trainTest_02_dtw_fullwindow_1nn;
    }
    
    
    public static double trainTest_03_DTW_optimalWindow_1NN(String dataName, Instances train, Instances test) throws Exception{
        // get best window with from cv file
        
        //1. check that the cv file exists:
        File cvFile = new File(OUTPUT_DIR_CV+"/"+dataName+"/cv_03_DTW_bestWindow_1NN.txt");
        if(!cvFile.exists()){
            throw new Exception("ERROR! CV hasn't been carried out for fully for "+cvFile.getName()+".");
        }
        
        Scanner scan = new Scanner(cvFile);
        scan.useDelimiter("\n");
        String[] resultsAndParams = scan.next().split(",");
        
        double bestR = Double.parseDouble(resultsAndParams[1].trim());
        
        StringBuilder st = new StringBuilder();
        SakoeChibaDTW sdtw = new SakoeChibaDTW(bestR);
        
        double trainTest_03_DTW_optimalWindow_1NN = trainTest(train, test, 1, sdtw, st);
        
        FileWriter log = new FileWriter(OUTPUT_DIR_TRAIN_TEST+"/"+dataName+"/trainTest_03_DTW_optimalWindow_1NN.txt");
        log.append(trainTest_03_DTW_optimalWindow_1NN+","+bestR+"\n");
        log.append(st);
        log.close();
        return trainTest_03_DTW_optimalWindow_1NN;
    }
    
    
    public static double trainTest_04_WDTW_1NN(String dataName, Instances train, Instances test) throws Exception{
        // get best window with from cv file
        
        //1. check that the cv file exists:
        File cvFile = new File(OUTPUT_DIR_CV+"/"+dataName+"/cv_04_WDTW_1NN.txt");
        if(!cvFile.exists()){
            throw new Exception("ERROR! CV hasn't been carried out for fully for "+cvFile.getName()+".");
        }
        
        Scanner scan = new Scanner(cvFile);
        scan.useDelimiter("\n");
        String[] resultsAndParams = scan.next().split(",");
        
        double bestG = Double.parseDouble(resultsAndParams[1].trim());
        
        StringBuilder st = new StringBuilder();
        WeightedDTW wdtw = new WeightedDTW(bestG);
        
        double trainTest_04_WDTW_1NN = trainTest(train, test, 1, wdtw, st);
        
        FileWriter log = new FileWriter(OUTPUT_DIR_TRAIN_TEST+"/"+dataName+"/trainTest_04_WDTW_1NN.txt");
        log.append(trainTest_04_WDTW_1NN+","+bestG+"\n");
        log.append(st);
        log.close();
        return trainTest_04_WDTW_1NN;
    }
    
    public static double trainTest_05_Euclidean_kNN(String dataName, Instances train, Instances test) throws Exception{
        // get best window with from cv file
        
        //1. check that the cv file exists:
        File cvFile = new File(OUTPUT_DIR_CV+"/"+dataName+"/cv_05_Euclidean_kNN.txt");
        if(!cvFile.exists()){
            throw new Exception("ERROR! CV hasn't been carried out for fully for "+cvFile.getName()+".");
        }
        
        Scanner scan = new Scanner(cvFile);
        scan.useDelimiter("\n");
        String[] resultsAndParams = scan.next().split(",");
        
        int bestK = Integer.parseInt(resultsAndParams[1].trim());
        
        StringBuilder st = new StringBuilder();
        EuclideanDistance euclid = new EuclideanDistance();
        euclid.setDontNormalize(true);
        
        double trainTest_05_Euclidean_kNN = trainTest(train, test, bestK, euclid, st);
        
        FileWriter log = new FileWriter(OUTPUT_DIR_TRAIN_TEST+"/"+dataName+"/trainTest_05_Euclidean_kNN.txt");
        log.append(trainTest_05_Euclidean_kNN+","+bestK+"\n");
        log.append(st);
        log.close();
        return trainTest_05_Euclidean_kNN;
    }
    
    public static double trainTest_06_DTW_fullWindow_kNN(String dataName, Instances train, Instances test) throws Exception{
        // get best window with from cv file
        
        //1. check that the cv file exists:
        File cvFile = new File(OUTPUT_DIR_CV+"/"+dataName+"/cv_06_DTW_fullWindow_kNN.txt");
        if(!cvFile.exists()){
            throw new Exception("ERROR! CV hasn't been carried out for fully for "+cvFile.getName()+".");
        }
        
        Scanner scan = new Scanner(cvFile);
        scan.useDelimiter("\n");
        String[] resultsAndParams = scan.next().split(",");
        
        int bestK = Integer.parseInt(resultsAndParams[1].trim());
        
        StringBuilder st = new StringBuilder();
        BasicDTW dtw = new BasicDTW();
        double trainTest_06_DTW_fullWindow_kNN = trainTest(train, test, bestK, dtw, st);
        
        FileWriter log = new FileWriter(OUTPUT_DIR_TRAIN_TEST+"/"+dataName+"/trainTest_06_DTW_fullWindow_kNN.txt");
        log.append(trainTest_06_DTW_fullWindow_kNN+","+bestK+"\n");
        log.append(st);
        log.close();
        return trainTest_06_DTW_fullWindow_kNN;
    }
    
    public static double trainTest_11_DTW_bestWindow_kNN(String dataName, Instances train, Instances test) throws Exception{
        // get best window with from cv file
        
        //1. check that the cv file exists:
        File cvFile = new File(OUTPUT_DIR_CV+"/"+dataName+"/cv_11_DTW_optimalWindow_kNN.txt");
        if(!cvFile.exists()){
            throw new Exception("ERROR! CV hasn't been carried out for fully for "+cvFile.getName()+".");
        }
        
        Scanner scan = new Scanner(cvFile);
        scan.useDelimiter("\n");
        String[] resultsAndParams = scan.next().split(",");
        
        int bestK = Integer.parseInt(resultsAndParams[1].trim());
        double bestR = Double.parseDouble(resultsAndParams[2].trim());
        
        StringBuilder st = new StringBuilder();
        SakoeChibaDTW sdtw = new SakoeChibaDTW(bestR);
        double trainTest_11_DTW_bestWindow_kNN = trainTest(train, test, bestK, sdtw, st);
        
        FileWriter log = new FileWriter(OUTPUT_DIR_TRAIN_TEST+"/"+dataName+"/trainTest_11_DTW_bestWindow_kNN.txt");
        log.append(trainTest_11_DTW_bestWindow_kNN+","+bestK+"\n");
        log.append(st);
        log.close();
        return trainTest_11_DTW_bestWindow_kNN;
    }
    
    public static double trainTest_12_WDTW_kNN(String dataName, Instances train, Instances test) throws Exception{
        // get best window with from cv file
        
        //1. check that the cv file exists:
        File cvFile = new File(OUTPUT_DIR_CV+"/"+dataName+"/cv_12_WDTW_kNN.txt");
        if(!cvFile.exists()){
            throw new Exception("ERROR! CV hasn't been carried out for fully for "+cvFile.getName()+".");
        }
        
        Scanner scan = new Scanner(cvFile);
        scan.useDelimiter("\n");
        String[] resultsAndParams = scan.next().split(",");
        
        int bestK = Integer.parseInt(resultsAndParams[1].trim());
        double bestG = Double.parseDouble(resultsAndParams[2].trim());
        
        StringBuilder st = new StringBuilder();
        WeightedDTW wdtw = new WeightedDTW(bestG);
        double trainTest_12_WDTW_kNN = trainTest(train, test, bestK, wdtw, st);
        
        FileWriter log = new FileWriter(OUTPUT_DIR_TRAIN_TEST+"/"+dataName+"/trainTest_12_WDTW_kNN.txt");
        log.append(trainTest_12_WDTW_kNN+","+bestK+"\n");
        log.append(st);
        log.close();
        return trainTest_12_WDTW_kNN;
    }
    
    public static double trainTest_21_DDTW_fullWindow_1NN(String dataName, Instances train, Instances test) throws Exception{
        if(!train.relationName().contains("derivative") || !test.relationName().contains("derivative")){
            throw new Exception("WARNING! Instances object does not include derivative in the relation name! If this is intentional, please either update relation name, or remove this exception from the code.");
        }
        StringBuilder st = new StringBuilder();
        BasicDTW dtw = new BasicDTW();
        
        double trainTest_21_DDTW_fullWindow_1NN = trainTest(train, test, 1, dtw, st);
        
        FileWriter log = new FileWriter(OUTPUT_DIR_TRAIN_TEST+"/"+dataName+"/trainTest_21_DDTW_fullWindow_1NN.txt");
        log.append(trainTest_21_DDTW_fullWindow_1NN+"\n");
        log.append(st);
        log.close();
        return trainTest_21_DDTW_fullWindow_1NN;
    }
    
    public static double trainTest_22_DDTW_optimalWindow_1NN(String dataName, Instances train, Instances test) throws Exception{
        // get best window with from cv file
        
        //1. check that the cv file exists:
        File cvFile = new File(OUTPUT_DIR_CV+"/"+dataName+"/cv_22_DDTW_bestWindow_1NN.txt");
        if(!cvFile.exists()){
            throw new Exception("ERROR! CV hasn't been carried out for fully for "+cvFile.getName()+".");
        }
        
        Scanner scan = new Scanner(cvFile);
        scan.useDelimiter("\n");
        String[] resultsAndParams = scan.next().split(",");
        
        double bestR = Double.parseDouble(resultsAndParams[1].trim());
        
        StringBuilder st = new StringBuilder();
        SakoeChibaDTW sdtw = new SakoeChibaDTW(bestR);
        
        double trainTest_22_DDTW_optimalWindow_1NN = trainTest(train, test, 1, sdtw, st);
        
        FileWriter log = new FileWriter(OUTPUT_DIR_TRAIN_TEST+"/"+dataName+"/trainTest_22_DDTW_optimalWindow_1NN.txt");
        log.append(trainTest_22_DDTW_optimalWindow_1NN+","+bestR+"\n");
        log.append(st);
        log.close();
        return trainTest_22_DDTW_optimalWindow_1NN;
    }
    
    public static double trainTest_23_WDDTW_1NN(String dataName, Instances train, Instances test) throws Exception{
        // get best window with from cv file
        
        //1. check that the cv file exists:
        File cvFile = new File(OUTPUT_DIR_CV+"/"+dataName+"/cv_23_WDDTW_1NN.txt");
        if(!cvFile.exists()){
            throw new Exception("ERROR! CV hasn't been carried out for fully for "+cvFile.getName()+".");
        }
        
        Scanner scan = new Scanner(cvFile);
        scan.useDelimiter("\n");
        String[] resultsAndParams = scan.next().split(",");
        
        double bestG = Double.parseDouble(resultsAndParams[1].trim());
        
        StringBuilder st = new StringBuilder();
        WeightedDTW wdtw = new WeightedDTW(bestG);
        
        double trainTest_23_WDDTW_1NN = trainTest(train, test, 1, wdtw, st);
        
        FileWriter log = new FileWriter(OUTPUT_DIR_TRAIN_TEST+"/"+dataName+"/trainTest_23_WDDTW_1NN.txt");
        log.append(trainTest_23_WDDTW_1NN+","+bestG+"\n");
        log.append(st);
        log.close();
        return trainTest_23_WDDTW_1NN;
    }
    
    public static double trainTest_24_DDTW_fullWindow_kNN(String dataName, Instances train, Instances test) throws Exception{
        // get best window with from cv file
        
        //1. check that the cv file exists:
        File cvFile = new File(OUTPUT_DIR_CV+"/"+dataName+"/cv_24_DDTW_fullWindow_kNN.txt");
        if(!cvFile.exists()){
            throw new Exception("ERROR! CV hasn't been carried out for fully for "+cvFile.getName()+".");
        }
        
        Scanner scan = new Scanner(cvFile);
        scan.useDelimiter("\n");
        String[] resultsAndParams = scan.next().split(",");
        
        int bestK = Integer.parseInt(resultsAndParams[1].trim());
        
        StringBuilder st = new StringBuilder();
        BasicDTW dtw = new BasicDTW();
        double trainTest_24_DDTW_fullWindow_kNN = trainTest(train, test, bestK, dtw, st);
        
        FileWriter log = new FileWriter(OUTPUT_DIR_TRAIN_TEST+"/"+dataName+"/trainTest_24_DDTW_fullWindow_kNN.txt");
        log.append(trainTest_24_DDTW_fullWindow_kNN+","+bestK+"\n");
        log.append(st);
        log.close();
        return trainTest_24_DDTW_fullWindow_kNN;
    }
    
    public static double trainTest_25_DDTW_bestWindow_kNN(String dataName, Instances train, Instances test) throws Exception{
        // get best window with from cv file
        
        //1. check that the cv file exists:
        File cvFile = new File(OUTPUT_DIR_CV+"/"+dataName+"/cv_25_DDTW_optimalWindow_kNN.txt");
        if(!cvFile.exists()){
            throw new Exception("ERROR! CV hasn't been carried out for fully for "+cvFile.getName()+".");
        }
        
        Scanner scan = new Scanner(cvFile);
        scan.useDelimiter("\n");
        String[] resultsAndParams = scan.next().split(",");
        
        int bestK = Integer.parseInt(resultsAndParams[1].trim());
        double bestR = Double.parseDouble(resultsAndParams[2].trim());
        
        StringBuilder st = new StringBuilder();
        SakoeChibaDTW sdtw = new SakoeChibaDTW(bestR);
        double trainTest_25_DDTW_bestWindow_kNN = trainTest(train, test, bestK, sdtw, st);
        
        FileWriter log = new FileWriter(OUTPUT_DIR_TRAIN_TEST+"/"+dataName+"/trainTest_25_DDTW_bestWindow_kNN.txt");
        log.append(trainTest_25_DDTW_bestWindow_kNN+","+bestK+"\n");
        log.append(st);
        log.close();
        return trainTest_25_DDTW_bestWindow_kNN;
    }
    
    public static double trainTest_26_WDDTW_kNN(String dataName, Instances train, Instances test) throws Exception{
        // get best window with from cv file
        
        //1. check that the cv file exists:
        File cvFile = new File(OUTPUT_DIR_CV+"/"+dataName+"/cv_26_WDTW_kNN.txt");
        if(!cvFile.exists()){
            throw new Exception("ERROR! CV hasn't been carried out for fully for "+cvFile.getName()+".");
        }
        
        Scanner scan = new Scanner(cvFile);
        scan.useDelimiter("\n");
        String[] resultsAndParams = scan.next().split(",");
        
        int bestK = Integer.parseInt(resultsAndParams[1].trim());
        double bestG = Double.parseDouble(resultsAndParams[2].trim());
        
        StringBuilder st = new StringBuilder();
        WeightedDTW wdtw = new WeightedDTW(bestG);
        double trainTest_26_WDDTW_kNN = trainTest(train, test, bestK, wdtw, st);
        
        FileWriter log = new FileWriter(OUTPUT_DIR_TRAIN_TEST+"/"+dataName+"/trainTest_26_WDDTW_kNN.txt");
        log.append(trainTest_26_WDDTW_kNN+","+bestK+"\n");
        log.append(st);
        log.close();
        return trainTest_26_WDDTW_kNN;
    }
    
    public static double trainTest_31_LCSS_1NN(String dataName, Instances train, Instances test) throws Exception{
        // get best window with from cv file
        
        //1. check that the cv file exists:
        File cvFile = new File(OUTPUT_DIR_CV+"/"+dataName+"/cv_31_LCSS_1NN.txt");
        if(!cvFile.exists()){
            throw new Exception("ERROR! CV hasn't been carried out for fully for "+cvFile.getName()+".");
        }
        
        Scanner scan = new Scanner(cvFile);
        scan.useDelimiter("\n");
        String[] resultsAndParams = scan.next().split(",");
        
        int bestDelta = Integer.parseInt(resultsAndParams[1].trim());
        double bestEpsilon = Double.parseDouble(resultsAndParams[2].trim());
        
        StringBuilder st = new StringBuilder();
        LCSSDistance lcss = new LCSSDistance(bestDelta, bestEpsilon);
        
        double trainTest_31_LCSS_1NN = trainTest(train, test, 1, lcss, st);
        
        FileWriter log = new FileWriter(OUTPUT_DIR_TRAIN_TEST+"/"+dataName+"/trainTest_31_LCSS_1NN.txt");
        log.append(trainTest_31_LCSS_1NN+","+bestDelta+","+bestEpsilon+"\n");
        log.append(st);
        log.close();
        return trainTest_31_LCSS_1NN;
    }
    
    public static double trainTest_32_LCSS_kNN(String dataName, Instances train, Instances test) throws Exception{
        // get best window with from cv file
        
        //1. check that the cv file exists:
        File cvFile = new File(OUTPUT_DIR_CV+"/"+dataName+"/cv_32_LCSS_kNN.txt");
        if(!cvFile.exists()){
            throw new Exception("ERROR! CV hasn't been carried out for fully for "+cvFile.getName()+".");
        }
        
        Scanner scan = new Scanner(cvFile);
        scan.useDelimiter("\n");
        String[] resultsAndParams = scan.next().split(",");
        
        int bestDelta = Integer.parseInt(resultsAndParams[1].trim());
        double bestEpsilon = Double.parseDouble(resultsAndParams[2].trim());
        int bestK = Integer.parseInt(resultsAndParams[3].trim());
        
        StringBuilder st = new StringBuilder();
        LCSSDistance lcss = new LCSSDistance(bestDelta, bestEpsilon);
        
        double trainTest_32_LCSS_kNN = trainTest(train, test, bestK, lcss, st);
        
        FileWriter log = new FileWriter(OUTPUT_DIR_TRAIN_TEST+"/"+dataName+"/trainTest_32_LCSS_kNN.txt");
        log.append(trainTest_32_LCSS_kNN+","+bestDelta+","+bestEpsilon+","+bestK+"\n");
        log.append(st);
        log.close();
        return trainTest_32_LCSS_kNN;
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="Complete Dataset CV and Train/Test Methods">
    public static void datasetCrossValidation(String dataName) throws Exception{
        
        initCv(dataName);
        Instances train_raw = loadData(DATA_DIR+"/"+dataName+"/"+dataName+"_TRAIN.arff");
        
        // get derivative training data (can store locally and read in to save computation time for large datasets)
        DerivativeFilter df = new DerivativeFilter();
        Instances train_derivative = df.process(train_raw);
        
        double   cv_01_euclidean_1nn = cv_01_Euclidean_1NN(dataName,train_raw);                                 // 01 Euclidean 1NN
        double   cv_02_dtw_fullWindow_1nn = cv_02_DTW_fullWindow_1NN(dataName,train_raw);                       // 02 DTW Full Window 1NN
        double[] cv_03_dtw_cvWindow_1nn = cv_03_DTW_bestWindow_1NN(dataName,train_raw);                         // 03 DTW variable window 1NN (try all possible values of R from 0% to 100% in increments of 1%)
        double[] cv_04_wdtw_1nn = cv_04_WDTW_1NN(dataName,train_raw);                                           // 04 Weighted DTW with cv to find the optimal weight, g. Possible values for g range from 0 to 1 in increments of 0.01
        double[] cv_05_euclidean_knn = cv_05_Euclidean_kNN(dataName,train_raw);                                 // 05 Euclidean kNN (k = 1, 2, ..., 100)
        double[] cv_06_dtw_fullWindow_knn = cv_06_DTW_fullWindow_kNN(dataName,train_raw);                       // 06 DTW Full Window kNN (k = 1, 2, ..., 100)
        
        double[] cv_11_dtw_optimalWindow_knn = cv_11_DTW_optimalWindow_kNN(dataName,train_raw);                 // 11 DTW Optimal Window kNN (r = 0, 0.01, 0.02, ..., 1) (k = 1, 2, ..., 100)
        double[] cv_12_wdtw_knn = cv_12_WDTW_kNN(dataName,train_raw);                                           // 12 WDTW kNN (g = 0, 0.01, 0.02, ..., 1) (k = 1, 2, ..., 100)
        
        double   cv_21_ddtw_fullWindow_1nn = cv_21_DDTW_fullWindow_1NN(dataName,train_derivative);              // 21 Derivative DTW Full Window 1NN
        double[] cv_22_ddtw_cvWindow_1nn = cv_22_DDTW_bestWindow_1NN(dataName,train_derivative);                // 22 Derivative DTW Variable Window 1NN (r 0-100%, increaments of 1%)
        double[] cv_23_wdtw_1nn = cv_23_WDDTW_1NN(dataName,train_derivative);                                   // 23 Erighted Derivative DTW 1NN (g 0-1, increments of 0.01)
        double[] cv_24_ddtw_fullWindow_knn = cv_24_DDTW_fullWindow_kNN(dataName,train_derivative);              // 24 Derivative DTW Full Window kNN (k 1-100, increments of 1)
        double[] cv_25_ddtw_optimalWindow_knn = cv_25_DDTW_optimalWindow_kNN(dataName,train_derivative);        // 25 Derivative DTW Variable Window
        double[] cv_26_wddtw_knn = cv_26_WDTW_kNN(dataName,train_derivative);
        
        double[] cv_31_lcss_1nn = cv_31_LCSS_1NN(dataName,train_raw);
        double[] cv_32_lcss_knn = cv_32_LCSS_kNN(dataName,train_raw,(int)cv_31_lcss_1nn[1],cv_31_lcss_1nn[2]);
        
        
        
        // print results
        /***** RAW DATA ****/
        System.out.printf("Euclidean 1NN:%33.3f%n",cv_01_euclidean_1nn);
        System.out.printf("DTW Full Window 1NN:%27.3f%n",cv_02_dtw_fullWindow_1nn);
        System.out.printf("DTW Optimal Window 1NN (r=%1.2f):%15.3f%n",cv_03_dtw_cvWindow_1nn[1],cv_03_dtw_cvWindow_1nn[0]);
        System.out.printf("WDTW 1NN (g=%1.2f):%29.3f%n",cv_04_wdtw_1nn[1],cv_04_wdtw_1nn[0]);
        System.out.printf("Euclidean kNN (k=%3.0f):%25.3f%n",cv_05_euclidean_knn[1],cv_05_euclidean_knn[0]);
        System.out.printf("DTW Full Window kNN (k=%3.0f):%19.3f%n",cv_06_dtw_fullWindow_knn[1],cv_06_dtw_fullWindow_knn[0]);
        System.out.printf("DTW Optimal Window kNN (k=%3.0f, r=%1.2f):%9.3f%n",cv_11_dtw_optimalWindow_knn[1],cv_11_dtw_optimalWindow_knn[2],cv_11_dtw_optimalWindow_knn[0]);
        System.out.printf("WDTW kNN (k=%3.0f, g=%1.2f):%22.3f%n",cv_12_wdtw_knn[1],cv_12_wdtw_knn[2],cv_12_wdtw_knn[0]);
        
        /***** DERIVATIVE TRANSFORMED DATA ****/
        System.out.printf("DDTW Full Window 1NN:%26.3f%n",cv_21_ddtw_fullWindow_1nn);
        System.out.printf("DDTW Optimal Window 1NN (r=%1.2f):%14.3f%n",cv_22_ddtw_cvWindow_1nn[1],cv_22_ddtw_cvWindow_1nn[0]);
        System.out.printf("WDDTW 1NN (g=%1.2f):%28.3f%n",cv_23_wdtw_1nn[1],cv_23_wdtw_1nn[0]);
        System.out.printf("DDTW Full Window kNN (k=%3.0f):%18.3f%n",cv_24_ddtw_fullWindow_knn[1],cv_24_ddtw_fullWindow_knn[0]);
        System.out.printf("DDTW Optimal Window kNN (k=%3.0f, r=%1.2f):%7.3f%n",cv_25_ddtw_optimalWindow_knn[1],cv_25_ddtw_optimalWindow_knn[2],cv_25_ddtw_optimalWindow_knn[0]);
        System.out.printf("WDDTW kNN (k=%3.0f, g=%1.2f):%21.3f%n",cv_26_wddtw_knn[1],cv_26_wddtw_knn[2],cv_26_wddtw_knn[0]);
        
        /***** LCSS ******/
        System.out.printf("LCSS (d=%3.0f, e=%1.3f) 1NN:%21.3f%n",cv_31_lcss_1nn[1],cv_31_lcss_1nn[2],cv_31_lcss_1nn[0]);
        System.out.printf("LCSS (d=%3.0f, e=%1.3f) kNN (k=%3.0f):%13.3f%n",cv_32_lcss_knn[1],cv_32_lcss_knn[2],cv_32_lcss_knn[3],cv_32_lcss_knn[0]);
        
    }
    

    
    public static void datasetTrainTest(String dataName) throws Exception{
        // Pre-requisite of train/test classification is that necessary params have been found in CV stage.
        // Therefore, check to see if CV has been carried out previously. If not, begin then CV automatically.
        File cvDir = new File(OUTPUT_DIR_CV);
        if(!cvDir.exists()){
            System.out.println("Cross-validation for "+dataName+" doesn't appear to have taken place in '"+OUTPUT_DIR_CV+"'. Starting cross-validation.");
            datasetCrossValidation(dataName);
        }
        
        initTrainTest(dataName);
        
        Instances train_raw = loadData(DATA_DIR+"/"+dataName+"/"+dataName+"_TRAIN.arff");
        Instances test_raw = loadData(DATA_DIR+"/"+dataName+"/"+dataName+"_TEST.arff");
        
        DerivativeFilter df = new DerivativeFilter();
        Instances train_derivative = df.process(train_raw);
        df = new DerivativeFilter();
        Instances test_derivative = df.process(test_raw);
        
        /***** RAW DATA ****/
        System.out.printf("Euclidean 1NN:%33.3f%n",trainTest_01_Euclidean_1NN(dataName, train_raw, test_raw));
        System.out.printf("DTW Full Window 1NN:%27.3f%n",trainTest_02_DTW_fullWindow_1NN(dataName, train_raw, test_raw));
        System.out.printf("DTW Best Window 1NN:%27.3f%n",trainTest_03_DTW_optimalWindow_1NN(dataName, train_raw, test_raw));
        System.out.printf("WDTW 1NN:%38.3f%n",trainTest_04_WDTW_1NN(dataName, train_raw, test_raw));
        System.out.printf("Euclidean kNN:%33.3f%n",trainTest_05_Euclidean_kNN(dataName, train_raw, test_raw));
        System.out.printf("DTW Full Window kNN:%27.3f%n",trainTest_06_DTW_fullWindow_kNN(dataName, train_raw, test_raw));
        System.out.printf("DTW Best Window kNN:%27.3f%n",trainTest_11_DTW_bestWindow_kNN(dataName, train_raw, test_raw));
        System.out.printf("WDTW kNN:%38.3f%n",trainTest_12_WDTW_kNN(dataName, train_raw, test_raw));
        
        /***** DERIVATIVE TRANSFORMED DATA ****/
        System.out.printf("DDTW Full Window 1NN:%26.3f%n",trainTest_21_DDTW_fullWindow_1NN(dataName, train_derivative, test_derivative));
        System.out.printf("DDTW Best Window 1NN:%26.3f%n",trainTest_22_DDTW_optimalWindow_1NN(dataName, train_derivative, test_derivative));
        System.out.printf("WDDTW 1NN:%37.3f%n",trainTest_23_WDDTW_1NN(dataName, train_derivative, test_derivative));
        System.out.printf("DDTW Full Window kNN:%26.3f%n",trainTest_24_DDTW_fullWindow_kNN(dataName, train_derivative, test_derivative));
        System.out.printf("DDTW Best Window kNN:%26.3f%n",trainTest_25_DDTW_bestWindow_kNN(dataName, train_derivative, test_derivative));
        System.out.printf("WDDTW kNN:%37.3f%n",trainTest_26_WDDTW_kNN(dataName, train_derivative, test_derivative));
        
        /***** LCSS ******/
        System.out.printf("LCSS 1NN:%38.3f%n",trainTest_31_LCSS_1NN(dataName, train_raw, test_raw));
        System.out.printf("LCSS kNN:%38.3f%n",trainTest_32_LCSS_kNN(dataName, train_raw, test_raw));
        
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Result Parsing for CV and Train/Test">
    public static void printPreCalculatedCvResults(String dataset) throws Exception{
        Scanner scan;
        
        File[] cvResults = new File(OUTPUT_DIR_CV+"/"+dataset).listFiles();
        String classifierName;
        String[] resultLineParts;
        for(int i = 0; i < cvResults.length; i++){
            classifierName = cvResults[i].getName().substring(6).replace(".txt", "").replaceAll("_", " ");
            scan = new Scanner(cvResults[i]);
            scan.useDelimiter("\n");
            resultLineParts = scan.next().split(",");
            
            double accuracy = Double.parseDouble(resultLineParts[0]);
            System.out.printf(classifierName+"%"+(40-classifierName.length())+".3f\n",accuracy);
        }
    }
    
    public static void printPrecalculatedTrainTestResults(String dataset) throws Exception{
        Scanner scan;
        
        File[] cvResults = new File(OUTPUT_DIR_TRAIN_TEST+"/"+dataset).listFiles();
        String classifierName;
        String[] resultLineParts;
        for(int i = 0; i < cvResults.length; i++){
            classifierName = cvResults[i].getName().substring(13).replace(".txt", "").replaceAll("_", " ");
            scan = new Scanner(cvResults[i]);
            scan.useDelimiter("\n");
            resultLineParts = scan.next().split(",");
            
            double accuracy = Double.parseDouble(resultLineParts[0]);
            System.out.printf(classifierName+"%"+(40-classifierName.length())+".3f\n",accuracy);
        }
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Ensemble Classification Methods">
    public static double ensembleClassification_best(String dataName) throws Exception{
        // prerequisites:
        //      1. Cv must have been carried out for all classifiers
        //      2. Train/Test must have been carried out for all classifiers
        
        // Step 1.  Get cv accuracies for all classifiers (either single classifier, or multiple if classifiers are tied)
        double[] cvAccuracies = getCvAccuracies(dataName);
        
        // Step 2:  Find the best classifier according to cv accuracies.
        //          in case there isn't a single best classifier, need to store the id's of each of the best classifier
        //          justification: to chose a random classifier from the best on each classification decision, otherwise
        //          random selection must happen before classification, and then only one classifier would be used.
        
        ArrayList<Integer> bestClassifiers = new ArrayList<Integer>();
        double bestAccuracy = -1;
        
        for(int i = 0; i < cvAccuracies.length;i++){
            if(cvAccuracies[i]>bestAccuracy){
                bestClassifiers = new ArrayList<Integer>();
                bestClassifiers.add(i);
                bestAccuracy = cvAccuracies[i];
            }else if(cvAccuracies[i]==bestAccuracy){
                bestClassifiers.add(i);
            }
        }
        
        Instances test = loadData(DATA_DIR+"/"+dataName+"/"+dataName+"_TEST.ARFF");
        
        //3. get test predictions for all classifiers
        double[][] predictions = getTestPredictions(dataName, test);
        
        //4. get actual class vlaues
        double[] actualClassValues = getClassValues(test);
        
        // if there is more than one best classifier, we must radomly pick one of this subset each time we make
        // a classification decision.
        boolean moreThanOneBest = false;
        if(bestClassifiers.size() > 1){
            moreThanOneBest = true;
        }
        Random r = new Random();
        
        
        double[] ensemblePredictions = new double[actualClassValues.length];
        
        for(int i = 0; i < actualClassValues.length; i++){
            if(moreThanOneBest){
                ensemblePredictions[i] = predictions[i][bestClassifiers.get(r.nextInt(bestClassifiers.size()))];
            }else{
                ensemblePredictions[i] = predictions[i][bestClassifiers.get(0)];
            }
        }
        
        int correct = 0;
        for(int i = 0; i < ensemblePredictions.length; i++){
//            System.out.println(ensemblePredictions[i]+","+actualClassValues[i]);
            if(ensemblePredictions[i]==actualClassValues[i]){
                correct++;
            }
        }
//        System.out.println("Best: "+100.0/test.numInstances()*correct);
        return 100.0/test.numInstances()*correct;
        
    }
    
    
    public static double ensembleClassification_equal(String dataName) throws Exception{
        // for each instances, we get the predictions for all classifiers. Then we count the occurances
        // of each class value as votes, and the class value with the highest vote is selected as the
        // classification decision. In cases where there is no majority, ties are split randomly
        
        // 1.   We don't need to use the cv accuracies with this ensemble strategy, so we can go straight
        //      to getting the class values and test predictions
        Instances test = loadData(DATA_DIR+"/"+dataName+"/"+dataName+"_TEST.arff");
        double[] actualClassValues = getClassValues(test);
        double[][] predictions = getTestPredictions(dataName, test);
        
        
        // 2.   For each instances, count all votes. When voting has finished, extact the most frequent
        //      class value(s) and make classification decision
        TreeMap<Double, Integer> votes;
        ArrayList<Double> majorityClasses;
        Random r = new Random();
        int count, bsfCount;
        
        double[] ensemblePredictions = new double[actualClassValues.length];
        
        for(int i = 0; i < actualClassValues.length; i++){      // for instance i
            votes = new TreeMap<Double, Integer>();
            for(int j = 0; j < predictions[i].length; j++){     // for classifier j
                if(votes.containsKey(predictions[i][j])){       // add vote from classifier j on instance i
                    count = votes.get(predictions[i][j]);
                    count+=1;
                    votes.put(predictions[i][j], count);
                }else{
                    votes.put(predictions[i][j], 1);
                }
            }
            
            // extract top class(es)
            bsfCount = -1;
            majorityClasses = new ArrayList<Double>();
            for(Double d:votes.keySet()){
                if(votes.get(d) > bsfCount){                    // if better than bsf, must be better than all others so reinitialise store
                    majorityClasses = new ArrayList<Double>();
                    majorityClasses.add(d);
                    bsfCount = votes.get(d);
                }else if(votes.get(d)==bsfCount){               // if equal to bsf, store but don't remove previous.
                    majorityClasses.add(d);
                }
            }
            
            // if there is a dominant class, no need to randomly select (will be @ index 0). Else, randomly pick on of the best class values
            // as they must be equally represented to be included in the majority classes store.
            if(majorityClasses.size()==1){
                ensemblePredictions[i] = majorityClasses.get(0);
            }else{
                ensemblePredictions[i] = majorityClasses.get(r.nextInt(majorityClasses.size()));
            }
        }
        
        int correct = 0;
        for(int i = 0; i < ensemblePredictions.length; i++){
            if(ensemblePredictions[i]==actualClassValues[i]){
                correct++;
            }
        }
//        System.out.println("Equal: "+100.0/test.numInstances()*correct);
        return 100.0/test.numInstances()*correct;
        
    }
    
    public static double ensembleClassification_proportional(String dataName) throws Exception{
        // for each classifier, we calcaulte a voting weight according to cv performance. The classifier then proceeds much like the ensemble_equal
        // strategy; for each instance, a vote is taken from each classifier. However, rather than each classifier having an equal vote (i.e. 1 each),
        // the vote is adjusted according to the classifier's weighting.
        
        // 1. get cv accuracies for all classifiers
        double[] cvAccuracies = getCvAccuracies(dataName);
        
        // 2. calculate weights
        double[] weights = new double[cvAccuracies.length];
        
        // get sum of accuracy
        double sumOfAccuracies = 0;
        for(int i = 0; i < cvAccuracies.length;i++){
            sumOfAccuracies+=cvAccuracies[i];
        }
        
        // assign values to weights according to cv accuracy
        for(int i = 0; i < cvAccuracies.length; i++){
            weights[i] = 100.0/sumOfAccuracies*cvAccuracies[i];
        }
        
        // 3. get predictions
        Instances test = loadData(DATA_DIR+"/"+dataName+"/"+dataName+"_TEST.arff");
        double[][] predictions = getTestPredictions(dataName, test);
        double[] actualClassValues = getClassValues(test);
        
        // 4. get classification decisions
        TreeMap<Double, Double> classVotes;
        ArrayList<Double> electedClasses;
        double vote;
        double bsfVote;
        
        double[] ensemblePredictions = new double[test.numInstances()];
        Random r = new Random();
        
        for(int i = 0;i < test.numInstances(); i++){                // for each instance i
            classVotes = new TreeMap<Double, Double>();
            for(int j = 0; j < predictions[i].length; j++){         // for each classifier j
                if(classVotes.containsKey(predictions[i][j])){      // if class value has already been voted for, update vote
                    vote = classVotes.get(predictions[i][j]);
                    vote += weights[j];
                    classVotes.put(predictions[i][j], vote);
                }else{                                              // else, add a new entry for this class value
                    classVotes.put(predictions[i][j], weights[j]);
                }
            }
            
            bsfVote = -1;
            electedClasses = new ArrayList<Double>();
            for(Double d:classVotes.keySet()){
                if(classVotes.get(d) > bsfVote){                    // if best so far, re-initialise the store and add this class
                    electedClasses = new ArrayList<Double>();
                    electedClasses.add(d);
                    bsfVote = classVotes.get(d);
                }else if(classVotes.get(d) == bsfVote){             // else if it is equal to the best so far, retain original value(s) and add this too.
                    electedClasses.add(d);
                }
            }
            
            // if there is a dominant class value, pick that. Else, randomly pick one from the set of best classes
            if(electedClasses.size()==1){
                ensemblePredictions[i] = electedClasses.get(0);
            }else{
                ensemblePredictions[i] = electedClasses.get(r.nextInt(electedClasses.size()));
            }
            
        }
        
        int correct = 0;
        for(int i = 0; i < ensemblePredictions.length; i++){
            if(ensemblePredictions[i]==actualClassValues[i]){
                correct++;
            }
        }
//        System.out.println("Proportional: "+100.0/test.numInstances()*correct);
        return 100.0/test.numInstances()*correct;
        
    }
    
    
    public static double ensembleClassification_significant(String dataName) throws Exception{
        // 1. get cv accuracies for all classifiers
        Instances train = loadData(DATA_DIR+"/"+dataName+"/"+dataName+"_TRAIN.arff");
        
        double[] cvAccuracies = getCvAccuracies(dataName);
        
        // 2. calculate weights, with McNemar's taken into consideration
        double[] weights = new double[cvAccuracies.length];
        int[] mcNemarsInclusion = mcNemars(dataName, train);
        
        // get sum of accuracy
        double sumOfAccuracies = 0;
        for(int i = 0; i < cvAccuracies.length;i++){
            if(mcNemarsInclusion[i]==1){
                sumOfAccuracies+=cvAccuracies[i];
            }
        }
        
        // assign values to weights according to cv accuracy
        for(int i = 0; i < cvAccuracies.length; i++){
            if(mcNemarsInclusion[i]==1){
                weights[i] = 100.0/sumOfAccuracies*cvAccuracies[i];
            }else{
                weights[i]=0;
            }
        }
        
        // 3. get predictions
        Instances test = loadData(DATA_DIR+"/"+dataName+"/"+dataName+"_TEST.arff");
        double[][] predictions = getTestPredictions(dataName, test);
        double[] actualClassValues = getClassValues(test);
        
        // 4. get classification decisions
        TreeMap<Double, Double> classVotes;
        ArrayList<Double> electedClasses;
        double vote;
        double bsfVote;
        
        double[] ensemblePredictions = new double[test.numInstances()];
        Random r = new Random();
        
        for(int i = 0;i < test.numInstances(); i++){                // for each instance i
            classVotes = new TreeMap<Double, Double>();
            for(int j = 0; j < predictions[i].length; j++){         // for each classifier j
                if(classVotes.containsKey(predictions[i][j])){      // if class value has already been voted for, update vote
                    vote = classVotes.get(predictions[i][j]);
                    vote += weights[j];
                    classVotes.put(predictions[i][j], vote);
                }else{                                              // else, add a new entry for this class value
                    classVotes.put(predictions[i][j], weights[j]);
                }
            }
            
            bsfVote = -1;
            electedClasses = new ArrayList<Double>();
            for(Double d:classVotes.keySet()){
                if(classVotes.get(d) > bsfVote){                    // if best so far, re-initialise the store and add this class
                    electedClasses = new ArrayList<Double>();
                    electedClasses.add(d);
                    bsfVote = classVotes.get(d);
                }else if(classVotes.get(d) == bsfVote){             // else if it is equal to the best so far, retain original value(s) and add this too.
                    electedClasses.add(d);
                }
            }
            
            // if there is a dominant class value, pick that. Else, randomly pick one from the set of best classes
            if(electedClasses.size()==1){
                ensemblePredictions[i] = electedClasses.get(0);
            }else{
                ensemblePredictions[i] = electedClasses.get(r.nextInt(electedClasses.size()));
            }
            
        }
        
        int correct = 0;
        for(int i = 0; i < ensemblePredictions.length; i++){
            if(ensemblePredictions[i]==actualClassValues[i]){
                correct++;
            }
        }
//        System.out.println("Significant: "+100.0/test.numInstances()*correct);
        return 100.0/test.numInstances()*correct;
        
    }
   

    public static int[] mcNemars(String dataName, Instances train)throws Exception{
        
        //1. Get the cv predictions for each classifier
        double[][] cvPredictions = getCVPredictions(dataName, train);
        
        //2. get cv accuracies
        double[] cvAccuracies = getCvAccuracies(dataName);
        
        //3. pick the best classifier to build ensemble around
        
        double bsfAccuracy = -1;
        ArrayList<Integer> bestClassifierIds = null;
        for(int i = 0; i < cvAccuracies.length;i++){
            if(cvAccuracies[i] > bsfAccuracy){          // new single-best classifier, so reinitialise store and add the id of this classifier
                bsfAccuracy = cvAccuracies[i];
                bestClassifierIds = new ArrayList<Integer>();
                bestClassifierIds.add(i);
            }else if(cvAccuracies[i] == bsfAccuracy){   // equals best so far, so retain previous best classifier(s) and add this id to the store
                bestClassifierIds.add(i);
            }
        }
        
        int bestClassifierId = -1;

        // split ties randomly
        if(bestClassifierIds.size() > 1){
            Random r = new Random();
            bestClassifierId = bestClassifierIds.get(r.nextInt(bestClassifierIds.size()));
        }else{
            bestClassifierId = bestClassifierIds.get(0);
        }
	
        double[] bestClassifierPredictions = new double[train.numInstances()];

        for(int i = 0; i < train.numInstances();i++){
            bestClassifierPredictions[i] = cvPredictions[i][bestClassifierId];
        }
        
        int numClassifiers = cvAccuracies.length;
        double[] actualClassValues = getClassValues(train);

        int[] logicalOutput = new int[numClassifiers];
        for(int c = 0; c < numClassifiers; c++){
            if(c==bestClassifierId){
                logicalOutput[c] = 1;
            }else if(cvAccuracies[c]==100){
                logicalOutput[c] = 1;   // if classifier isn't picked as the best and still has 100%, must be equivilient, as best must also be 100% so all class decision were the same 
            }else{
                // create contingency table 
            
                // best classifier = classifier a
                // other = classificer b

                int wrongByBoth = 0; // top-left
                int rightByAWrongByB = 0; // bottom-left
                int wrongByaRightByB = 0; // top-right
                int rightByBoth = 0; // bottom-right

                double actualClass, a, b;
                for(int i = 0; i < train.numInstances();i++){
                    actualClass = actualClassValues[i];
                    a = bestClassifierPredictions[i];
                    b = cvPredictions[i][c];

                    if(a!=actualClass && b!=actualClass){
                        wrongByBoth++;
                    }else if(a==actualClass && b!=actualClass){
                        rightByAWrongByB++;
                    }else if(a!=actualClass&&b==actualClass){
                        wrongByaRightByB++;
                    }else if(a==actualClass && b==actualClass){
                        rightByBoth++;
                    }
                }
                if(wrongByBoth+rightByAWrongByB+wrongByaRightByB+rightByBoth!=train.numInstances()){
                    throw new Exception("Count of instances is incorrect. Please ensure inputs are correct");
                }

                if(rightByAWrongByB+wrongByaRightByB==0){
                    logicalOutput[c] = 1; // equivilent to the best classifier, so include it to add weight to proportional votes
                }else{
                    double chiPart = (Math.abs(wrongByaRightByB-rightByAWrongByB)-1);
                    double chi = (chiPart*chiPart)/(wrongByaRightByB+rightByAWrongByB);

                    if(chi >= 6.634897){  // Alpha = 0.01
                        logicalOutput[c] = 0;
                    }else{
                        logicalOutput[c] = 1;
                    }
                }
            }
        }

        return logicalOutput;
    }
    
    public static void print10RunEnsembles(String dataset) throws Exception{
        double best = 0;
        double equal = 0;
        double prop = 0;
        double sig = 0;
        
        for(int i = 0; i < 10; i++){
            best  += ensembleClassification_best(dataset);
            equal += ensembleClassification_equal(dataset);
            prop  += ensembleClassification_proportional(dataset);
            sig   += ensembleClassification_significant(dataset);
        }
        
        System.out.printf("Best%36.3f\n",(best/10));
        System.out.printf("Equal%35.3f\n",(equal/10));
        System.out.printf("Proportional%28.3f\n",(prop/10));
        System.out.printf("Significant%29.3f\n",(sig/10));
    }
    
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Utility Methods">
    public static Instances loadData(String fileName){
        Instances data = null;
        try{
            FileReader r;
            r = new FileReader(fileName);
            data = new Instances(r);
            
            data.setClassIndex(data.numAttributes() - 1);
        } catch(Exception e){
            System.out.println(" Error =" + e + " in method loadData");
        }
        return data;
    }
    //</editor-fold>
        
    //<editor-fold defaultstate="collapsed" desc="Helper Methods">
    private static double[] getCvAccuracies(String dataName) throws Exception{
        
        // get cv accuracies for each classifier
        File cvFolder = new File(OUTPUT_DIR_CV+"/"+dataName);
        File[] cvFiles = cvFolder.listFiles();
        double[] cvAccuracies = new double[16];
        
        for(int i = 0; i < 16; i++){
            File classifierFile = cvFiles[i];
            Scanner scan = new Scanner(classifierFile);
            scan.useDelimiter("\n");
            cvAccuracies[i] = Double.parseDouble(scan.next().split(",")[0].trim());
        }
        
        return cvAccuracies;
    }
    
    
    private static double[][] getCVPredictions(String dataName, Instances train) throws Exception{
        
        // read in train/test predictions
        File cvFolder = new File(OUTPUT_DIR_CV+"/"+dataName);
        File[] cvFiles = cvFolder.listFiles();
        
        
        double[][] predictions = new double[train.numInstances()][cvFiles.length];
        
        for(int c = 0; c < cvFiles.length; c++){
            File cvFile = cvFiles[c];
            Scanner scan = new Scanner(cvFile);
            scan.useDelimiter("\n");
            scan.next(); // header
            
            int idx = 0;
            while(scan.hasNext()){
                String[] resultLineParts = scan.next().split(",");
                double prediction = Double.parseDouble(resultLineParts[0]);
                double classVal = Double.parseDouble(resultLineParts[1]);
                if(classVal!=train.instance(idx).classValue()){
                    throw new Exception("Class value mismatch! Found:"+classVal+", expected:"+train.instance(idx).classValue());
                }
                predictions[idx][c] = prediction;
                idx++;
            }
            if(idx!=train.numInstances()){
                throw new Exception("Incorrect number of instances! Found:"+idx+", expected: "+train.numInstances());
            }
        }
        return predictions;
        
    }
    
    
    private static double[][] getTestPredictions(String dataName, Instances test) throws Exception{
        
        // read in train/test predictions
        File testFolder = new File(OUTPUT_DIR_TRAIN_TEST+"/"+dataName);
        File[] testFiles = testFolder.listFiles();
        
        
        double[][] predictions = new double[test.numInstances()][testFiles.length];
        
        for(int c = 0; c < testFiles.length; c++){
            File trainTestFile = testFiles[c];
            Scanner scan = new Scanner(trainTestFile);
            scan.useDelimiter("\n");
            scan.next(); // header
            
            int idx = 0;
            while(scan.hasNext()){
                String[] resultLineParts = scan.next().split(",");
                double prediction = Double.parseDouble(resultLineParts[0]);
                double classVal = Double.parseDouble(resultLineParts[1]);
                if(classVal!=test.instance(idx).classValue()){
                    throw new Exception("Class value mismatch! Found:"+classVal+", expected:"+test.instance(idx).classValue());
                }
                predictions[idx][c] = prediction;
                idx++;
            }
            if(idx!=test.numInstances()){
                throw new Exception("Incorrect number of instances! Found:"+idx+", expected: "+test.numInstances());
            }
        }
        return predictions;
        
    }
    
    public static double[] getClassValues(Instances input){
        double[] classValues = new double[input.numInstances()];
        for(int i = 0; i < input.numInstances();i++){
            classValues[i]=input.instance(i).classValue();
        }
        return classValues;
    }
    //</editor-fold>

    public static void main(String[] args) {

        // A main method to carry out CV, train/test, and ensemble classification for a given dataset name. Before running, please ensure that 
        // the static fields at the start of this class suit your needs (i.e. correct output location and input Instances data location).
        
        // Contained in this method are two options that are automatically selected according to the results that are in place:
        //      1. If no experiments have been carried out for the dataset specified in the field dataName:
        String dataName = "ItalyPowerDemand";
        //         then all cross-validation and train/test experiments will be carried out when the code is executed.
        //      2. If 'dataName' has previously been processed and the results are stored under the paths listed in the 'OUTPUT_DIR' fields,
        //         summary methods will be read in to parse the existing results. This is much faster on all datasets, as classification will 
        //         not need to be repeated unecessarily.
        
        try{
            // Part 1: Cross-Validation
            // If CV hasn't been carried out, perform CV (ASSUMPTION: if cv folder exists, cv has been carried out FULLY)
            // Else, print CV results
            System.out.println("Cross-Validation Results");
            System.out.println("----------------------------------------");
            File cvResultsDir = new File(OUTPUT_DIR_CV+"/"+dataName);
            if(cvResultsDir.exists()){
                printPreCalculatedCvResults(dataName);
            }else{
                datasetCrossValidation(dataName);
            }
            System.out.println();
            
            // Part 2: Train/Test
            // If train/test hasn't been carried out, perform Train/Test (ASSUMPTION: if results dir exists, Train/Test has been fully carried out)
            // Else, print Train/Test results
            System.out.println("Train/Test Results");
            System.out.println("----------------------------------------");
            File trainTestResultsDir = new File(OUTPUT_DIR_TRAIN_TEST+"/"+dataName);
            if(trainTestResultsDir.exists()){
                printPrecalculatedTrainTestResults(dataName);
            }else{
                datasetTrainTest(dataName);
            }
            System.out.println();
            
            // Part 3: Ensembles
            // CV and Train/Test results must be in place to reach this statement under the intended conditions when this code was released. If code has
            // been modified, please ensure CV and train/test classification has been carried out before running ensembles (CV needed for weighting, 
            // Train/Test for final classification)
            
            // Note: in method below, ensembling is carried out 10 times for each strategy and the average for each is reported. This is justified because 
            // it is not unlikely that classifiers of very similar natures (i.e. all time-domain NN) may have very similar CV performance on some datasets, 
            // therefore leading to slightly different ensembles when ties are settled randomly. Since ensembling takes place on pre-calculated results, it is very
            // time efficient and allows us to average over multilpe runs to smooth the results to obtain consistent results from multiple runs. 
            System.out.println("Ensemble Results");
            System.out.println("----------------------------------------");
            print10RunEnsembles(dataName);

        }catch(Exception e){
            e.printStackTrace();
        }
    }









}
