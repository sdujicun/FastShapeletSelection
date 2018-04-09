package tsc_algorithms;
//import grabocka_reproduction.LearnShapeletsGeneralized;
import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import development.DataSets;
import utilities.StatisticalUtilities;
import utilities.InstanceTools;
import static utilities.InstanceTools.fromWekaInstancesArray;
import static utilities.StatisticalUtilities.calculateSigmoid;
//import static utilities.StatisticalUtilities.normalize;
import static utilities.StatisticalUtilities.normalize2D;
import weka.classifiers.*;
import weka.clusterers.SimpleKMeans;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

public class LearnShapeletsFeb2015Version extends AbstractClassifier implements ParameterSplittable{

    long seed;
    
    // length of a time-series 
    public int seriesLength;
    // length of shapelet
    public int lengthsOfShapelet[];
    public double percentageOfSeriesLength;
    // number of latent patterns
    public int numLatentPatterns;
    // scales of the shapelet length
    public int shapeletLengthScale;
    // number of classes
    public int numClasses;
    // number of segments
    public int numberOfSegments[];
    
    int L_min;
    // shapelets
    double Shapelets[][][];
    // classification weights
    double W[][][];
    double biasW[];

    // the softmax parameter
    public double alpha;
    
    public Instances trainSet;
    public Instance testSet;
    // time series data and the label 
    public double[][] train, classValues_train;
    public double[] test;

    // the number of iterations
    public int maxIter=1000;
    // the learning rate
    public double eta;

    // the regularization parameters
    public double lambdaW;

    public List<Double> nominalLabels;

    // structures for storing the precomputed terms
    double D_train[][][][]; //mean square error for each shapelet compared with each shapelet centroid. Formula 20
    double E_train[][][][]; // e^alpha*D_r,i,k,j part of Formula 23.
    double M_train[][][];   //Generalised Soft Minimum. Formula 19.
    double Psi_train[][][]; //Sum 1->j(e^alpha*D_r,i,k,j'). Denominator of Formula 23.
    double sigY_train[][];
    
    double D_test[][][];
    double E_test[][][];
    double M_test[][];
    double Psi_test[][];
    double sigY_test[];

    Random rand = new Random();

    List<Integer> instanceIdxs;
    List<Integer> rIdxs;
//Parameter search settings
    boolean paraSearch=true;
    double[] lambdaWRange = {0.01, 0.1};
    double[] percentageOfSeriesLengthRange = {0.1, 0.2};
    int[] shapeletLengthScaleRange = {2, 3};
    public void setParamSearch(boolean b){
        paraSearch=b;
    }
/* The actual parameter values should be set internally. This integer
  is just a key to maintain different parameter sets 
    */
    public void setPara(int x){
//Map integer: filthy hack,could be done better. Range is 1-8
        if(x<=4)
            lambdaW=lambdaWRange[0];
        else
            lambdaW=lambdaWRange[1];
        if(x==1 || x==2||x==5||x==6)    
            percentageOfSeriesLength=percentageOfSeriesLengthRange[0];
        else
            percentageOfSeriesLength=percentageOfSeriesLengthRange[1];
        if(x%2==1)
            shapeletLengthScale=shapeletLengthScaleRange[0];
        else
            shapeletLengthScale=shapeletLengthScaleRange[1];
    }
    public String getParas(){
        return lambdaW+","+percentageOfSeriesLength+","+shapeletLengthScale;
    }
    public double getAcc(){
        return maxAcc;
    }
    double maxAcc;
    
    // constructor
    public LearnShapeletsFeb2015Version() {
    }
    
    public void setSeed(long seed)
    {
        this.seed = seed;
        rand = new Random(seed);
    }

    // initialize the data structures
    public void initialize() throws Exception {
        
        // avoid K=0 
        if (numLatentPatterns == 0) {
            numLatentPatterns = 1;
        }
        
        L_min = (int)(percentageOfSeriesLength * seriesLength);

        // set the labels to be binary 0 and 1, needed for the logistic loss
        createOneVsAllTargets();

        // initialize the shapelets (complete initialization during the clustering)
        Shapelets = new double[shapeletLengthScale][][];
        // initialize the number of shapelets (by their starting point) and the length of the shapelets 
        numberOfSegments = new int[shapeletLengthScale];
        lengthsOfShapelet = new int[shapeletLengthScale];
        // set the lengths of shapelets and the number of segments
        // at each scale r
        int totalSegments = 0;
        
        //for each scale we create a number of segments and a shapelet length based on the scale value and our minimum shapelet length.
        for (int r = 0; r < shapeletLengthScale; r++) {
            lengthsOfShapelet[r] = (r + 1) * L_min;
            numberOfSegments[r] = seriesLength - lengthsOfShapelet[r];

            totalSegments += train.length * numberOfSegments[r];
        }

        // set the total number of shapelets per scale as a rule of thumb
        // to the logarithm of the total segments
        numLatentPatterns = (int) Math.log(totalSegments);

        // initialize an array of the sizes
        rIdxs = new ArrayList<>();
        for (int r = 0; r < shapeletLengthScale; r++) {
            rIdxs.add(r);
        }

        // initialize shapelets
        initializeShapeletsKMeans();


        // initialize the terms for pre-computation
        D_train = new double[train.length][shapeletLengthScale][numLatentPatterns][];
        E_train = new double[train.length][shapeletLengthScale][numLatentPatterns][];

        for (int i = 0; i < train.length; i++) {
            for (int r = 0; r < shapeletLengthScale; r++) {
                for (int k = 0; k < numLatentPatterns; k++) {
                    D_train[i][r][k] = new double[numberOfSegments[r]];
                    E_train[i][r][k] = new double[numberOfSegments[r]];
                }
            }
        }
        
        // initialize the placeholders for the precomputed values
        M_train = new double[train.length][shapeletLengthScale][numLatentPatterns];
        Psi_train = new double[train.length][shapeletLengthScale][numLatentPatterns];
        sigY_train = new double[train.length][numClasses];

        // initialize the weights
        W = new double[numClasses][shapeletLengthScale][numLatentPatterns];
        biasW = new double[numClasses];
        
        for (int c = 0; c < numClasses; c++) {
            for (int r = 0; r < shapeletLengthScale; r++) {
                for (int k = 0; k < numLatentPatterns; k++) {
                    W[c][r][k] = 2 * rand.nextDouble() - 1;
                }
            }

            biasW[c] = 2 * rand.nextDouble() - 1;
        }

        // precompute the M, Psi, sigY, used later for setting initial W
        for (int i = 0; i < train.length; i++) {
            preCompute(D_train[i], E_train[i], Psi_train[i], M_train[i], sigY_train[i], train[i]);
        }

        // initialize W by learning the model on the centroid data
        learnFOnlyW();

        // store all the instances indexes for
        instanceIdxs = new ArrayList<>();
        for (int i = 0; i < train.length; i++) {
            instanceIdxs.add(i);
        }
        // shuffle the order for a better convergence
        Collections.shuffle(instanceIdxs, rand);
    }

    // create one-cs-all targets
    public void createOneVsAllTargets() {
        
        numClasses = nominalLabels.size();

        classValues_train = new double[train.length][numClasses];

        // initialize the extended representation  
        for (int i = 0; i < train.length; i++) {
            // firts set everything to zero
            for (int c = 0; c < numClasses; c++) {
                classValues_train[i][c] =  0;
            }

            // then set the real label index to 1
            int indexLabel = nominalLabels.indexOf(trainSet.get(i).classValue());
            classValues_train[i][indexLabel] = 1.0;
        }
    }

    // initialize the shapelets from the centroids of the segments
    public void initializeShapeletsKMeans() throws Exception {
        //for each scale r, i.e. for each set of K shapelets at
        // length L_min*(r+1)
        for (Integer r : rIdxs) {
            double[][] segmentsR = new double[train.length * numberOfSegments[r]][lengthsOfShapelet[r]];
            
            //construct the segments from the train set.
            for (int i = 0; i < train.length; i++) {
                for (int j = 0; j < numberOfSegments[r]; j++) {
                    for (int l = 0; l < lengthsOfShapelet[r]; l++) {
                        segmentsR[i * numberOfSegments[r] + j][l] = train[i][j + l];
                    }
                }
            }

            // normalize segments
            for (int i = 0; i < train.length; i++) {
                for (int j = 0; j < numberOfSegments[r]; j++) {
                    segmentsR[i * numberOfSegments[r] + j] = StatisticalUtilities.normalize(segmentsR[i * numberOfSegments[r] + j]);
                }
            }

            Instances ins = InstanceTools.toWekaInstances(segmentsR); 
            
            SimpleKMeans skm = new SimpleKMeans();
            skm.setNumClusters(numLatentPatterns);
            skm.setMaxIterations(100);
            skm.setSeed((int) (rand.nextDouble() * 1000)); 
            //skm.setInitializeUsingKMeansPlusPlusMethod(true); 
            skm.buildClusterer( ins );
            Instances centroidsWeka = skm.getClusterCentroids();
            Shapelets[r] =  InstanceTools.fromWekaInstancesArray(centroidsWeka);
              
            if (Shapelets[r] == null) {
                System.out.println("P not set");
            }
        }
    }

    // predict the label value vartheta_i
    public double predict_i(double[][] M, int c) {
        double Y_hat_ic = biasW[c];

        for (int r = 0; r < shapeletLengthScale; r++) {
            for (int k = 0; k < numLatentPatterns; k++) {
                Y_hat_ic += M[r][k] * W[c][r][k];
            }
        }

        return Y_hat_ic;
    }

    // precompute terms
    public void preCompute(double[][][] D, double[][][] E, double[][] Psi, double[][] M, double[] sigY, double[] series) {
        
        // precompute terms     
        for (int r = 0; r < shapeletLengthScale; r++) {
            //in most cases Shapelets[r].length == numLatentPatterns, this is not always true.
            for (int k = 0; k < Shapelets[r].length; k++) { 
                for(int j = 0; j < numberOfSegments[r]; j++)
                {
                    // precompute D
                    D[r][k][j] = 0;
                    double err = 0;

                    for(int l = 0; l < lengthsOfShapelet[r]; l++)
                    {
                        err = series[j + l] - Shapelets[r][k][l];
                        D[r][k][j] += err*err; 
                    }

                    D[r][k][j] /= (double)lengthsOfShapelet[r]; 

                    // precompute E
                    E[r][k][j] = Math.exp(alpha * D[r][k][j]);
                }

                // precompute Psi 
                Psi[r][k] = 0; 
                for(int j = 0; j < numberOfSegments[r]; j++) 
                        Psi[r][k] +=  Math.exp( alpha * D[r][k][j] );

                // precompute M 
                M[r][k] = 0;

                for(int j = 0; j < numberOfSegments[r]; j++)
                        M[r][k] += D[r][k][j]* E[r][k][j];

                M[r][k] /= Psi[r][k];
            }
        }

        for (int c = 0; c < numClasses; c++) {
            sigY[c] = calculateSigmoid(predict_i(M, c));
        }
    }

    // compute the MCR on the test set
    public double trainSetErrorRate() throws Exception {
        int numErrors = 0;

        for (int i=0; i < trainSet.numInstances(); i++) {
            preCompute(D_train[i], E_train[i], Psi_train[i], M_train[i], sigY_train[i], train[i]);

            Instance inst = trainSet.get(i);
            
            double max_Y_hat_ic = Double.MIN_VALUE;
            int label_i = -1;

            for (int c = 0; c < numClasses; c++) {
                double Y_hat_ic = calculateSigmoid(predict_i(M_train[i], c));

                if (Y_hat_ic > max_Y_hat_ic) {
                    max_Y_hat_ic = Y_hat_ic;
                    label_i = c;
                }
            }

            //we've predicted a value which doesn't exist.
            if (inst.classValue() != label_i) {
                numErrors++; 
            }  
       }

        return (double) numErrors / (double) trainSet.numInstances();
    }

    // compute the accuracy loss of instance i according to the 
    // smooth hinge loss 
    public double accuracyLoss(double[][] M, double[] classValues, int c) {
        double Y_hat_ic = predict_i(M, c);
        double sig_y_ic = calculateSigmoid(Y_hat_ic);

        //L(Y,Y_hat) = -Y ln(sig_y_hat) - (1-Y)ln(1-sig_y_hat) - Formula 3 in the paper.
        return -classValues[c] * Math.log(sig_y_ic) - (1 - classValues[c]) * Math.log(1 - sig_y_ic);
    }

    // compute the accuracy loss of the train set
    public double accuracyLossTrainSet() {
        double accuracyLoss = 0;

        for (int i = 0; i < train.length; i++) {
            preCompute(D_train[i], E_train[i], Psi_train[i], M_train[i], sigY_train[i], train[i]);

            for (int c = 0; c < numClasses; c++) {
                accuracyLoss += accuracyLoss(M_train[i], classValues_train[i], c);
            }
        }

        return accuracyLoss;
    }

    public void learnF() {       
        //instanceIdxs is a random reordering of the trin set.
        for (Integer i : instanceIdxs) {
            double regWConst = ((double) 2.0 * lambdaW) / ((double) train.length);

            double tmp2, tmp1, dLdY, dMdS;

            preCompute(D_train[i], E_train[i], Psi_train[i], M_train[i], sigY_train[i], train[i]);

            for (int c = 0; c < numClasses; c++) {
                //difference beteen our classes predicted and our actual values.
                dLdY = -(classValues_train[i][c] - sigY_train[i][c]);

                for (int r = 0; r < shapeletLengthScale; r++) {
                    //in most cases Shapelets[r].length == numLatentPatterns, this is not always true.
                    for (int k = 0; k < Shapelets[r].length; k++) {
                        W[c][r][k] -= eta * (dLdY * M_train[i][r][k] + regWConst * W[c][r][k]);

                        tmp1 = (2.0 / ((double) lengthsOfShapelet[r] * Psi_train[i][r][k]));

                        for (int l = 0; l < lengthsOfShapelet[r]; l++) {
                            tmp2 = 0;
                            for (int j = 0; j < numberOfSegments[r]; j++) {
                                tmp2 += E_train[i][r][k][j] * (1 + alpha * (D_train[i][r][k][j] - M_train[i][r][k])) * (Shapelets[r][k][l] - train[i][j + l]);
                            }

                            dMdS = tmp1 * tmp2;

                            Shapelets[r][k][l] -= eta * (dLdY * W[c][r][k] * dMdS);

                        }
                        
                    }
                }

                biasW[c] -= eta * dLdY;
            }
        }
    }

    public void learnFOnlyW() {
        double regWConst = ((double) 2.0 * lambdaW) / ((double) train.length);

        for (int epochs = 0; epochs < maxIter; epochs++) {
            for (int i = 0; i < train.length; i++) {
                for (int c = 0; c < numClasses; c++) {
                    sigY_train[i][c] = calculateSigmoid(predict_i(M_train[i], c));

                    for (int r = 0; r < shapeletLengthScale; r++) {
                        for (int k = 0; k < numLatentPatterns; k++) {
                            W[c][r][k] -= eta * (-(classValues_train[i][c] - sigY_train[i][c]) * M_train[i][r][k] + regWConst * W[c][r][k]);
                        }
                    }

                    biasW[c] -= eta * (-(classValues_train[i][c] - sigY_train[i][c]));
                }
            }
        }

    }

    // optimize the objective function
    @Override
    public void buildClassifier(Instances data) throws Exception {
        
        double[] paramsLambdaW;
        double[] paramsPercentageOfSeriesLength;
        int[] paramsShapeletLengthScale;
        Instances sample;
        if(paraSearch){ 
            paramsLambdaW=lambdaWRange;
            paramsPercentageOfSeriesLength=percentageOfSeriesLengthRange;
            paramsShapeletLengthScale=shapeletLengthScaleRange;
            sample=data;
        }else{// Hack to minimize changes to the method below
            paramsLambdaW=new double[1];
            paramsLambdaW[0]=lambdaW;
            paramsPercentageOfSeriesLength=new double[1];
            paramsPercentageOfSeriesLength[0]=percentageOfSeriesLength;
            paramsShapeletLengthScale=new int[1];
            paramsShapeletLengthScale[0]=shapeletLengthScale;
            sample=data;
        }
        

        int noFolds = 3;
        double bsfAccuracy = 0;
        int[] params = {0,0,0};
        double accuracy = 0;
        for (int i = 0; i < paramsLambdaW.length; i++) {
            for (int j = 0; j < paramsPercentageOfSeriesLength.length; j++) {
                for (int k = 0; k < paramsShapeletLengthScale.length; k++) {
                    double sumAccuracy = 0;
                    //build our test and train sets. for cross-validation.
                    System.out.println("Begin cross validation");
                    for (int l = 0; l < noFolds; l++) {
                        Instances trainCV = sample.trainCV(noFolds, l);
                        Instances testCV = sample.testCV(noFolds, l);

                        percentageOfSeriesLength = paramsPercentageOfSeriesLength[j];
                        shapeletLengthScale = paramsShapeletLengthScale[k];
                        lambdaW = paramsLambdaW[i];
                        eta = 0.1;
                        alpha = -30;
                        train(trainCV);
                        
                        //test on the remaining fold.
                        int correct=0;
                        for(Instance in : testCV){
                            if(classifyInstance(in) == in.classValue())
                                correct++;
                        }
                        
                        accuracy = (double)correct/(double)testCV.numInstances();
                        sumAccuracy += accuracy;
                        trainCV=null;
                        testCV=null;
                    }
                    sumAccuracy/=noFolds;
                    
                    //System.out.printf("%f,%d,%f,%f\n", paramsPercentageOfSeriesLength[j], paramsShapeletLengthScale[k], paramsLambdaW[i], sumAccuracy);
                    
                    if(sumAccuracy > bsfAccuracy){
                        int[] p = {i,j,k};
                        params = p;
                        bsfAccuracy = sumAccuracy;
                    }
                }
            }
        }
        System.out.println("End cross validation paras "+params[0]+", "+params[1]+", "+params[2]);
        sample=null;
        
        System.gc();
        maxAcc=bsfAccuracy;
        lambdaW = paramsLambdaW[params[0]];
        percentageOfSeriesLength = paramsPercentageOfSeriesLength[params[1]];
        shapeletLengthScale = paramsShapeletLengthScale[params[2]];
        if(paraSearch)  //Rebuild whole data set on optimal parameters 
            train(data); 
    }
    
    private void train(Instances data) throws Exception
    {

        trainSet = data;
        seriesLength = trainSet.numAttributes() - 1; //so we don't include the classLabel at the end.

        nominalLabels = readNominalTargets(trainSet);
        
        //convert the training set into a 2D Matrix.
        train = fromWekaInstancesArray(trainSet);

        // initialize the data structures
        initialize();

        List<Double> lossHistory = new ArrayList<>();
        lossHistory.add(Double.MIN_VALUE);
        int restarts=0;
        // apply the stochastic gradient descent in a series of iterations
        for (int iter = 0; iter <= maxIter; iter++) {
            // learn the latent matrices
            learnF();
//            if(iter%10==0)
//            System.out.println("Max it ="+maxIter+" iter ="+iter);

            // measure the loss
            if ((iter %(maxIter/2)) == 0 && iter>0) {
                double mcrTrain = trainSetErrorRate();
                double lossTrain = accuracyLossTrainSet();

                lossHistory.add(lossTrain);
                // if divergence is detected start from the beginning 
                // at a lower learning rate, 
                if (Double.isNaN(lossTrain) || mcrTrain == 1.0) {
                    restarts++;
  //                  System.out.println("Resetting "+restarts);    
                    eta /= 3;
                    lossHistory.clear();
                    initialize();
                    if(restarts<5)
                        iter = 0;
                }

                if (lossHistory.size() > maxIter/2) {
                    if (lossTrain > lossHistory.get(lossHistory.size() - 2)) {
                        break;
                    }
                }
            }
//            System.out.println("END: Max it ="+maxIter+" iter ="+iter);            
        }
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
            
        testSet = instance;
        
        test = testSet.toDoubleArray();
        
//        test = normalize(test, true);
        
        // initialize the terms for pre-computation
        D_test = new double[shapeletLengthScale][numLatentPatterns][];
        E_test = new double[shapeletLengthScale][numLatentPatterns][];

        for (int r = 0; r < shapeletLengthScale; r++) {
            for (int k = 0; k < numLatentPatterns; k++) {
                D_test[r][k] = new double[numberOfSegments[r]];
                E_test[r][k] = new double[numberOfSegments[r]];
            }
        }

        // initialize the placeholders for the precomputed values
        M_test = new double[shapeletLengthScale][numLatentPatterns];
        Psi_test = new double[shapeletLengthScale][numLatentPatterns];
        sigY_test = new double[numClasses];
        
        preCompute(D_test, E_test, Psi_test, M_test, sigY_test, test);


        double max_Y_hat_ic = Double.MIN_VALUE;
        int label_i = -1;

        for (int c = 0; c < numClasses; c++) {
            double Y_hat_ic = calculateSigmoid(predict_i(M_test, c));

            if (Y_hat_ic > max_Y_hat_ic) {
                max_Y_hat_ic = Y_hat_ic;
                label_i = c;
            }
        }

        return label_i;
    }

    
    public void printShapeletsAndWeights() {
        for (int r = 0; r < shapeletLengthScale; r++) {
            for (int k = 0; k < numLatentPatterns; k++) {
                System.out.print("Shapelets(" + r + "," + k + ")= [ ");

                for (int l = 0; l < lengthsOfShapelet[r]; l++) {
                    System.out.print(Shapelets[r][k][l] + " ");
                }

                System.out.println("]");
            }
        }

        for (int c = 0; c < numClasses; c++) {
            for (int r = 0; r < shapeletLengthScale; r++) {
                System.out.print("W(" + c + "," + r + ")= [ ");

                for (int k = 0; k < numLatentPatterns; k++) {
                    System.out.print(W[c][r][k] + " ");
                }

                System.out.print(biasW[c] + " ");

                System.out.println("]");
            }
        }
    }

    public void printProjectedData() {
        int r = 0, c = 0;

        System.out.print("Data= [ ");

        for (int i = 0; i < train.length; i++) {
            preCompute(D_train[i], E_train[i], Psi_train[i], M_train[i], sigY_train[i], train[i]);
            
            System.out.print(classValues_train[i][c] + " ");

            for (int k = 0; k < numLatentPatterns; k++) {
                System.out.print(M_train[i][r][k] + " ");
            }

            System.out.println(";");
        }

        System.out.println("];");
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
    public static ArrayList<Double> readNominalTargets(Instances instances)
    {
        if (instances.size() <= 0)  return null;
        
        ArrayList<Double> nominalLabels = new ArrayList<>();
        
        for (Instance ins : instances) {
            boolean alreadyAdded = false;

            for (Double nominalLabel : nominalLabels) {
                if (nominalLabel == ins.classValue()) {
                    alreadyAdded = true;
                    break;
                }
            }

            if (!alreadyAdded) {
                nominalLabels.add(ins.classValue());
            }
        }

        Collections.sort(nominalLabels);

        return nominalLabels;
    }
    
    
    public static void main(String[] args) throws Exception{
        
        //resample 1 of the italypowerdemand dataset
        String dataset = "Beef";
        String fileExtension = File.separator + dataset + File.separator + dataset;
        String samplePath = "C:\\LocalData\\time-series-datasets\\TSC Problems (1)\\" + fileExtension;

        //load the train and test.
        Instances testSet = utilities.ClassifierTools.loadData(DataSets.problemPath + dataset + "\\" + dataset + "_TRAIN");
        Instances trainSet = utilities.ClassifierTools.loadData(DataSets.problemPath + dataset + "\\" + dataset + "_TEST");        

        
        
        LearnShapeletsFeb2015Version ls = new LearnShapeletsFeb2015Version();
        ls.setSeed(0);
        ls.buildClassifier(trainSet);
        double accuracy = utilities.ClassifierTools.accuracy(testSet, ls);
        System.out.println("LS: " + (1 - accuracy));
        
    }
}
