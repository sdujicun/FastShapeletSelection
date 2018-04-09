
/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package applications;

import java.util.ArrayList;
import utilities.ClassifierTools;
import weka.classifiers.Classifier;
import weka.core.Instances;
import fileIO.*;
import java.io.File;
import java.text.DecimalFormat;
import java.util.Random;
import papers.ICDM2013_Lines;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.lazy.DTW_1NN;
import weka.classifiers.lazy.IBk;
import weka.classifiers.lazy.kNN;
import weka.classifiers.meta.RotationForest;
import weka.classifiers.meta.timeseriesensembles.TransformEnsembles;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.filters.*;
import weka.filters.timeseries.*;

public class Otoliths {
    static String dataPath="C:\\Users\\ajb\\Dropbox\\TSC Problems\\Herring\\HERRING500";
    Instances data;
    Otoliths(){
        data=ClassifierTools.loadData(dataPath);
    }
        public static void CSSClassifier(int folds){
        ArrayList<String> names=new ArrayList<>();
        Instances all=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\Otoliths\\CSSHerringA");
        all.randomize(new Random());
        NormalizeCase nc=new NormalizeCase();
        nc.setNormType(NormalizeCase.NormType.STD);
        Classifier[] c =setSingleClassifiers(names);
        OutFile res=new OutFile("C:\\Research\\Results\\Otoliths\\singleClassifiersCSSA"
                + ".csv");
        try{
            all=nc.process(all);
            for(int i=0;i<c.length;i++){
                System.out.print(" running classifier "+names.get(i));
                Evaluation e=new Evaluation(all);
                e.crossValidateModel(c[i], all, folds, new Random());
                res.writeLine(names.get(i)+","+e.correct()/all.numInstances());
                System.out.println(" Acc = "+e.correct()/all.numInstances());
            }
        }catch(Exception ex){
                ex.printStackTrace();
                System.exit(0);
        }
    }
	
	public static Classifier[] setSingleClassifiers(ArrayList<String> names){
		ArrayList<Classifier> sc2=new ArrayList<Classifier>();
		Classifier c;
                kNN k=new kNN(100);
                k.setCrossValidate(true);
                sc2.add(k);
		names.add("kNN_ED");
                sc2.add(new NaiveBayes());
		names.add("NB");
		sc2.add(new J48());
		names.add("C45");
		c=new SMO();
		PolyKernel kernel = new PolyKernel();
		kernel.setExponent(1);
		((SMO)c).setKernel(kernel);
		sc2.add(c);
		names.add("SVML");
		c=new SMO();
		kernel = new PolyKernel();
		kernel.setExponent(2);
		((SMO)c).setKernel(kernel);
		sc2.add(c);
		names.add("SVMQ");
		c=new SMO();
		c=new RandomForest();
		((RandomForest)c).setNumTrees(200);
		sc2.add(c);
		names.add("RandF200");
		c=new RotationForest();
		sc2.add(c);
                ((RotationForest) c).setNumIterations(50);
		names.add("RotF30");
                c=new DTW_1NN();
		sc2.add(c);
		names.add("NN_DTW");
	
		Classifier[] sc=new Classifier[sc2.size()];
		for(int i=0;i<sc.length;i++)
			sc[i]=sc2.get(i);

		return sc;
	}
    public void rawOutlinesSingleClassifier(boolean normalise){
        ArrayList<String> names=new ArrayList<String>();
        data.randomize(new Random());
        if(normalise){
            NormalizeCase nc=new NormalizeCase();
            nc.setNormType(NormalizeCase.NormType.STD);
        try{
            data=nc.process(data);
        }catch(Exception ex){
                System.out.println(" Error normalising ");
                ex.printStackTrace();
                System.exit(0);
        }
        }
        Classifier[] c =setSingleClassifiers(names);
        OutFile res;
        if(normalise)
             res=new OutFile("C:\\Users\\ajb\\Dropbox\\Results\\Otoliths\\singleClassifiers.csv");
          else
             res=new OutFile("C:\\Users\\ajb\\Dropbox\\Results\\Otoliths\\singleClassifiersNormalised.csv");
        try{
            
            for(int i=0;i<c.length;i++){
                System.out.print(" running classifier "+names.get(i));
                Evaluation e=new Evaluation(data);
                e.crossValidateModel(c[i], data, data.numInstances(), new Random());
                res.writeLine(names.get(i)+","+e.correct()/data.numInstances());
                System.out.println(" Acc = "+e.correct()/data.numInstances());
            }
        }catch(Exception ex){
                ex.printStackTrace();
                System.exit(0);
        }
    }
    
    public void powerSpectrumSingleClassifier(boolean normalise){
        ArrayList<String> names=new ArrayList<String>();
        data.randomize(new Random());
        if(normalise){
            NormalizeCase nc=new NormalizeCase();
            nc.setNormType(NormalizeCase.NormType.STD);
        try{
            data=nc.process(data);
        }catch(Exception ex){
                System.out.println(" Error normalising ");
                ex.printStackTrace();
                System.exit(0);
        }
        }
        Classifier[] c =setSingleClassifiers(names);
        OutFile res;
        if(normalise)
             res=new OutFile("C:\\Users\\ajb\\Dropbox\\Results\\Otoliths\\singleClassifiersPS.csv");
          else
             res=new OutFile("C:\\Users\\ajb\\Dropbox\\Results\\Otoliths\\singleClassifiersNormalisedPS.csv");
        try{
            PowerSpectrum ps= new PowerSpectrum();
            data=ps.process(data);
            for(int i=0;i<c.length;i++){
                System.out.print(" running classifier "+names.get(i));
                Evaluation e=new Evaluation(data);
                e.crossValidateModel(c[i], data, data.numInstances(), new Random());
                res.writeLine(names.get(i)+","+e.correct()/data.numInstances());
                System.out.println(" Acc = "+e.correct()/data.numInstances());
            }
        }catch(Exception ex){
                ex.printStackTrace();
                System.exit(0);
        }
    }
    
    public static double icdmEnsemble(Instances train, Instances test){
        
        return 0;
    }
    
    
    
	public static void basicDataTransforms(String baseClassifier, int nosFolds){
		DecimalFormat dc= new DecimalFormat("###.###");
        Instances all=ClassifierTools.loadData("C:\\Research\\Data\\Time Series Classification\\Otoliths\\Herring");
		OutFile of=new OutFile("C:\\Research\\Results\\Otoliths\\baseTransforms"+"baseClassifier.csv");
		System.out.println("NEAREST NEIGHBOUR CLASSIFIERS");
		of.writeLine("10 fold cross validation results with"+baseClassifier);
		of.writeLine(",TimeDomain,PowerSpectrumDomain,ACFDomain,PCADomain");
		Classifier base=null;
		if(baseClassifier.equals("1NN"))
			base=new kNN(1);
		else if(baseClassifier.equals("DTW"))
			base=new DTW_1NN();
		else if(baseClassifier.equals("RotationForest"))
			base=new RotationForest();
		else if(baseClassifier.equals("RandomForest")){
			base=new RandomForest();
			((RandomForest)base).setNumTrees(30);
		}
		else if(baseClassifier.equals("SVMQ")){
                    base=new SMO();
                    PolyKernel kernel = new PolyKernel();
                    kernel.setExponent(2);
                    ((SMO)base).setKernel(kernel);
                }
		else{
			System.out.println("Classifier Not Included, exiting");
			System.exit(0);
		}
		try{
                    System.out.println("******************Time Domain******************");
                    Evaluation e=new Evaluation(all);
                    e.crossValidateModel(AbstractClassifier.makeCopy(base), all, nosFolds, new Random());
                    System.out.println(" Acc = "+e.correct()/all.numInstances());
 //                res.writeLine(names.get(i)+","+e.correct()/all.numInstances());
                    of.writeString(e.correct()/all.numInstances()+",");
                    System.out.println("******************Power Spectrum Domain******************");
                    PowerSpectrum ps=new PowerSpectrum();
                    Instances psAll=ps.process(all);
/* Delete the duplicate half of the spectrum */
                    int atts=(psAll.numAttributes()-1)/2-2;
                    for(int j=0;j<atts;j++)
			psAll.deleteAttributeAt(psAll.numAttributes()-2);
                     e=new Evaluation(psAll);
                    e.crossValidateModel(AbstractClassifier.makeCopy(base), psAll, nosFolds, new Random());
                    System.out.println(" Acc = "+e.correct()/all.numInstances());
                   of.writeString(e.correct()/all.numInstances()+",");
                    System.out.println("\n******************ACF Domain******************");
                    ACF acf=new ACF(); 
                    acf.setMaxLag(atts);
                    Instances acfAll=acf.process(all);
                     e=new Evaluation(acfAll);
                    e.crossValidateModel(AbstractClassifier.makeCopy(base), acfAll, nosFolds, new Random());
                    System.out.println(" Acc = "+e.correct()/all.numInstances());
                   of.writeString(e.correct()/all.numInstances()+",");
                    
                    
		}catch(Exception e){System.out.println("Exception ="+e);e.printStackTrace();System.exit(0);}
	}	
		
     public static double combineResults(String results){
         InFile f;
         OutFile of=new OutFile(results);
         double acc=0;
         for(int i=1;i<=100;i++){
             f=new InFile("C:/Users/ajb/Dropbox/Results/Herring/fold"+i);
             String s=f.readLine();
             of.writeLine(s);
         }
         return 0;
         
         
     }   
        public static String dataName="Herring500";
     
    public static void datasetCrossValidation(Instances train_raw) throws Exception{
        ICDM2013_Lines.initCv(dataName);
        
        // get derivative training data (can store locally and read in to save computation time for large datasets)
        DerivativeFilter df = new DerivativeFilter();
        Instances train_derivative = df.process(train_raw);
        
        double   cv_01_euclidean_1nn = ICDM2013_Lines.cv_01_Euclidean_1NN(dataName,train_raw);                                 // 01 Euclidean 1NN
        double   cv_02_dtw_fullWindow_1nn = ICDM2013_Lines.cv_02_DTW_fullWindow_1NN(dataName,train_raw);                       // 02 DTW Full Window 1NN
        double[] cv_03_dtw_cvWindow_1nn = ICDM2013_Lines.cv_03_DTW_bestWindow_1NN(dataName,train_raw);                         // 03 DTW variable window 1NN (try all possible values of R from 0% to 100% in increments of 1%)
        double[] cv_04_wdtw_1nn = ICDM2013_Lines.cv_04_WDTW_1NN(dataName,train_raw);                                           // 04 Weighted DTW with cv to find the optimal weight, g. Possible values for g range from 0 to 1 in increments of 0.01
        double[] cv_05_euclidean_knn = ICDM2013_Lines.cv_05_Euclidean_kNN(dataName,train_raw);                                 // 05 Euclidean kNN (k = 1, 2, ..., 100)
        double[] cv_06_dtw_fullWindow_knn = ICDM2013_Lines.cv_06_DTW_fullWindow_kNN(dataName,train_raw);                       // 06 DTW Full Window kNN (k = 1, 2, ..., 100)
        
        double[] cv_11_dtw_optimalWindow_knn = ICDM2013_Lines.cv_11_DTW_optimalWindow_kNN(dataName,train_raw);                 // 11 DTW Optimal Window kNN (r = 0, 0.01, 0.02, ..., 1) (k = 1, 2, ..., 100)
        double[] cv_12_wdtw_knn = ICDM2013_Lines.cv_12_WDTW_kNN(dataName,train_raw);                                           // 12 WDTW kNN (g = 0, 0.01, 0.02, ..., 1) (k = 1, 2, ..., 100)
        
        double   cv_21_ddtw_fullWindow_1nn = ICDM2013_Lines.cv_21_DDTW_fullWindow_1NN(dataName,train_derivative);              // 21 Derivative DTW Full Window 1NN
        double[] cv_22_ddtw_cvWindow_1nn = ICDM2013_Lines.cv_22_DDTW_bestWindow_1NN(dataName,train_derivative);                // 22 Derivative DTW Variable Window 1NN (r 0-100%, increaments of 1%)
        double[] cv_23_wdtw_1nn = ICDM2013_Lines.cv_23_WDDTW_1NN(dataName,train_derivative);                                   // 23 Erighted Derivative DTW 1NN (g 0-1, increments of 0.01)
        double[] cv_24_ddtw_fullWindow_knn = ICDM2013_Lines.cv_24_DDTW_fullWindow_kNN(dataName,train_derivative);              // 24 Derivative DTW Full Window kNN (k 1-100, increments of 1)
        double[] cv_25_ddtw_optimalWindow_knn = ICDM2013_Lines.cv_25_DDTW_optimalWindow_kNN(dataName,train_derivative);        // 25 Derivative DTW Variable Window
        double[] cv_26_wddtw_knn = ICDM2013_Lines.cv_26_WDTW_kNN(dataName,train_derivative);
        
        double[] cv_31_lcss_1nn = ICDM2013_Lines.cv_31_LCSS_1NN(dataName,train_raw);
        double[] cv_32_lcss_knn = ICDM2013_Lines.cv_32_LCSS_kNN(dataName,train_raw,(int)cv_31_lcss_1nn[1],cv_31_lcss_1nn[2]);
        
        
        
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
    public static void datasetTrainTest(Instances train_raw, Instances test_raw) throws Exception{
        // Pre-requisite of train/test classification is that necessary params have been found in CV stage.
        // Therefore, check to see if CV has been carried out previously. If not, begin then CV automatically.
        File cvDir = new File(ICDM2013_Lines.OUTPUT_DIR_CV);
        if(!cvDir.exists()){
            System.out.println("Cross-validation for "+dataName+" doesn't appear to have taken place. Starting cross-validation.");
            datasetCrossValidation(train_raw);
        }
        
        ICDM2013_Lines.initTrainTest(dataName);
        
        
        DerivativeFilter df = new DerivativeFilter();
        Instances train_derivative = df.process(train_raw);
        df = new DerivativeFilter();
        Instances test_derivative = df.process(test_raw);
        
        /***** RAW DATA ****/
        System.out.printf("Euclidean 1NN:%33.3f%n",ICDM2013_Lines.trainTest_01_Euclidean_1NN(dataName, train_raw, test_raw));
        System.out.printf("DTW Full Window 1NN:%27.3f%n",ICDM2013_Lines.trainTest_02_DTW_fullWindow_1NN(dataName, train_raw, test_raw));
        System.out.printf("DTW Best Window 1NN:%27.3f%n",ICDM2013_Lines.trainTest_03_DTW_optimalWindow_1NN(dataName, train_raw, test_raw));
        System.out.printf("WDTW 1NN:%38.3f%n",ICDM2013_Lines.trainTest_04_WDTW_1NN(dataName, train_raw, test_raw));
        System.out.printf("Euclidean kNN:%33.3f%n",ICDM2013_Lines.trainTest_05_Euclidean_kNN(dataName, train_raw, test_raw));
        System.out.printf("DTW Full Window kNN:%27.3f%n",ICDM2013_Lines.trainTest_06_DTW_fullWindow_kNN(dataName, train_raw, test_raw));
        System.out.printf("DTW Best Window kNN:%27.3f%n",ICDM2013_Lines.trainTest_11_DTW_bestWindow_kNN(dataName, train_raw, test_raw));
        System.out.printf("WDTW kNN:%38.3f%n",ICDM2013_Lines.trainTest_12_WDTW_kNN(dataName, train_raw, test_raw));
        
        /***** DERIVATIVE TRANSFORMED DATA ****/
        System.out.printf("DDTW Full Window 1NN:%26.3f%n",ICDM2013_Lines.trainTest_21_DDTW_fullWindow_1NN(dataName, train_derivative, test_derivative));
        System.out.printf("DDTW Best Window 1NN:%26.3f%n",ICDM2013_Lines.trainTest_22_DDTW_optimalWindow_1NN(dataName, train_derivative, test_derivative));
        System.out.printf("WDDTW 1NN:%37.3f%n",ICDM2013_Lines.trainTest_23_WDDTW_1NN(dataName, train_derivative, test_derivative));
        System.out.printf("DDTW Full Window kNN:%26.3f%n",ICDM2013_Lines.trainTest_24_DDTW_fullWindow_kNN(dataName, train_derivative, test_derivative));
        System.out.printf("DDTW Best Window kNN:%26.3f%n",ICDM2013_Lines.trainTest_25_DDTW_bestWindow_kNN(dataName, train_derivative, test_derivative));
        System.out.printf("WDDTW kNN:%37.3f%n",ICDM2013_Lines.trainTest_26_WDDTW_kNN(dataName, train_derivative, test_derivative));
        
        /***** LCSS ******/
        System.out.printf("LCSS 1NN:%38.3f%n",ICDM2013_Lines.trainTest_31_LCSS_1NN(dataName, train_raw, test_raw));
        System.out.printf("LCSS kNN:%38.3f%n",ICDM2013_Lines.trainTest_32_LCSS_kNN(dataName, train_raw, test_raw));
        
    }   
//Nasty hack to creat 100 test train data sets!     
    public static void splitAllData(){
        String path="C:/Users/ajb/Dropbox/";
        Instances all=ClassifierTools.loadData(path+"TSC_Problems/Herring500/Herring500");
        for(int fold=0;fold<all.numInstances();fold++){
            Instances train=new  Instances(all);
            train.delete(fold);
            Instances test=new  Instances(all,0);
            test.add(all.instance(fold));
            //Create fold directory
            File f = new File(path+"TSC_Problems/Herring500_"+(fold+1));
            if(!f.isDirectory())//Test whether directory exists
                f.mkdir();

            //Save train/test
            OutFile of=new OutFile(path+"TSC_Problems/Herring500_"+(fold+1)+"/Herring500_"+(fold+1)+"_TRAIN.arff");
            of.writeLine("% Train split "+fold+"\n"+train.toString());
            OutFile of2=new OutFile(path+"TSC_Problems/Herring500_"+(fold+1)+"/Herring500_"+(fold+1)+"_TEST.arff");
            of2.writeLine("% Test split "+fold+"\n"+test.toString());
            
        }
    }
    
    
    public static void main(String[] args){
        int fold=Integer.parseInt(args[0]);
        
        String dataName = "Herring500_"+fold;
        System.out.println(" Running fold ="+fold);
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
            File cvResultsDir = new File(ICDM2013_Lines.OUTPUT_DIR_CV+"/"+dataName);
            if(cvResultsDir.exists()){
                ICDM2013_Lines.printPreCalculatedCvResults(dataName);
            }else{
                ICDM2013_Lines.datasetCrossValidation(dataName);
            }
            System.out.println();
            
            // Part 2: Train/Test
            // If train/test hasn't been carried out, perform Train/Test (ASSUMPTION: if results dir exists, Train/Test has been fully carried out)
            // Else, print Train/Test results
            System.out.println("Train/Test Results");
            System.out.println("----------------------------------------");
            File trainTestResultsDir = new File(ICDM2013_Lines.OUTPUT_DIR_TRAIN_TEST+"/"+dataName);
            if(trainTestResultsDir.exists()){
                ICDM2013_Lines.printPrecalculatedTrainTestResults(dataName);
            }else{
                ICDM2013_Lines.datasetTrainTest(dataName);
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
            ICDM2013_Lines.print10RunEnsembles(dataName);

        }catch(Exception e){
            e.printStackTrace();
        }        
/*        Instances all=ClassifierTools.loadData("TSC_Problems/Herring/Herring500");
        Instances train=new  Instances(all);
        Instances test=new  Instances(all,0);
        test.add(all.instance(fold));
        train.delete(fold);
 //       double acc=icdmEnsemble(train,test);
        double pred=icdmEnsemble(train,test);
        
        OutFile of=new OutFile("Results/Herring/fold"+(fold+1));
        of.writeLine(fold+","+pred+","+test.instance(0).classValue());
 */       
       System.exit(0);
                //                CSSClassifier(100);
       Otoliths o= new Otoliths();
       o.rawOutlinesSingleClassifier(false);
       o.rawOutlinesSingleClassifier(true);
       o.powerSpectrumSingleClassifier(false);
       o.powerSpectrumSingleClassifier(true);
       
//        basicDataTransforms("NB",10);
 //       basicDataTransforms("SVMQ",10);
 //       basicDataTransforms("NNDTW",10);
        
    }
}
