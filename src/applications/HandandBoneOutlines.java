package applications;

import utilities.*;
import utilities.ClassifierTools.ResultsStats;
import weka.core.*;
import weka.classifiers.*;

import java.text.DecimalFormat;
import java.util.*;

import fileIO.*;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.lazy.kNN;
import weka.classifiers.meta.RotationForest;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.shapelet.QualityMeasures;
import weka.filters.NormalizeCase;
import weka.filters.timeseries.SummaryStats;
import weka.filters.timeseries.shapelet_transforms.FullShapeletTransform;

/*
 * Classification problems related to Hand XRays dervided from the 
 * project with Luke Davis on automated bone ageing. Data in directory HandandBoneOutlines
 * 
 * 1. Hand Outlines: There are 1000 training cases and 300 testing cases
 *      Original problems used in [1,2] are length 2709, files HandOutlines_TEST and _TRAIN 
 *      Down sampled problems of size 512 and 1024 are also now present
 *  A basic test of the effect of the downsizing is in boneOutLinesTest()
 * 2. Bone outline classification

 * 3. TWStagePrediction: two problems: With Epiphysis (classes and without 
 * In folder ABAA_Problems/TW_Series
dp_series_tw
mp_series_tw
pp_series_tw
and in bone_outline_series
bone_series_epi.arff
bone_series_no_epi.arff

* 
* 4. Age group prediction
* directory BoneTrainTest
* Three files for each phalange, need to check what these are and if they are alligned. 
* 
 */

public class HandandBoneOutlines {
//Link to the hand outlines, you might need to change this
	static String handPath="C:\\Users\\ajb\\Dropbox\\TSC Problems\\ASMA_Datasets\\Hand_Outlines\\ARFF\\";
	static String bonePath="C:\\Users\\ajb\\Dropbox\\TSC Problems\\ASMA_Datasets\\Bone_Outlines\\ARFF\\";
        static String[] handOutlines={"hand_2709","hand_1024","hand_512"};
        static String[] boneOutlines={"pp_seg","mp_seg","dp_seg","comb_seg"};
	static DecimalFormat df=new DecimalFormat("###.###");
 	public static Classifier[] setSingleClassifiers(ArrayList<String> names){
		ArrayList<Classifier> sc2=new ArrayList<>();
		sc2.add(new kNN(1));
		names.add("NN");
		Classifier c;
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
		c=new RandomForest();
		((RandomForest)c).setNumTrees(100);
		sc2.add(c);
		names.add("RandF100");
		c=new RotationForest();
		sc2.add(c);
		names.add("RotF30");
	
		Classifier[] sc=new Classifier[sc2.size()];
		for(int i=0;i<sc.length;i++)
			sc[i]=sc2.get(i);

		return sc;
	}
   
        public static void classifierCheck(String path,String[] files) throws Exception{
            ArrayList<String> names= new ArrayList<>();
            Classifier[] c=setSingleClassifiers(names);

            Instances[] train;
            Instances[] test;
            train=new Instances[files.length];
            test=new Instances[files.length];
            for(int i=0;i<files.length;i++){
                train[i]=ClassifierTools.loadData(path+files[i]+"_train");
                test[i]=ClassifierTools.loadData(path+files[i]+"_test");
                    System.out.print("\t"+files[i]+"\t");
                
            }
            
            for(int i=0;i<c.length;i++){
                System.out.print("\nClassifier "+names.get(i));
                for(int j=0;j<train.length;j++){
                    c[i].buildClassifier(train[j]);
                    System.out.print("\t"+ClassifierTools.accuracy(test[j],c[i]));
                }
            }            
        } 
 
        
        /**
 * This outputs the cross validation accuracy and sd in latex format	
 * @param data
 * @param results
 * @param folds
 */
 public static void crossValidateHandsExperiment(Instances data, String results, int folds){
	 ArrayList<String> classifierNames=new ArrayList<String>();
	 Classifier[] c=ClassifierTools.setSingleClassifiers(classifierNames);
	 OutFile f=new OutFile(results);
	 f.writeString("\\begin{tabular}\n");
	 double[][] preds;
	 f.writeString(",");
	 for(int i=0;i<c.length;i++)
		 f.writeString(classifierNames.get(i)+"  &  ");
	 f.writeString("\n & ");
	 for(int i=0;i<c.length;i++){
		 try{
			 preds=ClassifierTools.crossValidation(c[i],data,folds);
			 ResultsStats r=new ResultsStats(preds,folds);
			 f.writeString("&"+df.format(r.accuracy)+" ("+df.format(r.sd)+") ");
			 System.out.println(classifierNames.get(i)+" Accuracy = "+df.format(r.accuracy)+" ("+df.format(r.sd)+") ");
		 }catch(Exception e)
		 {
			 System.out.println(" Error in crossValidate ="+e);
			 e.printStackTrace();
			 System.exit(0);
		 }
	 }
	 f.writeString("\\\\ \n");
 }
 public static String[] ageGroups={
			"DP_Little", // 400,645,250,3
			"DP_Middle", // 400,645,250,3
			"DP_Thumb", // 400,645,250,3
			"MP_Little", // 400,645,250,3
			"MP_Middle", // 400,645,250,3
			"PP_Little", // 400,645,250,3
			"PP_Middle", // 400,645,250,3
			"PP_Thumb" // 400,645,250,3
 };
 public static String filePath="C:\\Users\\ajb\\Dropbox\\TSC Problems\\";
 
 public static void ageGroupProblems(){
     Instances[] train= new Instances[8];
     Instances[] test= new Instances[8];
     Instances[] momentsTrain= new Instances[8];
     Instances[] momentsTest= new Instances[8];
     for(int i=0;i<ageGroups.length;i++){
         train[i]=ClassifierTools.loadData(filePath+ageGroups[i]+"\\"+ageGroups[i]+"_TRAIN");
         test[i]=ClassifierTools.loadData(filePath+ageGroups[i]+"\\"+ageGroups[i]+"_TRAIN");
//Get stats
         SummaryStats mom=new SummaryStats();
         try{
            momentsTrain[i]=mom.process(train[i]);
            momentsTest[i]=mom.process(test[i]);
//Find zero variance examples.
            for(int j=0;j<momentsTrain[i].numInstances();j++){
                if(momentsTrain[i].instance(j).value(0)==0){
                    System.out.println(" Train case "+j+" has zero mean");
                }
                if(momentsTrain[i].instance(j).value(1)==0){
                    System.out.println(" Train case "+j+" has zero variance");
                }
            }
            for(int j=0;j<momentsTest[i].numInstances();j++){
                if(momentsTest[i].instance(j).value(0)==0){
                    System.out.println(" Test case "+j+" has zero mean");
                }
                if(momentsTest[i].instance(j).value(1)==0){
                    System.out.println(" Test case "+j+" has zero variance");
                }
            }
            NormalizeCase norm=new NormalizeCase();
            norm.setNormType(NormalizeCase.NormType.STD_NORMAL);
            norm.process(train[i]);
 //           norm.process(test[i]);

            
         }catch(Exception e){
			System.out.println(" Error with file+"+ageGroups[i]+" ="+e);
                        System.out.println(" Instance 1="+train[i].instance(1));
                        
			e.printStackTrace();
			System.exit(0);
             
         } 
     }
 }
 
 public static void shapeletTWClassification(){
     Instances distalTrain,middleTrain,proximalTrain;
     Instances distalTest,middleTest,proximalTest;
     distalTrain=ClassifierTools.loadData(filePath+"DistalPhalanxTW\\DistalPhalanxTW_TRAIN");
     distalTest=ClassifierTools.loadData(filePath+"DistalPhalanxTW\\DistalPhalanxTW_TEST");
     middleTrain=ClassifierTools.loadData(filePath+"MiddlePhalanxTW\\MiddlePhalanxTW_TRAIN");
     middleTest=ClassifierTools.loadData(filePath+"MiddlePhalanxTW\\MiddlePhalanxTW_TEST");
     proximalTrain=ClassifierTools.loadData(filePath+"ProximalPhalanxTW\\ProximalPhalanxTW_TRAIN");
     proximalTest=ClassifierTools.loadData(filePath+"ProximalPhalanxTW\\ProximalPhalanxTW_TEST");
     double accDistal=shapeletTrans(distalTrain,distalTest);
     System.out.println(" Distal Acc ="+accDistal);
     double accMiddle=shapeletTrans(middleTrain,middleTest);
     System.out.println(" Middle Acc ="+accMiddle);
     double accProximal=shapeletTrans(proximalTrain,proximalTest);
     System.out.println(" Proximal Acc ="+accProximal);
     
     
 }
public static double shapeletTrans(Instances train, Instances test){
     FullShapeletTransform st =new FullShapeletTransform();
        int nosShapelets=(train.numAttributes()-1)*train.numInstances()/5;
        if(nosShapelets<FullShapeletTransform.DEFAULT_NUMSHAPELETS)
            nosShapelets=FullShapeletTransform.DEFAULT_NUMSHAPELETS;
        st.setNumberOfShapelets(nosShapelets);
        int minLength=4;
        int maxLength=(train.numAttributes()-2);
        if(maxLength<FullShapeletTransform.DEFAULT_MINSHAPELETLENGTH)
            maxLength=FullShapeletTransform.DEFAULT_MINSHAPELETLENGTH;
        st.setShapeletMinAndMax(minLength, maxLength);

/*Next you need to set the quality measure. This defaults to IG, but         
 * we recommend using the F stat. It is faster and (debatably) more accurate.
 */
        st.setQualityMeasure(QualityMeasures.ShapeletQualityChoice.F_STAT);
// You can set the filter to output details of the shapelets or not  
//        st.setLogOutputFile("ShapeletExampleLog.csv");
// Alternatively, you can turn the logging off
        st.turnOffLog();        
 
/* Thats the basic options. Now you need to perform the transform.
 * FullShapeletTransform extends the weka SimpleBatchFilter, but we have made 
 * the method process public to make usage easier.
 */
        Instances shapeletTrain=null;
        Instances shapeletTest=null;
        try {
            shapeletTrain=st.process(train);
            shapeletTest=st.process(test);
        } catch (Exception ex) {
            System.out.println("Error performing the shapelet transform"+ex);
            ex.printStackTrace();
            System.exit(0);
        }
// Build an SVM classifier
    Classifier c=new SMO();
    PolyKernel kernel = new PolyKernel();
    kernel.setExponent(1);
    ((SMO)c).setKernel(kernel);
    double a=ClassifierTools.singleTrainTestSplitAccuracy(c, shapeletTrain, shapeletTest);
    return a;
}
 
 public static void main(String[] args){
     try{
         shapeletTWClassification();
         
 //           System.out.print("\n***************** HAND OUTLINES ****************\n");         
 //        classifierCheck(handPath,handOutlines);
 //           System.out.print("\n***************** BONE OUTLINES ****************\n");         
 //        classifierCheck(bonePath,boneOutlines);
     }catch(Exception e){
             System.out.println(" ERROR ="+e);
             e.printStackTrace();
             System.exit(0);
             
             
         }
//     ageGroupProblems();
/*	 Instances train =ClassifierTools.loadData(path+"HandOutlines_TRAIN");
	 Instances test =ClassifierTools.loadData(path+"HandOutlines_TEST");
	 for(int i=0; i<test.numInstances();i++)
		 train.add(test.instance(i));
	 train.setClassIndex(train.numAttributes()-1);
	 System.out.println(" Data Loaded ");
	 crossValidateHandsExperiment(train,path+"Results1.tex",10);
*/	 
 }

	
}
