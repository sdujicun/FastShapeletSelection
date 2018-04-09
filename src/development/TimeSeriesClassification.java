package development;

import weka.core.elastic_distance_measures.BasicDTW;
import utilities.ClassifierTools;
import java.util.Random;
import fileIO.*;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;

import weka.core.*;
import weka.filters.*;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.classifiers.lazy.DTW_kNN;
import weka.classifiers.lazy.IBk;
import weka.classifiers.lazy.kNN;
import weka.classifiers.meta.RotationForest;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;


public class TimeSeriesClassification {
    
    
	/*Data sets Eamonn lists
	GOT
	original format 
	C:\Research\Data\TimeSeriesData\Time Series Classification
	/*	Synthetic Control : ControlChart
		Gun-Point  		  : GunX  
		CBF				  : CBF
		Face (all)  	  : facedbase_norm
		OSU Leaf 		  : Leaf
		Lightning-2  	  : lighting
		Lightning-7  
		Two Patterns   	  : Two Pat
		Trace Roverso 	  : Trace
		Wafer 			  :  wafer
		50Words			  :	WordSpotting
		ECG 			  : ecg

							: Pulse?
							:haptic?
	listed, not gots

		PlaneHandImages
		Car 
	*/	
//		public static String dropboxPath="C:\\Research\\Data\\Time Series Classification\\";
		public static String path="C:\\Users\\ajb\\Dropbox\\TSC Problems\\";
		public static String dropboxPath=path;                
		public static String clusterPath="TSC Problems/"; ///gpfs/sys/ajb/TSC Problems/";

                //Train Size, Test Size, Series Length, Number of Classes
		public static String[] fileNames={	
 			"Adiac", // 390,391,176,37
                        "ArrowHead",                        
			"ARSim", // 2000,2000,500,2
			"Beef", // 30,30,470,5
			"BeetleFly", // 20,20,512,2
			"BirdChicken", // 20,20,512,2
                        "Car",
			"CBF", // 30,900,128,3
			"ChlorineConcentration", // 467,3840,166,3
			"CinC_ECG_torso", // 40,1380,1639,4
			"Coffee", // 28,28,286,2
                        "Computers",
			"Cricket_X", // 390,390,300,12
			"Cricket_Y", // 390,390,300,12
			"Cricket_Z", // 390,390,300,12
			"DiatomSizeReduction", // 16,306,345,4
                        "DistalPhalanxOutlineCorrect",
                        "DistalPhalanxOutlineAgeGroup",
//                        "DistalPhalanxOIntensityAgeGroup",
                        "DistalPhalanxTW",
			"Earthquakes", // 322,139,512,2
                        "ECG200",   //This problem is flawed and can be 100% classified using the sum of squares!
			"ECGFiveDays", // 23,861,136,2
			"ElectricDevices", // 8953,7745,96,7
			"FaceAll", // 560,1690,131,14
			"FaceFour", // 24,88,350,4
			"FacesUCR", // 200,2050,131,14
			"fiftywords", // 450,455,270,50
			"fish", // 175,175,463,7
			"FordA", // 3601,1320,500,2
			"FordB", // 3636,810,500,2
			"GunPoint", // 50,150,150,2
			"HandOutlines", // 1000,370,512,2
			"Herring", // 64,64,512,2
			"Haptics", // 155,308,1092,5
			"InlineSkate", // 100,550,1882,7
			"ItalyPowerDemand", // 67,1029,24,2
                        "LargeKitchenAppliances",
			"Lightning2", // 60,61,637,2
			"Lightning7", // 70,73,319,7
			"MALLAT", // 55,2345,1024,8
			"MedicalImages", // 381,760,99,10
                        "MiddlePhalanxOutlineCorrect",
                        "MiddlePhalanxOutlineAgeGroup",
                        "MiddlePhalanxTW",
			"MoteStrain", // 20,1252,84,2
                        "NonInvasiveFatalECG_Thorax1",
                        "NonInvasiveFatalECG_Thorax2",
			"OliveOil", // 30,30,570,4
			"OSULeaf", // 200,242,427,6
                        "PhalangesOutlinesCorrect",
//                      "PassGraphs,  
                        "Plane",
                        "ProximalPhalanxOutlineCorrect",
                        "ProximalPhalanxOutlineAgeGroup",
                        "ProximalPhalanxTW",
                        "RefrigerationDevices",
                        "ScreenType",
                        "ShapeletSim", //Previouslt Synthetic data
			"ShapesAll", // 600,600,512,60
                        "SmallKitchenAppliances",
			"SonyAIBORobotSurface", // 20,601,70,2
                        "SonyAIBORobotSurfaceII",
			"StarLightCurves", // 1000,8236,1024,3
			"SwedishLeaf", // 500,625,128,15
			"Symbols", // 25,995,398,6
			"SyntheticControl", // 300,300,60,6
                        "ToeSegmentation1",
                        "ToeSegmentation2",
			"Trace", // 100,100,275,4
			"TwoLeadECG", // 23,1139,82,2
			"TwoPatterns", // 1000,4000,128,4
			"UWaveGestureLibrary_X", // 896,3582,315,8
			"UWaveGestureLibrary_Y", // 896,3582,315,8
			"UWaveGestureLibrary_Z", // 896,3582,315,8
			"UWaveGestureLibraryAll", // 896,3582,945,8
			"wafer", // 1000,6164,152,2
                        "WordSynonyms",
                        "yoga"
                };
                public static String[] fileNamesLengthSorted={
			"ItalyPowerDemand", // 67,1029,24,2
			"SyntheticControl", // 300,300,60,6
			"SonyAIBORobotSurfaceII", // 27,953,65,2
			"SonyAIBORobotSurface", // 20,601,70,2
			"TwoLeadECG", // 23,1139,82,2
			"MoteStrain", // 20,1252,84,2
			"ECG200", // 100,100,96,2
			"ElectricDevices", // 8953,7745,96,7
			"MedicalImages", // 381,760,99,10
			"CBF", // 30,900,128,3
			"SwedishLeaf", // 500,625,128,15
			"TwoPatterns", // 1000,4000,128,4
			"FaceAll", // 560,1690,131,14
			"FacesUCR", // 200,2050,131,14
			"ECGFiveDays", // 23,861,136,2
			"Plane", // 105,105,144,7
			"GunPoint", // 50,150,150,2
			"wafer", // 1000,6164,152,2
			"ChlorineConcentration", // 467,3840,166,3
			"Adiac", // 390,391,176,37
			"DP_Little", // 400,645,250,3
			"DP_Middle", // 400,645,250,3
			"DP_Thumb", // 400,645,250,3
			"MP_Little", // 400,645,250,3
			"MP_Middle", // 400,645,250,3
			"PP_Little", // 400,645,250,3
			"PP_Middle", // 400,645,250,3
			"PP_Thumb", // 400,645,250,3
			"fiftywords", // 450,455,270,50
			"WordSynonyms", // 267,638,270,25
			"Trace", // 100,100,275,4
			"Coffee", // 28,28,286,2
			"Cricket_X", // 390,390,300,12
			"Cricket_Y", // 390,390,300,12
			"Cricket_Z", // 390,390,300,12
			"UWaveGestureLibrary_X", // 896,3582,315,8
			"UWaveGestureLibrary_Y", // 896,3582,315,8
			"UWaveGestureLibrary_Z", // 896,3582,315,8
			"Lighting7", // 70,73,319,7
			"DiatomSizeReduction", // 16,306,345,4
			"FaceFour", // 24,88,350,4
			"Symbols", // 25,995,398,6
			"yoga", // 300,3000,426,2
			"OSULeaf", // 200,242,427,6
			"fish", // 175,175,463,7
			"Beef", // 30,30,470,5
			"ARSim", // 2000,2000,500,2
			"FordA", // 3601,1320,500,2
			"FordB", // 3636,810,500,2
			"BeetleFly", // 20,20,512,2
			"BirdChicken", // 20,20,512,2
			"Earthquakes", // 322,139,512,2
			"Herring", // 64,64,512,2
			"ShapesAll", // 600,600,512,60
			"OliveOil", // 30,30,570,4
			"Car", // 60,60,577,4
			"Lighting2", // 60,61,637,2
			"NonInvasiveFatalECG_Thorax1", // 1800,1965,750,42
			"NonInvasiveFatalECG_Thorax2", // 1800,1965,750,42
			"CricketAll", // 390,390,900,12
			"UWaveGestureLibraryAll", // 896,3582,945,8
			"MALLAT", // 55,2345,1024,8
			"StarLightCurves", // 1000,8236,1024,3
			"Haptics", // 155,308,1092,5
			"CinC_ECG_torso", // 40,1380,1639,4
			"InlineSkate", // 100,550,1882,7
			"HandOutlines", // 1000,370,2709,2
                };

		public static String[] fileNamesTotalSizeSorted={
			"SonyAIBORobotSurface", // 20,601,70,2
			"ItalyPowerDemand", // 67,1029,24,2
			"MoteStrain", // 20,1252,84,2
			"SonyAIBORobotSurfaceII", // 27,953,65,2
			"TwoLeadECG", // 23,1139,82,2
			"ECGFiveDays", // 23,861,136,2
			"CBF", // 30,900,128,3
			"DiatomSizeReduction", // 16,306,345,4
			"GunPoint", // 50,150,150,2
			"Coffee", // 28,28,286,2
			"FaceFour", // 24,88,350,4
			"ECG200", // 100,100,96,2
			"Symbols", // 25,995,398,6
			"BeetleFly", // 20,20,512,2
			"BirdChicken", // 20,20,512,2
			"Beef", // 30,30,470,5
			"Plane", // 105,105,144,7
			"OliveOil", // 30,30,570,4
			"SyntheticControl", // 300,300,60,6
			"Lighting7", // 70,73,319,7
			"FacesUCR", // 200,2050,131,14
			"Trace", // 100,100,275,4
			"Herring", // 64,64,512,2
			"Car", // 60,60,577,4
			"MedicalImages", // 381,760,99,10
			"Lighting2", // 60,61,637,2
			"MALLAT", // 55,2345,1024,8
			"SwedishLeaf", // 500,625,128,15
			"CinC_ECG_torso", // 40,1380,1639,4
			"Adiac", // 390,391,176,37
			"WordSynonyms", // 267,638,270,25
			"FaceAll", // 560,1690,131,14
			"ChlorineConcentration", // 467,3840,166,3
			"fish", // 175,175,463,7
			"OSULeaf", // 200,242,427,6
			"DP_Little", // 400,645,250,3
			"DP_Middle", // 400,645,250,3
			"DP_Thumb", // 400,645,250,3
			"MP_Little", // 400,645,250,3
			"MP_Middle", // 400,645,250,3
			"PP_Little", // 400,645,250,3
			"PP_Middle", // 400,645,250,3
			"PP_Thumb", // 400,645,250,3
			"Cricket_X", // 390,390,300,12
			"Cricket_Y", // 390,390,300,12
			"Cricket_Z", // 390,390,300,12
			"fiftywords", // 450,455,270,50
			"yoga", // 300,3000,426,2
			"TwoPatterns", // 1000,4000,128,4
			"wafer", // 1000,6164,152,2
			"Earthquakes", // 322,139,512,2
			"Haptics", // 155,308,1092,5
			"InlineSkate", // 100,550,1882,7
			"UWaveGestureLibrary_X", // 896,3582,315,8
			"UWaveGestureLibrary_Y", // 896,3582,315,8
			"UWaveGestureLibrary_Z", // 896,3582,315,8
			"ShapesAll", // 600,600,512,60
			"UWaveGestureLibraryAll", // 896,3582,945,8
			"ElectricDevices", // 8953,7745,96,7
			"ARSim", // 2000,2000,500,2
			"StarLightCurves", // 1000,8236,1024,3
			"NonInvasiveFatalECG_Thorax1", // 1800,1965,750,42
			"NonInvasiveFatalECG_Thorax2", // 1800,1965,750,42
			"FordA", // 3601,1320,500,2
			"FordB", // 3636,810,500,2
			"HandOutlines", 		
                };

                public static String[] fileNamesTrainSizeSorted={
			"DiatomSizeReduction", // 16,306,345,4
			"SonyAIBORobotSurface", // 20,601,70,2
			"MoteStrain", // 20,1252,84,2
			"TwoLeadECG", // 23,1139,82,2
			"ECGFiveDays", // 23,861,136,2
			"FaceFour", // 24,88,350,4
			"Symbols", // 25,995,398,6
			"Coffee", // 28,28,286,2
			"CBF", // 30,900,128,3
			"Beef", // 30,30,470,5
			"OliveOil", // 30,30,570,4
			"CinC_ECG_torso", // 40,1380,1639,4
			"GunPoint", // 50,150,150,2
			"MALLAT", // 55,2345,1024,8
			"Lighting2", // 60,61,637,2
			"Lighting2", // 60,61,637,2
			"ItalyPowerDemand", // 67,1029,24,2
			"Lighting7", // 70,73,319,7
			"Lighting7", // 70,73,319,7
			"Trace", // 100,100,275,4
			"InlineSkate", // 100,550,1882,7
			"Haptics", // 155,308,1092,5
			"fish", // 175,175,463,7
			"FacesUCR", // 200,2050,131,14
			"OSULeaf", // 200,242,427,6
			"SyntheticControl", // 300,300,60,6
			"Earthquakes", // 322,139,512,2
			"MedicalImages", // 381,760,99,10
			"Adiac", // 390,391,176,37
			"fiftywords", // 450,455,270,50
			"ChlorineConcentration", // 467,3840,166,3
			"SwedishLeaf", // 500,625,128,15
			"FaceAll", // 560,1690,131,14
			"TwoPatterns", // 1000,4000,128,4
			"wafer", // 1000,6164,152,2
			"StarLightCurves", // 1000,8236,1024,3
			"HandOutlines", // 1000,370,2709,2
			"ARSim", // 2000,2000,500,2
			"FordA", // 3601,1320,500,2
			"FordB", // 3636,810,500,2
			"ElectricDevices", // 8953,7745,96,7
                };

		//Small defined as a data set size less than 250 instances
		public static String[] smallProblems={
			"Lighting2",//60,61,637,2
			"Lighting7",//70,73,319,7
			"Adiac",//390,391,176,37
			"FaceFour",//24,88,350,4
			"50words",//450,455,270,50
			"CBF",//30,900,128,3
			"fish",//175,175,463,7
			"Gun_Point",//50,150,150,2
			"OSULeaf", //200,242,427,6
			"SwedishLeaf", //500,625,128,15
			"synthetic_control", //300,300,60,6
			"Trace",//100,100,275,4
			//Index 18, after this the data has not been normalised.
			"Beef", //30,30,470,5
			"Coffee", //28,28,286,2
			"OliveOil"
		};
//Sort by series length
                
//Sorted by training set size
		public static String[] UCRProblems={	
 			"Adiac", // 390,391,176,37
			"Beef", // 30,30,470,5
                        "Car",
			"CBF", // 30,900,128,3
			"ChlorineConcentration", // 467,3840,166,3
			"CinC_ECG_torso", // 40,1380,1639,4
			"Coffee", // 28,28,286,2
			"Cricket_X", // 390,390,300,12
			"Cricket_Y", // 390,390,300,12
			"Cricket_Z", // 390,390,300,12
			"DiatomSizeReduction", // 16,306,345,4
                        "ECG200",   //This problem is flawed and can be 100% classified using the sum of squares!
			"ECGFiveDays", // 23,861,136,2
			"FaceAll", // 560,1690,131,14
			"FaceFour", // 24,88,350,4
			"FacesUCR", // 200,2050,131,14
			"fiftywords", // 450,455,270,50
			"fish", // 175,175,463,7
			"GunPoint", // 50,150,150,2
			"Haptics", // 155,308,1092,5
			"InlineSkate", // 100,550,1882,7
			"ItalyPowerDemand", // 67,1029,24,2
			"Lighting2", // 60,61,637,2
			"Lighting7", // 70,73,319,7
			"MALLAT", // 55,2345,1024,8
			"MedicalImages", // 381,760,99,10
			"MoteStrain", // 20,1252,84,2
                        "NonInvasiveFatalECG_Thorax1",
                        "NonInvasiveFatalECG_Thorax2",
			"OliveOil", // 30,30,570,4
			"OSULeaf", // 200,242,427,6
                        "Plane",
			"SonyAIBORobotSurface", // 20,601,70,2
                        "SonyAIBORobotSurfaceII",
			"StarLightCurves", // 1000,8236,1024,3
			"SwedishLeaf", // 500,625,128,15
			"Symbols", // 25,995,398,6
			"SyntheticControl", // 300,300,60,6
			"Trace", // 100,100,275,4
			"TwoLeadECG", // 23,1139,82,2
			"TwoPatterns", // 1000,4000,128,4
			"UWaveGestureLibrary_X", // 896,3582,315,8
			"UWaveGestureLibrary_Y", // 896,3582,315,8
			"UWaveGestureLibrary_Z", // 896,3582,315,8
			"wafer", // 1000,6164,152,2
                        "WordSynonyms",
                        "yoga"
                };
//Reported Errors for 1-NN, DTW full and TO FOLLOW: DTW with window set through CV
                static public double[] ucrError1NN={0.389,0.467,0.267,0.148,0.35,0.103,0.25,0.426,0.356,0.38,0.065,0.12,0.203,0.286,0.216,0.231,0.369,0.217,0.087,0.63,0.658,0.045,0.246,0.425,0.086,0.316,0.121,0.171,0.12,0.133,0.483,0.038,0.305,0.141,0.151,0.213,0.1,0.12,0.24,0.253,0.09,0.261,0.338,0.35,0.005,0.382,0.17};
                static public double[] ucrErrorDTWFull={0.396,0.5,0.267,0.003,0.352,0.349,0.179,0.223,0.208,0.208,0.033,0.23,0.232,0.192,0.17,0.0951,0.31,0.167,0.093,0.623,0.616,0.05,0.131,0.274,0.066,0.263,0.165,0.209,0.135,0.133,0.409,0,0.275,0.169,0.093,0.21,0.05,0.007,0,0.096,0,0.273,0.366,0.342,0.02,0.351,0.164};
                
    public static Classifier[] setDefaultSingleClassifiers(ArrayList<String> names){
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
            names.add("RotF10");

            Classifier[] sc=new Classifier[sc2.size()];
            for(int i=0;i<sc.length;i++)
                    sc[i]=sc2.get(i);

            return sc;
    }
    public static void recreateUCR_1NN_Results(String resultsPath){
        DecimalFormat df = new DecimalFormat("###.###");
        OutFile of = new OutFile(resultsPath);
            try{
                System.out.println("************** EUCLIDEAN DISTANCE*******************");
                System.out.println("\t\t  UCR \t 1NN \t IBk(1) \t kNN ");
                of.writeLine("Problem,UCR1NN,UEA1NN,IBk(1),kNN");
                    for(int i=0;i<UCRProblems.length;i++)
                    {
                            Instances test=utilities.ClassifierTools.loadData(TimeSeriesClassification.path+UCRProblems[i]+"\\"+TimeSeriesClassification.UCRProblems[i]+"_TEST");
                            Instances train=utilities.ClassifierTools.loadData(TimeSeriesClassification.path+UCRProblems[i]+"\\"+TimeSeriesClassification.UCRProblems[i]+"_TRAIN");			
                            kNN a=new kNN(1);
                            a.normalise(false);
                            a.buildClassifier(train);
                            double acc1=utilities.ClassifierTools.accuracy(test,a);
                            Classifier b=new IBk(1);
                            b.buildClassifier(train);
                            double acc2=utilities.ClassifierTools.accuracy(test,b);
                            kNN knn = new kNN(100);
                            knn.normalise(false);
                            knn.setCrossValidate(true);
                            knn.buildClassifier(train);
                            double acc3=utilities.ClassifierTools.accuracy(test,knn);
                            System.out.println(UCRProblems[i]+"\t"+df.format(ucrError1NN[i])+"\t"+df.format((1-acc1))+"\t"+df.format((1-acc2))+"\t"+df.format((1-acc3))+"\t");
                            of.writeLine(UCRProblems[i]+","+df.format(ucrError1NN[i])+","+df.format((1-acc1))+","+df.format((1-acc2))+","+df.format((1-acc3)));
/*
                            b.buildClassifier(train);
                            double acc2=utilities.ClassifierTools.accuracy(test,b);
                            System.out.println(eamonnFiles[i]+" : 1NN Full DTW Error ="+(1-acc2)+" From website ="+TimeSeriesClassification.eamonnEuclidErrors[i]+" Difference ="+((1-acc1)-TimeSeriesClassification.eamonnDTW_FullErrors[i]));
                            Classifier[] sc=new Classifier[1];
*/

                    }
            }catch(Exception e){
                    System.out.println(" Error in accuracy ="+e);
                    e.printStackTrace();
                    System.exit(0);
            }
    }

    public static void recreateUCR_FullDTW_Results(String resultsPath){
        DecimalFormat df = new DecimalFormat("###.###");
        OutFile of = new OutFile(resultsPath);
            try{
                System.out.println("************** DTW DISTANCE*******************");
                System.out.println("\t\t  UCR \t DTW_1NN ");
                of.writeLine("Problem,UCR_DTW_1NN_FULL,UEA_DTW_1NN_FULL");
                    for(int i=0;i<UCRProblems.length;i++)
                    {
                            Instances test=utilities.ClassifierTools.loadData(TimeSeriesClassification.path+UCRProblems[i]+"\\"+TimeSeriesClassification.UCRProblems[i]+"_TEST");
                            Instances train=utilities.ClassifierTools.loadData(TimeSeriesClassification.path+UCRProblems[i]+"\\"+TimeSeriesClassification.UCRProblems[i]+"_TRAIN");			
                            kNN a=new kNN(1);
                            a.normalise(false);
                            BasicDTW dtw = new BasicDTW();
                            a.setDistanceFunction(dtw);
                            a.buildClassifier(train);
                            double acc1=utilities.ClassifierTools.accuracy(test,a);

                            System.out.println(UCRProblems[i]+"\t"+df.format(ucrErrorDTWFull[i])+"\t"+df.format((1-acc1)));
                            of.writeLine(UCRProblems[i]+","+df.format(ucrError1NN[i])+","+df.format((1-acc1)));
/*
                            b.buildClassifier(train);
                            double acc2=utilities.ClassifierTools.accuracy(test,b);
                            System.out.println(eamonnFiles[i]+" : 1NN Full DTW Error ="+(1-acc2)+" From website ="+TimeSeriesClassification.eamonnEuclidErrors[i]+" Difference ="+((1-acc1)-TimeSeriesClassification.eamonnDTW_FullErrors[i]));
                            Classifier[] sc=new Classifier[1];
*/

                    }
            }catch(Exception e){
                    System.out.println(" Error in accuracy ="+e);
                    e.printStackTrace();
                    System.exit(0);
            }
    }
	               
                 
		public static void main(String[] args){
 //                   formatCricket();
 //                   formatUWave();
//                       recreateUCR_1NN_Results("C:\\Users\\ajb\\Dropbox\\Results\\TimeDomain\\1NN_UCR_Comparison.csv");
 //                   recreateUCR_FullDTW_Results("C:\\Users\\ajb\\Dropbox\\Results\\TimeDomain\\CTW_1NN_UCR_Comparison.csv");
                }
 
            
	public static Classifier[] setSingleClassifiers(ArrayList<String> names){
		ArrayList<Classifier> sc2=new ArrayList<>();
		sc2.add(new kNN(1));
		names.add("NN");
		Classifier c;
//		c=new DTW_kNN(1);
//		((DTW_kNN)c).setMaxR(0.1);
//		((DTW_kNN)c).optimiseWindow(false);
		
//		sc2.add(c);
//		names.add("NNDTW");
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
		((RandomForest)c).setNumTrees(30);
		sc2.add(c);
		names.add("RandF30");
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
   
  	public static Classifier[] setSimpleClassifiers(ArrayList<String> names){
		ArrayList<Classifier> sc2=new ArrayList<>();
		sc2.add(new kNN(1));
		names.add("NN");
		Classifier c;
		c=new DTW_kNN(1);
		((DTW_kNN)c).setMaxR(1);
		((DTW_kNN)c).optimiseWindow(false);
		
//		sc2.add(c);
//		names.add("NNDTW");
//		sc2.add(new NaiveBayes());
//		names.add("NB");
		sc2.add(new J48());
		names.add("C45");
//		c=new SMO();
//		PolyKernel kernel = new PolyKernel();
//		kernel.setExponent(1);
//		((SMO)c).setKernel(kernel);
//		sc2.add(c);
//		names.add("SVML");

		Classifier[] sc=new Classifier[sc2.size()];
		for(int i=0;i<sc.length;i++)
			sc[i]=sc2.get(i);

		return sc;
	}
             
//Method to apply a BatchFilter to a Instances
            public static double assessFilter(Instances train, Instances test, SimpleBatchFilter s, Classifier c){
                double a=0;
                try{
//                    Instances trainNew=s.process(train);
//                    Instances testNew=s.process(test);
                    s.setInputFormat(train);
                    Instances trainNew=Filter.useFilter(train, s);
                    s.setInputFormat(test);
                    Instances testNew=Filter.useFilter(test, s);
                    
                    c.buildClassifier(trainNew);
                    a=ClassifierTools.accuracy(testNew,c);
                    
                }catch(Exception e){
                   System.out.println("\n Error: ="+e);
                   e.printStackTrace();
                   System.exit(0);
                }
                return a;
                
            }
            
 
       public static void formatUWave(){
                Instances x=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\UWaveGestureLibrary_X\\UWaveGestureLibrary_X_TEST");
                Instances y=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\UWaveGestureLibrary_Y\\UWaveGestureLibrary_Y_TEST");
                Instances z=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\UWaveGestureLibrary_Z\\UWaveGestureLibrary_Z_TEST");


//TEST                
                x.setClassIndex(-1);
                x.deleteAttributeAt(x.numAttributes()-1);
                y.setClassIndex(-1);
                y.deleteAttributeAt(y.numAttributes()-1);
                Instances all=Instances.mergeInstances(x,Instances.mergeInstances(y,z));
                OutFile of=new OutFile("C:\\Users\\ajb\\Dropbox\\TSC Problems\\UWaveGestureLibraryAll\\UWaveGestureLibraryAll_TEST.arff");
                of.writeLine(all.toString());
//TRAIN
                Instances z2=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\UWaveGestureLibrary_Z\\UWaveGestureLibrary_Z_TRAIN");
                Instances x2=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\UWaveGestureLibrary_X\\UWaveGestureLibrary_X_TRAIN");
                Instances y2=ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\UWaveGestureLibrary_Y\\UWaveGestureLibrary_Y_TRAIN");

                x2.setClassIndex(-1);
                x2.deleteAttributeAt(x2.numAttributes()-1);
                y2.setClassIndex(-1);
                y2.deleteAttributeAt(y2.numAttributes()-1);
                Instances all2=Instances.mergeInstances(x2,Instances.mergeInstances(y2,z2));
                OutFile of2=new OutFile("C:\\Users\\ajb\\Dropbox\\TSC Problems\\UWaveGestureLibraryAll\\UWaveGestureLibraryAll_TRAIN.arff");
                of2.writeLine(all2.toString());
            
            }

}


