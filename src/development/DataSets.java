/*
 Class containing data sets used in various TSC papers
 */
package development;

import fileIO.InFile;
import fileIO.OutFile;
import java.io.File;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Random;
import java.util.TreeSet;
import utilities.ClassifierTools;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.timeseries.SummaryStats;

/**
 *
 * @author ajb
 */
public class DataSets {

	public static final String clusterPath = "/gpfs/home/ajb/";
	//public static final String dropboxPath = "C:/Users/ajb/Dropbox/";
	public static final String dropboxPath="G:/Êý¾Ý/";
	public static final String path = dropboxPath;

//	public static String problemPath = path + "/TSC Problems/";
//	public static String resultsPath = path + "Results/";
	public static String problemPath = path + "TSC Problems/";
	public static String resultsPath = path + "Results/";
	public static String uciPath = path + "UCI Classification Problems/";

	// ALL of our TSC data sets
	// <editor-fold defaultstate="collapsed"
	// desc="fileNames: The new 85 UCR datasets">
	public static String[] fileNames = {
			// Train Size, Test Size, Series Length, Nos Classes
			// Train Size, Test Size, Series Length, Nos Classes
			"Adiac", // 390,391,176,37
			"ArrowHead", // 36,175,251,3
			"Beef", // 30,30,470,5
			"BeetleFly", // 20,20,512,2
			"BirdChicken", // 20,20,512,2
			"Car", // 60,60,577,4
			"CBF", // 30,900,128,3
			"ChlorineConcentration", // 467,3840,166,3
			"CinCECGtorso", // 40,1380,1639,4
			"Coffee", // 28,28,286,2
			"Computers", // 250,250,720,2
			"CricketX", // 390,390,300,12
			"CricketY", // 390,390,300,12
			"CricketZ", // 390,390,300,12
			"DiatomSizeReduction", // 16,306,345,4
			"DistalPhalanxOutlineCorrect", // 600,276,80,2
			"DistalPhalanxOutlineAgeGroup", // 400,139,80,3
			"DistalPhalanxTW", // 400,139,80,6
			"Earthquakes", // 322,139,512,2
			"ECG200", // 100, 100, 96
			"ECG5000", // 4500, 500,140
			"ECGFiveDays", // 23,861,136,2
			"ElectricDevices", // 8926,7711,96,7
			"FaceAll", // 560,1690,131,14
			"FaceFour", // 24,88,350,4
			"FacesUCR", // 200,2050,131,14
			"FiftyWords", // 450,455,270,50
			"Fish", // 175,175,463,7
			"FordA", // 3601,1320,500,2
			"FordB", // 3636,810,500,2
			"GunPoint", // 50,150,150,2
			"Ham", // 105,109,431
			"HandOutlines", // 1000,370,2709,2
			"Haptics", // 155,308,1092,5
			"Herring", // 64,64,512,2
			"InlineSkate", // 100,550,1882,7
			"InsectWingbeatSound",// 1980,220,256
			"ItalyPowerDemand", // 67,1029,24,2
			"LargeKitchenAppliances", // 375,375,720,3
			"Lightning2", // 60,61,637,2
			"Lightning7", // 70,73,319,7
			"Mallat", // 55,2345,1024,8
			"Meat",// 60,60,448
			"MedicalImages", // 381,760,99,10
			"MiddlePhalanxOutlineCorrect", // 600,291,80,2
			"MiddlePhalanxOutlineAgeGroup", // 400,154,80,3
			"MiddlePhalanxTW", // 399,154,80,6
			"MoteStrain", // 20,1252,84,2
			"NonInvasiveFatalECGThorax1", // 1800,1965,750,42
			"NonInvasiveFatalECGThorax2", // 1800,1965,750,42
			"OliveOil", // 30,30,570,4
			"OSULeaf", // 200,242,427,6
			"PhalangesOutlinesCorrect", // 1800,858,80,2
			"Phoneme",// 1896,214, 1024
			"Plane", // 105,105,144,7
			"ProximalPhalanxOutlineCorrect", // 600,291,80,2
			"ProximalPhalanxOutlineAgeGroup", // 400,205,80,3
			"ProximalPhalanxTW", // 400,205,80,6
			"RefrigerationDevices", // 375,375,720,3
			"ScreenType", // 375,375,720,3
			"ShapeletSim", // 20,180,500,2
			"ShapesAll", // 600,600,512,60
			"SmallKitchenAppliances", // 375,375,720,3
			"SonyAIBORobotSurface1", // 20,601,70,2
			"SonyAIBORobotSurface2", // 27,953,65,2
			"StarlightCurves", // 1000,8236,1024,3
			"Strawberry",// 370,613,235
			"SwedishLeaf", // 500,625,128,15
			"Symbols", // 25,995,398,6
			"SyntheticControl", // 300,300,60,6
			"ToeSegmentation1", // 40,228,277,2
			"ToeSegmentation2", // 36,130,343,2
			"Trace", // 100,100,275,4
			"TwoLeadECG", // 23,1139,82,2
			"TwoPatterns", // 1000,4000,128,4
			"UWaveGestureLibraryX", // 896,3582,315,8
			"UWaveGestureLibraryY", // 896,3582,315,8
			"UWaveGestureLibraryZ", // 896,3582,315,8
			"UWaveGestureLibraryAll", // 896,3582,945,8
			"Wafer", // 1000,6164,152,2
			"Wine",// 54 57 234
			"WordSynonyms", // 267,638,270,25
			"Worms", // 77, 181,900,5
			"WormsTwoClass",// 77, 181,900,5
			"Yoga" // 300,3000,426,2
	};
	// </editor-fold>

	// <editor-fold defaultstate="collapsed"
	// desc="five splits of the new 85 UCR datasets">
	public static String[][] fiveSplits = { { "Adiac", // 390,391,176,37
			"ArrowHead", // 36,175,251,3
			"Beef", // 30,30,470,5
			"BeetleFly", // 20,20,512,2
			"BirdChicken", // 20,20,512,2
			"Car", // 60,60,577,4
			"CBF", // 30,900,128,3
			"ChlorineConcentration", // 467,3840,166,3
			"CinCECGtorso", // 40,1380,1639,4
			"Coffee", // 28,28,286,2
			"Computers", // 250,250,720,2
			"CricketX", // 390,390,300,12
			"CricketY", // 390,390,300,12
			"CricketZ", // 390,390,300,12
			"DiatomSizeReduction", // 16,306,345,4
			"DistalPhalanxOutlineCorrect", // 600,276,80,2
			"DistalPhalanxOutlineAgeGroup", // 400,139,80,3
			"DistalPhalanxTW", // 400,139,80,6
			"Earthquakes" // 322,139,512,2
	}, { "ECG200", // 100, 100, 96
			"ECG5000", // 4500, 500,140
			"ECGFiveDays", // 23,861,136,2
			"FaceFour", // 24,88,350,4
			"FacesUCR", // 200,2050,131,14
			"FiftyWords", // 450,455,270,50
			"Fish", // 175,175,463,7
			"GunPoint", // 50,150,150,2
			"Ham", // 105,109,431
			"Haptics", // 155,308,1092,5
			"Herring", // 64,64,512,2
			"ItalyPowerDemand", // 67,1029,24,2
			"LargeKitchenAppliances", // 375,375,720,3
			"Lightning2", // 60,61,637,2
			"Lightning7", // 70,73,319,7
			"Mallat", // 55,2345,1024,8
			"Meat",// 60,60,448
			"MedicalImages", // 381,760,99,10
	}, { "MiddlePhalanxOutlineCorrect", // 600,291,80,2
			"MiddlePhalanxOutlineAgeGroup", // 400,154,80,3
			"MiddlePhalanxTW", // 399,154,80,6
			"MoteStrain", // 20,1252,84,2
			"OliveOil", // 30,30,570,4
			"OSULeaf", // 200,242,427,6
			"Plane", // 105,105,144,7
			"ProximalPhalanxOutlineCorrect", // 600,291,80,2
			"ProximalPhalanxOutlineAgeGroup", // 400,205,80,3
			"ProximalPhalanxTW", // 400,205,80,6
			"RefrigerationDevices", // 375,375,720,3
			"ScreenType", // 375,375,720,3
			"ShapeletSim", // 20,180,500,2
			"SmallKitchenAppliances", // 375,375,720,3
			"SonyAIBORobotSurface1", // 20,601,70,2
			"SonyAIBORobotSurface2", // 27,953,65,2
			"Strawberry",// 370,613,235
			"SwedishLeaf", // 500,625,128,15
			"Symbols", // 25,995,398,6
			"SyntheticControl" // 300,300,60,6
	}, { "ToeSegmentation1", // 40,228,277,2
			"ToeSegmentation2", // 36,130,343,2
			"Trace", // 100,100,275,4
			"TwoLeadECG", // 23,1139,82,2
			"TwoPatterns", // 1000,4000,128,4
			"UWaveGestureLibraryX", // 896,3582,315,8
			"UWaveGestureLibraryY", // 896,3582,315,8
			"UWaveGestureLibraryZ", // 896,3582,315,8
			"Wafer", // 1000,6164,152,2
			"Wine",// 54 57 234
			"WordSynonyms", // 267,638,270,25
			"Worms", // 77, 181,900,5
			"WormsTwoClass",// 77, 181,900,5
			"Yoga", // 300,3000,426,2
			"InlineSkate", // 100,550,1882,7
			"InsectWingbeatSound",// 1980,220,256
			"FaceAll", // 560,1690,131,14
			"PhalangesOutlinesCorrect", // 1800,858,80,2
			"Phoneme", // 1896,214, 1024
			"ShapesAll", // 600,600,512,60
	}, { "ElectricDevices", // 8926,7711,96,7
			"FordA", // 3601,1320,500,2
			"FordB", // 3636,810,500,2
			"HandOutlines", // 1000,370,2709,2
			"NonInvasiveFatalECGThorax1", // 1800,1965,750,42
			"NonInvasiveFatalECGThorax2", // 1800,1965,750,42
			"StarlightCurves", // 1000,8236,1024,3
			"UWaveGestureLibraryAll", // 896,3582,945,8
	} };
	// </editor-fold>

	// UCR data sets
	// <editor-fold defaultstate="collapsed" desc="ucrNames: 46 UCR Data sets">
	public static String[] ucrNames = { "Adiac", // 390,391,176,37
			"Beef", // 30,30,470,5
			"Car", // 60,60,577,4
			"CBF", // 30,900,128,3
			"ChlorineConcentration", // 467,3840,166,3
			"CinCECGtorso", // 40,1380,1639,4
			"Coffee", // 28,28,286,2
			"CricketX", // 390,390,300,12
			"CricketY", // 390,390,300,12
			"CricketZ", // 390,390,300,12
			"DiatomSizeReduction", // 16,306,345,4
			"ECGFiveDays", // 23,861,136,2
			"FaceAll", // 560,1690,131,14
			"FaceFour", // 24,88,350,4
			"FacesUCR", // 200,2050,131,14
			"FiftyWords", // 450,455,270,50
			"Fish", // 175,175,463,7
			"GunPoint", // 50,150,150,2
			"Haptics", // 155,308,1092,5
			"InlineSkate", // 100,550,1882,7
			"ItalyPowerDemand", // 67,1029,24,2
			"Lightning2", // 60,61,637,2
			"Lightning7", // 70,73,319,7
			"Mallat", // 55,2345,1024,8
			"MedicalImages", // 381,760,99,10
			"MoteStrain", // 20,1252,84,2
			"NonInvasiveFatalECGThorax1", // 1800,1965,750,42
			"NonInvasiveFatalECGThorax2", // 1800,1965,750,42
			"OliveOil", // 30,30,570,4
			"OSULeaf", // 200,242,427,6
			"Plane", // 105,105,144,7
			"SonyAIBORobotSurface1", // 20,601,70,2
			"SonyAIBORobotSurface2", // 27,953,65,2
			"StarLightCurves", // 1000,8236,1024,3
			"SwedishLeaf", // 500,625,128,15
			"Symbols", // 25,995,398,6
			"SyntheticControl", // 300,300,60,6
			"Trace", // 100,100,275,4
			"TwoLeadECG", // 23,1139,82,2
			"TwoPatterns", // 1000,4000,128,4
			"UWaveGestureLibraryX", // 896,3582,315,8
			"UWaveGestureLibraryY", // 896,3582,315,8
			"UWaveGestureLibraryZ", // 896,3582,315,8
			"Wafer", // 1000,6164,152,2
			"WordSynonyms", // 267,638,270,25
			"Yoga" // 300,3000,426,2
	};
	
	public static String[] DSUsed = { 
		"Adiac", // 390,391,176,37
		"Beef", // 30,30,470,5
		"CBF", // 30,900,128,3
		"ChlorineConcentration",
		"CinCECGtorso", // 40,1380,1639,4
		"Coffee", // 28,28,286,2
		"CricketX", // 390,390,300,12
		"CricketY", // 390,390,300,12
		"CricketZ", // 390,390,300,12
		"DiatomSizeReduction", // 16,306,345,4
		"ECG200", // 100, 100, 96
		"ECGFiveDays", // 23,861,136,2
		"FaceAll", // 560,1690,131,14
		"FaceFour", // 24,88,350,4
		"FacesUCR", // 200,2050,131,14
		"FiftyWords", // 450,455,270,50
		"Fish", // 175,175,463,7
		"GunPoint", // 50,150,150,2
		"Haptics", // 155,308,1092,5
		"InlineSkate", // 100,550,1882,7
		"ItalyPowerDemand", // 67,1029,24,2
		"Lightning2", // 60,61,637,2
		"Lightning7", // 70,73,319,7
		"Mallat", // 55,2345,1024,8
		"MedicalImages", // 381,760,99,10
		"MoteStrain", // 20,1252,84,2
		"NonInvasiveFatalECGThorax1", // 1800,1965,750,42
		"NonInvasiveFatalECGThorax2", // 1800,1965,750,42
		"OliveOil", // 30,30,570,4
		"OSULeaf", // 200,242,427,6
		"StarLightCurves", // 1000,8236,1024,3
		"SonyAIBORobotSurface1", // 20,601,70,2
		"SonyAIBORobotSurface2", // 27,953,65,2
		"SwedishLeaf", // 500,625,128,15
		"Symbols", // 25,995,398,6
		"SyntheticControl", // 300,300,60,6
		"Trace", // 100,100,275,4
		"TwoLeadECG", // 23,1139,82,2
		"TwoPatterns", // 1000,4000,128,4
		"UWaveGestureLibraryX", // 896,3582,315,8
		"UWaveGestureLibraryY", // 896,3582,315,8
		"UWaveGestureLibraryZ", // 896,3582,315,8
		"Wafer", // 1000,6164,152,2
		"WordSynonyms", // 267,638,270,25
		"Yoga" // 300,3000,426,2
};
	// </editor-fold>

	// Small UCR data sets
	// <editor-fold defaultstate="collapsed"
	// desc="ucrSmall: Small UCR Data sets">
	public static String[] ucrSmall = { "Beef", // 30,30,470,5
			"Car", // 60,60,577,4
			"Coffee", // 28,28,286,2
			"Cricket_X", // 390,390,300,12
			"Cricket_Y", // 390,390,300,12
			"Cricket_Z", // 390,390,300,12
			"DiatomSizeReduction", // 16,306,345,4
			"fish", // 175,175,463,7
			"GunPoint", // 50,150,150,2
			"ItalyPowerDemand", // 67,1029,24,2
			"MoteStrain", // 20,1252,84,2
			"OliveOil", // 30,30,570,4
			"Plane", // 105,105,144,7
			"SonyAIBORobotSurface", // 20,601,70,2
			"SonyAIBORobotSurfaceII", // 27,953,65,2
			"SyntheticControl", // 300,300,60,6
			"Trace", // 100,100,275,4
			"TwoLeadECG", // 23,1139,82,2
	};
	// </editor-fold>

	// <editor-fold defaultstate="collapsed" desc="spectral: Spectral data">
	public static String[] spectral = {
			// Train Size, Test Size, Series Length, Nos Classes
			"Beef", // 30,30,470,5
			"Coffee", // 28,28,286,2
			"Ham", "Meat", "OliveOil", // 30,30,570,4
			"Strawberry", "Wine",
	// To add: spirits
	};
	// </editor-fold>

	// Small Files
	// <editor-fold defaultstate="collapsed" desc="smallFileNames:">
	public static String[] smallFileNames = { "Beef", "BeetleFly", "BirdChicken", "FaceFour", "Plane", "FacesUCR" };

	/*
	 * //Train Size, Test Size, Series Length, Nos Classes "Adiac", //
	 * 390,391,176,37 "ArrowHead", // 36,175,251,3 "Beef", // 30,30,470,5
	 * "BeetleFly", // 20,20,512,2 "BirdChicken", // 20,20,512,2 "Car", //
	 * 60,60,577,4 "CBF", // 30,900,128,3 "ChlorineConcentration", //
	 * 467,3840,166,3 "CinC_ECG_torso", // 40,1380,1639,4 "Computers", //
	 * 250,250,720,2 "Cricket_X", // 390,390,300,12 "Cricket_Y", //
	 * 390,390,300,12 "Cricket_Z", // 390,390,300,12 "DiatomSizeReduction", //
	 * 16,306,345,4 "DistalPhalanxOutlineAgeGroup", // 400,139,80,3
	 * "DistalPhalanxTW", // 400,139,80,6 "Earthquakes", // 322,139,512,2
	 * "ECGFiveDays", // 23,861,136,2 "ElectricDevices", // 8926,7711,96,7
	 * "FaceAll", // 560,1690,131,14 "FacesUCR", // 200,2050,131,14
	 * "fiftywords", // 450,455,270,50 "fish", // 175,175,463,7 "FordA", //
	 * 3601,1320,500,2 "FordB", // 3636,810,500,2 "GunPoint", // 50,150,150,2
	 * "Ham", "HandOutlines", // 1000,370,2709,2 "Haptics", // 155,308,1092,5
	 * "Herring", // 64,64,512,2 "InlineSkate", // 100,550,1882,7
	 * "ItalyPowerDemand", // 67,1029,24,2 "LargeKitchenAppliances", //
	 * 375,375,720,3 "Lightning2", // 60,61,637,2 "Lightning7", // 70,73,319,7
	 * "MALLAT", // 55,2345,1024,8 // "Meat", "MiddlePhalanxOutlineAgeGroup", //
	 * 400,154,80,3 "MiddlePhalanxTW", // 399,154,80,6 "MoteStrain", //
	 * 20,1252,84,2 "NonInvasiveFatalECG_Thorax1", // 1800,1965,750,42
	 * "NonInvasiveFatalECG_Thorax2", // 1800,1965,750,42 "OSULeaf", //
	 * 200,242,427,6 "PhalangesOutlinesCorrect", // 1800,858,80,2 "Plane", //
	 * 105,105,144,7 "ProximalPhalanxOutlineAgeGroup", // 400,205,80,3
	 * "ProximalPhalanxTW", // 400,205,80,6 "RefrigerationDevices", //
	 * 375,375,720,3 "ScreenType", // 375,375,720,3 // "ShapeletSim", //
	 * 20,180,500,2 "ShapesAll", // 600,600,512,60 "SmallKitchenAppliances", //
	 * 375,375,720,3 "SonyAIBORobotSurfaceII", // 27,953,65,2 "StarLightCurves",
	 * // 1000,8236,1024,3 "Strawberry", "Symbols", // 25,995,398,6
	 * "TwoLeadECG", // 23,1139,82,2 "TwoPatterns", // 1000,4000,128,4
	 * "UWaveGestureLibrary_X", // 896,3582,315,8 "UWaveGestureLibrary_Y", //
	 * 896,3582,315,8 "UWaveGestureLibrary_Z", // 896,3582,315,8
	 * "UWaveGestureLibraryAll", // 896,3582,945,8 "wafer", // 1000,6164,152,2
	 * // "Wine", "WordSynonyms", // 267,638,270,25 "Worms", "WormsTwoClass",
	 * "yoga" // 300,3000,426,2 };
	 */
	// </editor-fold>

	// Sets used in papers

	// <editor-fold defaultstate="collapsed" desc="rakthanmanon13fastshapelets">
	/*
	 * Problem sets used in @article{rakthanmanon2013fast, title={Fast
	 * Shapelets: A Scalable Algorithm for Discovering Time Series Shapelets},
	 * author={Rakthanmanon, T. and Keogh, E.}, journal={Proceedings of the 13th
	 * {SIAM} International Conference on Data Mining}, year={2013} } All
	 * included except Cricket. There are three criket problems and they are not
	 * alligned, the class values in the test set dont match
	 */
	public static String[] fastShapeletProblems = { "ItalyPowerDemand", // 67,1029,24,2
			"MoteStrain", // 20,1252,84,2
			"SonyAIBORobotSurfaceII", // 27,953,65,2
			"SonyAIBORobotSurface", // 20,601,70,2
			"Beef", // 30,30,470,5
			"GunPoint", // 50,150,150,2
			"TwoLeadECG", // 23,1139,82,2
			"Adiac", // 390,391,176,37
			"CBF", // 30,900,128,3
			"ChlorineConcentration", // 467,3840,166,3
			"Coffee", // 28,28,286,2
			"DiatomSizeReduction", // 16,306,345,4
			"ECGFiveDays", // 23,861,136,2
			"FaceFour", // 24,88,350,4
			"FacesUCR", // 200,2050,131,14
			"fish", // 175,175,463,7
			"Lighting2", // 60,61,637,2
			"Lighting7", // 70,73,319,7
			"FaceAll", // 560,1690,131,14
			"MALLAT", // 55,2345,1024,8
			"MedicalImages", // 381,760,99,10
			"OliveOil", // 30,30,570,4
			"OSULeaf", // 200,242,427,6
			"SwedishLeaf", // 500,625,128,15
			"Symbols", // 25,995,398,6
			"SyntheticControl", // 300,300,60,6
			"Trace", // 100,100,275,4
			"wafer", // 1000,6164,152,2
			"yoga", "FaceAll", "TwoPatterns", "CinC_ECG_torso" // 40,1380,1639,4
	};
	// </editor-fold>

	// <editor-fold defaultstate="collapsed" desc="marteau09stiffness: TWED">
	public static String[] marteau09stiffness = { "SyntheticControl", // 300,300,60,6
			"GunPoint", // 50,150,150,2
			"CBF", // 30,900,128,3
			"FaceAll", // 560,1690,131,14
			"OSULeaf", // 200,242,427,6
			"SwedishLeaf", // 500,625,128,15
			"fiftywords", // 450,455,270,50
			"Trace", // 100,100,275,4
			"TwoPatterns", // 1000,4000,128,4
			"wafer", // 1000,6164,152,2
			"FaceFour", // 24,88,350,4
			"Lightning2", // 60,61,637,2
			"Lightning7", // 70,73,319,7
			"ECG200", // 100,100,96,2
			"Adiac", // 390,391,176,37
			"yoga", // 300,3000,426,2
			"fish", // 175,175,463,7
			"Coffee", // 28,28,286,2
			"OliveOil", // 30,30,570,4
			"Beef" // 30,30,470,5
	};
	// </editor-fold>

	// <editor-fold defaultstate="collapsed"
	// desc="stefan13movesplit: Move-Split-Merge">
	public static String[] stefan13movesplit = { "Coffee", // 28,28,286,2
			"CBF", // 30,900,128,3
			"ECG200", // 100,100,96,2
			"SyntheticControl", // 300,300,60,6
			"GunPoint", // 50,150,150,2
			"FaceFour", // 24,88,350,4
			"Lightning7", // 70,73,319,7
			"Trace", // 100,100,275,4
			"Adiac", // 390,391,176,37
			"Beef", // 30,30,470,5
			"Lightning2", // 60,61,637,2
			"OliveOil", // 30,30,570,4
			"OSULeaf", // 200,242,427,6
			"SwedishLeaf", // 500,625,128,15
			"fish", // 175,175,463,7
			"FaceAll", // 560,1690,131,14
			"fiftywords", // 450,455,270,50
			"TwoPatterns", // 1000,4000,128,4
			"wafer", // 1000,6164,152,2
			"yoga" // 300,3000,426,2
	};
	// </editor-fold>

	// UCI Classification problems: NOTE THESE ARE -train NOT _TRAIN
	// <editor-fold defaultstate="collapsed" desc="UCI Classification problems">
	public static String[] uciFileNames = { "abalone", "banana", "cancer", "clouds", "concentric", "diabetes", "ecoli", "german", "glass2", "glass6", "haberman", "heart", "ionosphere", "liver", "magic", "pendigitis", "phoneme", "ringnorm", "satimage", "segment", "sonar", "thyroid", "twonorm", "vehicle", "vowel", "waveform", "wdbc", "wins", "yeast" };
	// </editor-fold>

	// Gavin banana
	/*
	 * flare_solar splice transfusion breast_cancer synthetic vertebra image
	 * spambase tiianic
	 */

	// ALL of our TSC data sets
	// <editor-fold defaultstate="collapsed" desc="fileNames: All Data sets">
	public static String[] datasetsForDAMI2014_Lines = {
			// Train Size, Test Size, Series Length, Nos Classes
			"Adiac", // 390,391,176,37
			"ArrowHead", // 36,175,251,3
			"Beef", // 30,30,470,5
			"BeetleFly", // 20,20,512,2
			"BirdChicken", // 20,20,512,2
			"Car", // 60,60,577,4
			"CBF", // 30,900,128,3
			"ChlorineConcentration", // 467,3840,166,3
			"CinC_ECG_torso", // 40,1380,1639,4
			"Coffee", // 28,28,286,2
			"Computers", // 250,250,720,2
			"Cricket_X", // 390,390,300,12
			"Cricket_Y", // 390,390,300,12
			"Cricket_Z", // 390,390,300,12
			"DiatomSizeReduction", // 16,306,345,4
			"DistalPhalanxOutlineCorrect", // 600,276,80,2
			"DistalPhalanxOutlineAgeGroup", // 400,139,80,3
			"DistalPhalanxTW", // 400,139,80,6
			"Earthquakes", // 322,139,512,2
			"ECGFiveDays", // 23,861,136,2
			"ElectricDevices", // 8926,7711,96,7
			"FaceAll", // 560,1690,131,14
			"FaceFour", // 24,88,350,4
			"FacesUCR", // 200,2050,131,14
			"fiftywords", // 450,455,270,50
			"fish", // 175,175,463,7
			"FordA", // 3601,1320,500,2
			"FordB", // 3636,810,500,2
			"GunPoint", // 50,150,150,2
			"HandOutlines", // 1000,370,2709,2
			"Haptics", // 155,308,1092,5
			"Herring", // 64,64,512,2
			"InlineSkate", // 100,550,1882,7
			"ItalyPowerDemand", // 67,1029,24,2
			"LargeKitchenAppliances", // 375,375,720,3
			"Lightning2", // 60,61,637,2
			"Lightning7", // 70,73,319,7
			"MALLAT", // 55,2345,1024,8
			"MedicalImages", // 381,760,99,10
			"MiddlePhalanxOutlineCorrect", // 600,291,80,2
			"MiddlePhalanxOutlineAgeGroup", // 400,154,80,3
			"MiddlePhalanxTW", // 399,154,80,6
			"MoteStrain", // 20,1252,84,2
			"NonInvasiveFatalECG_Thorax1", // 1800,1965,750,42
			"NonInvasiveFatalECG_Thorax2", // 1800,1965,750,42
			"OliveOil", // 30,30,570,4
			"OSULeaf", // 200,242,427,6
			"PhalangesOutlinesCorrect", // 1800,858,80,2
			"Plane", // 105,105,144,7
			"ProximalPhalanxOutlineCorrect", // 600,291,80,2
			"ProximalPhalanxOutlineAgeGroup", // 400,205,80,3
			"ProximalPhalanxTW", // 400,205,80,6
			"RefrigerationDevices", // 375,375,720,3
			"ScreenType", // 375,375,720,3
			"ShapeletSim", // 20,180,500,2
			"ShapesAll", // 600,600,512,60
			"SmallKitchenAppliances", // 375,375,720,3
			"SonyAIBORobotSurface", // 20,601,70,2
			"SonyAIBORobotSurfaceII", // 27,953,65,2
			"StarLightCurves", // 1000,8236,1024,3
			"SwedishLeaf", // 500,625,128,15
			"Symbols", // 25,995,398,6
			"SyntheticControl", // 300,300,60,6
			"ToeSegmentation1", // 40,228,277,2
			"ToeSegmentation2", // 36,130,343,2
			"Trace", // 100,100,275,4
			"TwoLeadECG", // 23,1139,82,2
			"TwoPatterns", // 1000,4000,128,4
			"UWaveGestureLibrary_X", // 896,3582,315,8
			"UWaveGestureLibrary_Y", // 896,3582,315,8
			"UWaveGestureLibrary_Z", // 896,3582,315,8
			"wafer", // 1000,6164,152,2
			"WordSynonyms", // 267,638,270,25
			"yoga" // 300,3000,426,2
	};
	// </editor-fold>

	public static String[] notNormalised = { "ArrowHead", "Beef", "BeetleFly", "BirdChicken", "Coffee", "Computers", "Cricket_X", "Cricket_Y", "Cricket_Z", "DistalPhalanxOutlineAgeGroup", "DistalPhalanxOutlineCorrect", "DistalPhalanxTW", "ECG200", "Earthquakes", "ElectricDevices", "FordA", "FordB", "Ham", "Herring", "LargeKitchenAppliances", "Meat", "MiddlePhalanxOutlineAgeGroup", "MiddlePhalanxOutlineCorrect", "MiddlePhalanxTW", "OliveOil", "PhalangesOutlinesCorrect", "Plane", "ProximalPhalanxOutlineAgeGroup", "ProximalPhalanxOutlineCorrect", "ProximalPhalanxTW", "RefrigerationDevices", "ScreenType", "ShapeletSim", "ShapesAll", "SmallKitchenAppliances", "Strawberry", "ToeSegmentation1", "ToeSegmentation2", "UWaveGestureLibraryAll", "UWaveGestureLibrary_Z", "Wine", "Worms",
			"WormsTwoClass", "fish" };

	public static void processUCRData() {
		System.out.println(" nos files =" + ucrNames.length);
		String s;
		for (int str = 39; str < 43; str++) {
			s = ucrNames[str];
			InFile trainF = new InFile(problemPath + s + "/" + s + "_TRAIN");
			InFile testF = new InFile(problemPath + s + "/" + s + "_TEST");
			Instances train = ClassifierTools.loadData(problemPath + s + "/" + s + "_TRAIN");
			Instances test = ClassifierTools.loadData(problemPath + s + "/" + s + "_TEST");
			int trainSize = trainF.countLines();
			int testSize = testF.countLines();
			Attribute a = train.classAttribute();
			String tt = a.value(0);
			int first = Integer.parseInt(tt);
			System.out.println(s + " First value =" + tt + " first =" + first);
			if (trainSize != train.numInstances() || testSize != test.numInstances()) {
				System.out.println(" ERROR MISMATCH SIZE TRAIN=" + trainSize + "," + train.numInstances() + " TEST =" + testSize + "," + test.numInstances());
				System.exit(0);
			}
			trainF = new InFile(problemPath + s + "/" + s + "_TRAIN");
			testF = new InFile(problemPath + s + "/" + s + "_TEST");
			File dir = new File(problemPath + s);
			if (!dir.exists()) {
				dir.mkdir();
			}
			OutFile newTrain = new OutFile(problemPath + s + "/" + s + "_TRAIN.arff");
			OutFile newTest = new OutFile(problemPath + s + "/" + s + "_TEST.arff");
			Instances header = new Instances(train, 0);
			newTrain.writeLine(header.toString());
			newTest.writeLine(header.toString());
			for (int i = 0; i < trainSize; i++) {
				String line = trainF.readLine();
				line = line.trim();
				String[] split = line.split("/s+");
				try {
					// System.out.println(split[0]+"First ="+split[1]+" last ="+split[split.length-1]+" length = "+split.length+" nos atts "+train.numAttributes());
					double c = Double.valueOf(split[0]);
					if ((int) (c - 1) != (int) train.instance(i).classValue() && (int) (c) != (int) train.instance(i).classValue() && (int) (c + 1) != (int) train.instance(i).classValue()) {
						System.out.println(" ERROR MISMATCH IN CLASS " + s + " from instance " + i + " ucr =" + (int) c + " mine =" + (int) train.instance(i).classValue());
						System.exit(0);
					}
					for (int j = 1; j < train.numAttributes(); j++) {
						double v = Double.valueOf(split[j]);
						newTrain.writeString(v + ",");
					}
					if (first <= 0)
						newTrain.writeString((int) train.instance(i).classValue() + "\n");
					else
						newTrain.writeString((int) (train.instance(i).classValue() + 1) + "\n");

				} catch (Exception e) {
					System.out.println("Error problem " + s + " instance =" + i + " length =" + split.length + " val =" + split[0]);
					System.exit(0);

				}
			}
			for (int i = 0; i < testSize; i++) {
				String line = testF.readLine();
				line = line.trim();
				String[] split = line.split("/s+");
				try {
					// System.out.println(split[0]+"First ="+split[1]+" last ="+split[split.length-1]+" length = "+split.length+" nos atts "+train.numAttributes());
					double c = Double.valueOf(split[0]);
					if ((int) (c - 1) != (int) test.instance(i).classValue() && (int) (c) != (int) test.instance(i).classValue() && (int) (c + 1) != (int) test.instance(i).classValue()) {
						System.out.println(" ERROR MISMATCH IN CLASS " + s + " from instance " + i + " ucr =" + (int) c + " mine =" + (int) test.instance(i).classValue());
						System.exit(0);
					}
					for (int j = 1; j < test.numAttributes(); j++) {
						double v = Double.valueOf(split[j]);
						newTest.writeString(v + ",");
					}
					if (first <= 0)
						newTest.writeString((int) test.instance(i).classValue() + "\n");
					else
						newTest.writeString((int) (test.instance(i).classValue() + 1) + "\n");

				} catch (Exception e) {
					System.out.println("Error problem " + s + " instance =" + i + " length =" + split.length + " val =" + split[0]);
					System.exit(0);

				}
			}

		}

	}

	public static void listNotNormalisedList() throws Exception {
		TreeSet<String> notNormed = new TreeSet<>();
		DecimalFormat df = new DecimalFormat("###.######");
		for (String s : fileNames) {
			// Load test train
			Instances train = ClassifierTools.loadData(problemPath + s + "/" + s + "_TRAIN");
			Instances test = ClassifierTools.loadData(problemPath + s + "/" + s + "_TEST");
			// Find summary
			SummaryStats ss = new SummaryStats();
			train = ss.process(train);
			test = ss.process(test);
			int i = 1;
			for (Instance ins : train) {
				double stdev = ins.value(1) * ins.value(1);
				// stdev*=train.numAttributes()-1/(train.numAttributes()-2);
				if (Math.abs(ins.value(0)) > 0.01 || Math.abs(1 - stdev) > 0.01) {
					System.out.println(" Not normalised train series =" + s + " index " + i + " mean = " + df.format(ins.value(0)) + " var =" + df.format(stdev));
					notNormed.add(s);
					break;
				}
			}
			for (Instance ins : test) {
				double stdev = ins.value(1) * ins.value(1);
				// stdev*=train.numAttributes()-1/(train.numAttributes()-2);
				if (Math.abs(ins.value(0)) > 0.01 || Math.abs(1 - stdev) > 0.01) {
					System.out.println(" Not normalised test series =" + s + " index " + i + " mean = " + df.format(ins.value(0)) + " var =" + df.format(stdev));
					notNormed.add(s);
					break;
				}
			}
		}
		System.out.print("String[] notNormalised={");
		for (String s : notNormed)
			System.out.print("\"" + s + "\",");
		System.out.println("}");
		System.out.println("TOTAL NOT NORMED =" + notNormed.size());

	}

	public static void dataDescription(String[] fileNames) {
		// Produce summary descriptions
		// dropboxPath=uciPath;
		OutFile f = new OutFile(problemPath + "DataDimensions.csv");
		MetaData[] all = new MetaData[fileNames.length];
		TreeSet<String> nm = new TreeSet<>();
		nm.addAll(Arrays.asList(notNormalised));

		try {
			for (int i = 0; i < fileNames.length; i++) {
				Instances test = ClassifierTools.loadData(problemPath + fileNames[i] + "/" + fileNames[i] + "_TEST");
				Instances train = ClassifierTools.loadData(problemPath + fileNames[i] + "/" + fileNames[i] + "_TRAIN");
				Instances allData = new Instances(test);
				for (int j = 0; j < train.numInstances(); j++)
					allData.add(train.instance(j));
				// allData.randomize(new Random());
				// OutFile combo=new
				// OutFile(problemPath+fileNames[i]+"/"+fileNames[i]+".arff");
				// combo.writeString(allData.toString());
				boolean normalised = true;
				if (nm.contains(fileNames[i]))
					normalised = false;
				int[] classCounts = new int[allData.numClasses() * 2];
				for (Instance ins : train)
					classCounts[(int) (ins.classValue())]++;
				for (Instance ins : test)
					classCounts[allData.numClasses() + (int) (ins.classValue())]++;
				all[i] = new MetaData(fileNames[i], train.numInstances(), test.numInstances(), test.numAttributes() - 1, test.numClasses(), classCounts, normalised);
				f.writeLine(all[i].toString());
			}
		} catch (Exception e) {
			System.out.println(" ERRROR" + e);
		}
		Arrays.sort(all);
		f = new OutFile(dropboxPath + "DataDimensionsBySeriesLength.csv");
		for (MetaData m : all)
			f.writeLine(m.toString());
		Arrays.sort(all, new MetaData.CompareByTrain());
		f = new OutFile(dropboxPath + "DataDimensionsByTrainSize.csv");
		for (MetaData m : all)
			f.writeLine(m.toString());
		Arrays.sort(all, new MetaData.CompareByClasses());
		f = new OutFile(dropboxPath + "DataDimensionsByNosClasses.csv");
		for (MetaData m : all)
			f.writeLine(m.toString());
		Arrays.sort(all, new MetaData.CompareByTotalSize());
		f = new OutFile(dropboxPath + "DataDimensionsByTotalSize.csv");
		for (MetaData m : all)
			f.writeLine(m.toString());

	}

	public static void main(String[] args) throws Exception {

		dataDescription(fileNames);
		System.exit(0);
		listNotNormalisedList();
		processUCRData();
		Instances train = ClassifierTools.loadData(problemPath + "wafer/wafer_TRAIN");
		Instances test = ClassifierTools.loadData(problemPath + "wafer/wafer_TEST");

		for (String s : fileNames) {
			Instances all = ClassifierTools.loadData(problemPath + s + "/" + s);
			train = ClassifierTools.loadData(problemPath + s + "/" + s + "_TRAIN");
			test = ClassifierTools.loadData(problemPath + s + "/" + s + "_TEST");
			System.out.println(s + " load ok ");
		}
		System.exit(0);

		// dataDescription(uciFileNames);
		/*
		 * for(String s:uciFileNames){ Instances train
		 * =ClassifierTools.loadData(uciPath+s+"\\"+s+"-train"); Instances test
		 * =ClassifierTools.loadData(uciPath+s+"\\"+s+"-test");
		 * System.out.println(s); }
		 */
	}

	public static class MetaData implements Comparable<MetaData> {
		String fileName;
		int trainSetSize;
		int testSetSize;
		int seriesLength;
		int nosClasses;
		int[] classDistribution;
		boolean normalised = true;

		public MetaData(String n, int t1, int t2, int s, int c, int[] dist, boolean norm) {
			fileName = n;
			trainSetSize = t1;
			testSetSize = t2;
			seriesLength = s;
			nosClasses = c;
			classDistribution = dist;
			normalised = norm;
		}

		@Override
		public String toString() {
			String str = fileName + "," + trainSetSize + "," + testSetSize + "," + seriesLength + "," + nosClasses + "," + normalised;
			for (int i : classDistribution)
				str += "," + i;
			return str;
		}

		@Override
		public int compareTo(MetaData o) {
			return seriesLength - o.seriesLength;
		}

		public static class CompareByTrain implements Comparator<MetaData> {
			@Override
			public int compare(MetaData a, MetaData b) {
				return a.trainSetSize - b.trainSetSize;
			}
		}

		public static class CompareByTrainSetSize implements Comparator<MetaData> {
			@Override
			public int compare(MetaData a, MetaData b) {
				return a.trainSetSize - b.trainSetSize;
			}
		}

		public static class CompareByClasses implements Comparator<MetaData> {
			@Override
			public int compare(MetaData a, MetaData b) {
				return a.nosClasses - b.nosClasses;
			}
		}

		public static class CompareByTotalSize implements Comparator<MetaData> {
			@Override
			public int compare(MetaData a, MetaData b) {
				return a.seriesLength * a.trainSetSize - b.seriesLength * b.trainSetSize;
			}
		}
	}

}
