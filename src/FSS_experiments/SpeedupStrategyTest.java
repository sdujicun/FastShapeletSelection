package FSS_experiments;

import java.io.File;
import java.util.ArrayList;

import tsc_algorithms.FastShapelets;
import tsc_algorithms.FastShapeletsWithSFA;
import utilities.ClassifierTools;
import weka.classifiers.lazy.kNN;
import weka.classifiers.meta.RotationForest;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.shapelet.QualityMeasures;
import weka.core.shapelet.Shapelet;
import weka.filters.timeseries.shapelet_transforms.FullShapeletTransform;
import weka.filters.timeseries.shapelet_transforms.ShapeletTransform;
import weka.filters.timeseries.shapelet_transforms.classValue.BinarisedClassValue;
import weka.filters.timeseries.shapelet_transforms.subclass.ShapeletTransformBasedOnLFDP;
import weka.filters.timeseries.shapelet_transforms.subclass.ShapeletTransformWithSubclassSample;
import weka.filters.timeseries.shapelet_transforms.subclass.ShapeletTransformWithSubclassSampleAndLFDP;
import weka.filters.timeseries.shapelet_transforms.subsequenceDist.ImprovedOnlineSubSeqDistance;
import development.DataSets;

public class SpeedupStrategyTest {
	public static void main(String[] args) throws Exception {

		String[] problems = { "ChlorineConcentration", "Coffee", "DiatomSizeReduction","ItalyPowerDemand",  "MedicalImages", "MoteStrain",  "SyntheticControl", "Trace" };		
		System.out.println(problems.length);
		for (int i = 0; i < problems.length; i++) {
			System.out.print(problems[i] + "\t");
			testFSS(problems[i]);
			testST_S(problems[i]);
			testST_F(problems[i]);
			testST(problems[i]);
			System.out.println();
			
		}
	}

	public static void testFSS(String problem) throws Exception {
		final String resampleLocation = DataSets.problemPath;
		final String dataset = problem;
		final String filePath = resampleLocation + File.separator + dataset + File.separator + dataset;
		
		Instances test, train;
		test = utilities.ClassifierTools.loadData(filePath + "_TEST");
		train = utilities.ClassifierTools.loadData(filePath + "_TRAIN");
		ShapeletTransformWithSubclassSampleAndLFDP transform = new ShapeletTransformWithSubclassSampleAndLFDP();
		transform.setRoundRobin(true);

		transform.setClassValue(new BinarisedClassValue());
		transform.setSubSeqDistance(new ImprovedOnlineSubSeqDistance());
		transform.useCandidatePruning();
		transform.setNumberOfShapelets(train.numInstances() / 2);
		transform.setQualityMeasure(QualityMeasures.ShapeletQualityChoice.INFORMATION_GAIN);

		//Shapelet Selection Time
		long d1, d2;
		d1 = System.nanoTime();
		Instances tranTrain = transform.process(train);
		d2 = System.nanoTime();
		System.out.print((d2 - d1) * 0.000000001 + "\t");
		
		
		Instances tranTest = transform.process(test);
		
		
		//Accuracy
		RotationForest rf = new RotationForest();
		rf.setNumIterations(50);
		rf.buildClassifier(tranTrain);
		double accuracy = ClassifierTools.accuracy(tranTest, rf);
		System.out.print(accuracy + "\t");


	}
	
	public static void testST_S(String problem) throws Exception {
		final String resampleLocation = DataSets.problemPath;
		final String dataset = problem;
		final String filePath = resampleLocation + File.separator + dataset + File.separator + dataset;

		
		
		Instances test, train;
		test = utilities.ClassifierTools.loadData(filePath + "_TEST");
		train = utilities.ClassifierTools.loadData(filePath + "_TRAIN");
		ShapeletTransformWithSubclassSample transform = new ShapeletTransformWithSubclassSample();
		transform.setRoundRobin(true);

		transform.setClassValue(new BinarisedClassValue());
		transform.setSubSeqDistance(new ImprovedOnlineSubSeqDistance());
		transform.useCandidatePruning();
		transform.setNumberOfShapelets(train.numInstances() / 2);
		transform.setQualityMeasure(QualityMeasures.ShapeletQualityChoice.INFORMATION_GAIN);

		//Shapelet Selection Time
		long d1, d2;
		d1 = System.nanoTime();
		Instances tranTrain = transform.process(train);
		d2 = System.nanoTime();
		System.out.print((d2 - d1) * 0.000000001 + "\t");
		
		
		Instances tranTest = transform.process(test);
		
		
		//Accuracy
		RotationForest rf = new RotationForest();
		rf.setNumIterations(50);
		rf.buildClassifier(tranTrain);
		double accuracy = ClassifierTools.accuracy(tranTest, rf);
		System.out.print(accuracy + "\t");
	}
	
	public static void testST_F(String problem) throws Exception {
		final String resampleLocation = DataSets.problemPath;
		final String dataset = problem;
		final String filePath = resampleLocation + File.separator + dataset + File.separator + dataset;

		
		
		Instances test, train;
		test = utilities.ClassifierTools.loadData(filePath + "_TEST");
		train = utilities.ClassifierTools.loadData(filePath + "_TRAIN");
		ShapeletTransformBasedOnLFDP transform = new ShapeletTransformBasedOnLFDP();
		transform.setRoundRobin(true);

		transform.setClassValue(new BinarisedClassValue());
		transform.setSubSeqDistance(new ImprovedOnlineSubSeqDistance());
		transform.useCandidatePruning();
		transform.setNumberOfShapelets(train.numInstances() / 2);
		transform.setQualityMeasure(QualityMeasures.ShapeletQualityChoice.INFORMATION_GAIN);

		//Shapelet Selection Time
		long d1, d2;
		d1 = System.nanoTime();
		Instances tranTrain = transform.process(train);
		d2 = System.nanoTime();
		System.out.print((d2 - d1) * 0.000000001 + "\t");
		
		
		Instances tranTest = transform.process(test);
		
		
		//Accuracy
		RotationForest rf = new RotationForest();
		rf.setNumIterations(50);
		rf.buildClassifier(tranTrain);
		double accuracy = ClassifierTools.accuracy(tranTest, rf);
		System.out.print(accuracy + "\t");


	}
	
	
	public static void testST(String problem) throws Exception {
		final String resampleLocation = DataSets.problemPath;
		final String dataset = problem;
		final String filePath = resampleLocation + File.separator + dataset + File.separator + dataset;

		
		
		Instances test, train;
		test = utilities.ClassifierTools.loadData(filePath + "_TEST");
		train = utilities.ClassifierTools.loadData(filePath + "_TRAIN");
		ShapeletTransform transform = new ShapeletTransform();
		transform.setShapeletMinAndMax(3, train.numAttributes() - 1);
		transform.setRoundRobin(true);

		transform.setClassValue(new BinarisedClassValue());
		transform.setSubSeqDistance(new ImprovedOnlineSubSeqDistance());
		transform.useCandidatePruning();
		transform.setNumberOfShapelets(train.numInstances() / 2);
		transform.setQualityMeasure(QualityMeasures.ShapeletQualityChoice.INFORMATION_GAIN);

		//Shapelet Selection Time
		long d1, d2;
		d1 = System.nanoTime();
		Instances tranTrain = transform.process(train);
		d2 = System.nanoTime();
		System.out.print((d2 - d1) * 0.000000001 + "\t");
				
		
		//The Accuracy of ST are take from Hills, J., Lines, J., Baranauskas, E., Mapp, J., Bagnall, A.: Classication of time series by shapelet transformation. Data Mining and Knowledge Discovery, 28(4),851-881 (2014)

	}
}
