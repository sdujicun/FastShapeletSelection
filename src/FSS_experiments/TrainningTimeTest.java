package FSS_experiments;

import java.io.File;
import java.util.ArrayList;

import tsc_algorithms.FastShapelets;
import tsc_algorithms.FastShapeletsWithSFA;
import tsc_algorithms.LearnShapelets;
import utilities.ClassifierTools;
import weka.classifiers.lazy.kNN;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.shapelet.QualityMeasures;
import weka.core.shapelet.Shapelet;
import weka.filters.timeseries.shapelet_transforms.FullShapeletTransform;
import weka.filters.timeseries.shapelet_transforms.classValue.BinarisedClassValue;
import weka.filters.timeseries.shapelet_transforms.subclass.ShapeletTransformWithSubclassSampleAndLFDP;
import weka.filters.timeseries.shapelet_transforms.subsequenceDist.ImprovedOnlineSubSeqDistance;
import development.DataSets;

public class TrainningTimeTest {
	public static void main(String[] args) throws Exception {

		String[] problems={"ChlorineConcentration","Coffee","DiatomSizeReduction","ECG200","ECGFiveDays","MoteStrain","Lightning7","Symbols","SyntheticControl","Trace"};
			

		for (int i = 0; i < problems.length; i++) {
			System.out.println(problems[i]);
			trainTimeForLS(problems[i]);
			trainTimeForST(problems[i]);
			trainTimeForFSS(problems[i]);
			trainTimeForFS(problems[i]);
			System.out.println();			
		}
		
	}

	public static void trainTimeForFSS(String problem) throws Exception {
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

		long d1, d2;
		d1 = System.nanoTime();
		Instances tranTrain = transform.process(train);
		
		d2 = System.nanoTime();
		System.out.print((d2 - d1) * 0.000000001 + "\t");
		
	}
	public static void trainTimeForST(String problem) throws Exception {
		final String resampleLocation = DataSets.problemPath;
		final String dataset = problem;
		final String filePath = resampleLocation + File.separator + dataset + File.separator + dataset;

		Instances test, train;
		test = utilities.ClassifierTools.loadData(filePath + "_TEST");
		train = utilities.ClassifierTools.loadData(filePath + "_TRAIN");
		FullShapeletTransform transform = new FullShapeletTransform();
		transform.setRoundRobin(true);
		// construct shapelet classifiers.
		transform.setClassValue(new BinarisedClassValue());
		transform.setSubSeqDistance(new ImprovedOnlineSubSeqDistance());
		transform.setShapeletMinAndMax(3, train.numAttributes() - 1);
		transform.useCandidatePruning();
		transform.setNumberOfShapelets(train.numInstances() / 2);
		transform.setQualityMeasure(QualityMeasures.ShapeletQualityChoice.INFORMATION_GAIN);
		long d1 = System.nanoTime();
		Instances tranTrain = transform.process(train);
		//Instances tranTest = transform.process(test);
		ArrayList<Shapelet> sh = transform.getShapelets();
		long d2 = System.nanoTime();
		System.out.print((d2 - d1) * 0.000000001 + "\t");
		
	}
	public static void trainTimeForFS(String problem) throws Exception {
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

		long d1, d2;
		d1 = System.nanoTime();
		Instances tranTrain = transform.process(train);
		
		d2 = System.nanoTime();
		System.out.print((d2 - d1) * 0.000000001 + "\t");
		
	}
	public static void trainTimeForLS(String problem) throws Exception {

		final String resampleLocation = DataSets.problemPath;
		final String dataset = problem;
		final String filePath = resampleLocation + File.separator + dataset + File.separator + dataset;

		Instances test, train;
		test = utilities.ClassifierTools.loadData(filePath + "_TEST");
		train = utilities.ClassifierTools.loadData(filePath + "_TRAIN");
		     

        LearnShapelets ls = new LearnShapelets();
        ls.setSeed(0);
        ls.setParamSearch(true);
        
		long d1, d2;
		d1 = System.nanoTime();
		ls.buildClassifier(train);
		d2 = System.nanoTime();
		System.out.print((d2 - d1) * 0.000000001 + "\t");
	}
	
}
