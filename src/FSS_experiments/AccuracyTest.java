package FSS_experiments;

import java.io.File;
import java.io.FileWriter;
import java.io.Writer;

import utilities.ClassifierTools;
import weka.classifiers.meta.timeseriesensembles.WeightedEnsemble;
import weka.core.Instances;
import weka.core.shapelet.QualityMeasures;
import weka.filters.timeseries.shapelet_transforms.classValue.BinarisedClassValue;
import weka.filters.timeseries.shapelet_transforms.subclass.ShapeletTransformWithSubclassSampleAndLFDP;
import weka.filters.timeseries.shapelet_transforms.subsequenceDist.ImprovedOnlineSubSeqDistance;
import development.DataSets;

public class AccuracyTest {
	public static void main(String[] args) throws Exception {
		
		//String[] problems=DataSets.DSUsed;	
		String[] problems= { 
			"Adiac", // 390,391,176,37
			"Beef", // 30,30,470,5
			"ChlorineConcentration",
			"Coffee", // 28,28,286,2
			"DiatomSizeReduction", // 16,306,345,4
			"ItalyPowerDemand", // 67,1029,24,2
			"Lightning7", // 70,73,319,7
			"MedicalImages", // 381,760,99,10
			"MoteStrain", // 20,1252,84,2
			"Symbols", // 25,995,398,6
			"Trace", // 100,100,275,4
			"TwoLeadECG", // 23,1139,82,2
	};
		System.out.println(problems.length);
		for (int i = 0; i < problems.length; i++) {
		
			System.out.print(problems[i] + "\t");
			accuracyForFSS(problems[i]);
			System.out.println();

		}
	}

	public static void accuracyForFSS(String problem) throws Exception {

		String filename = "accuracy.txt";
		File f = new File("./" + File.separator + filename);
		Writer out = null;
		out = new FileWriter(f, true);
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

		Instances tranTrain = transform.process(train);
		Instances tranTest = transform.process(test);

		double accuracy;
		WeightedEnsemble we = new WeightedEnsemble();
		we.buildClassifier(tranTrain);
		accuracy = ClassifierTools.accuracy(tranTest, we);

		
		out.write(problem + "\t" + accuracy + "\t\r\n");
		out.close();
		System.out.print(accuracy + "\t");

	}

}
