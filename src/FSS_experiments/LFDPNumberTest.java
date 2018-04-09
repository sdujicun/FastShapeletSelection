package FSS_experiments;

import java.io.File;
import java.util.ArrayList;

import utilities.ClassifierTools;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.lazy.kNN;
import weka.classifiers.meta.RotationForest;
import weka.classifiers.meta.timeseriesensembles.WeightedEnsemble;
import weka.classifiers.trees.EnhancedRandomForest;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.shapelet.QualityMeasures;
import weka.core.shapelet.Shapelet;
import weka.filters.timeseries.shapelet_transforms.FullShapeletTransform;
import weka.filters.timeseries.shapelet_transforms.classValue.BinarisedClassValue;
import weka.filters.timeseries.shapelet_transforms.subclass.ShapeletTransformWithSubclassSampleAndLFDP;
import weka.filters.timeseries.shapelet_transforms.subsequenceDist.ImprovedOnlineSubSeqDistance;
import development.DataSets;

public class LFDPNumberTest {
	public static void main(String[] args) throws Exception {
		String[] problems={"Coffee","MedicalImages","MoteStrain","Trace" };
		
		System.out.println(problems.length);
		for (int i = 0; i < problems.length; i++) {
			System.out.println(problems[i]);
			for(int j=1;j<=50;j++){
				double LFDPRate=1.0*j/100;
				System.out.print(LFDPRate+"\t");
				LFDPNumberTest(problems[i],LFDPRate);
				System.out.println();
			}
		}
	}

	public static void LFDPNumberTest(String problem,double LFDPRate) throws Exception {

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
		transform.setLFDPrate(LFDPRate);		
		
		long d1, d2;
		d1 = System.nanoTime();
		Instances tranTrain = transform.process(train);		
		d2 = System.nanoTime();
		System.out.print((d2 - d1) * 0.000000001 + "\t");
		
		Instances tranTest = transform.process(test);

		RotationForest rf = new RotationForest();
		rf.setNumIterations(50);
		rf.buildClassifier(tranTrain);
		double accuracy = ClassifierTools.accuracy(tranTest, rf);
		System.out.print(accuracy + "\t");

	}

}
