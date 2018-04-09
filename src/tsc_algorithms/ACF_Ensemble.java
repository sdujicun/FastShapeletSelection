package tsc_algorithms;

import bakeOffExperiments.Experiments;
import fileIO.OutFile;
import java.util.ArrayList;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.meta.RotationForest;
import weka.classifiers.meta.timeseriesensembles.WeightedEnsemble;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.timeseries.ACF;

/**
 *
 easiest way to generate these is to deconstruct the weighted ensemble.
 */
public class ACF_Ensemble extends AbstractClassifier implements SaveableEnsemble {
	private Classifier baseClassifier;
	private Instances format;
	private ClassifierType c = ClassifierType.RandF;
	private boolean saveResults = false;
	private String trainCV = "";
	private String testPredictions = "";
	private boolean doTransform = true;
	int[] constantFeatures;

	protected void saveResults(boolean s) {
		saveResults = s;
	}

	public void saveResults(String tr, String te) {
		saveResults(true);
		trainCV = tr;
		testPredictions = te;
	}

	public enum ClassifierType {
		RandF("RandF", 500), RotF("RotF", 50), WeightedEnsemble("WE", 8);
		String type;
		int numBaseClassifiers;

		ClassifierType(String s, int x) {
			type = s;
			numBaseClassifiers = x;
		}

		Classifier createClassifier() {
			switch (type) {
			case "RandF":
				RandomForest randf = new RandomForest();
				randf.setNumTrees(numBaseClassifiers);
				return randf;
			case "RotF":
				RotationForest rotf = new RotationForest();
				rotf.setNumIterations(numBaseClassifiers);
				return rotf;
			case "WE":
				WeightedEnsemble we = new WeightedEnsemble();
				we.setWeightType("prop");
				return we;
			default:
				RandomForest c = new RandomForest();
				c.setNumTrees(numBaseClassifiers);
				return c;
			}
		}
	}

	public void setClassifierType(String s) {
		s = s.toLowerCase();
		switch (s) {
		case "randf":
		case "randomforest":
		case "randomf":
			c = ClassifierType.RandF;
			break;
		case "rotf":
		case "rotationforest":
		case "rotationf":
			c = ClassifierType.RotF;
			break;
		case "weightedensemble":
		case "we":
		case "wens":
			c = ClassifierType.WeightedEnsemble;
			break;

		}
	}

	public void doACFTransform(boolean b) {
		doTransform = b;
	}

	@Override
	public void buildClassifier(Instances data) throws Exception {
		baseClassifier = c.createClassifier();
		/*
		 * This flag allows the perfomance of the transform outside of the
		 * classifier, which speeds up the cross validation and is ok because
		 * the transform is unsupervised and indendent of other cases
		 */

		if (doTransform)
			format = ACF.formChangeCombo(data);
		else
			format = data;
		constantFeatures = InstanceTools.removeConstantTrainAttributes(format);
		if (saveResults && c == ClassifierType.WeightedEnsemble) {
			// Set up the file space here
			((WeightedEnsemble) baseClassifier).saveTrainCV(trainCV);
			((WeightedEnsemble) baseClassifier).saveTestPreds(testPredictions);
		}
		baseClassifier.buildClassifier(format);

		// Record original format, empty of instances
		format = new Instances(data, 0);
	}

	@Override
	public double classifyInstance(Instance ins) throws Exception {
		//
		format.add(ins); // Should match!
		Instances temp;
		if (doTransform)
			temp = ACF.formChangeCombo(format);
		else
			temp = format;
		// Delete constants
		for (int del : constantFeatures)
			temp.deleteAttributeAt(del);
		Instance trans = temp.get(0);
		format.remove(0);
		return baseClassifier.classifyInstance(trans);
	}

	public static void main(String[] args) {
		Instances train = ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\ItalyPowerDemand\\ItalyPowerDemand_TRAIN");
		Instances test = ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\TSC Problems\\ItalyPowerDemand\\ItalyPowerDemand_TEST");
		String trainS = "C:\\Users\\ajb\\Dropbox\\TSC Problems\\ItalyPowerDemand\\TestPreds.csv";
		String testS = "C:\\Users\\ajb\\Dropbox\\TSC Problems\\ItalyPowerDemand\\TrainCV.csv";
		String preds = "C:\\Users\\ajb\\Dropbox\\TSC Problems\\ItalyPowerDemand";
		ACF_Ensemble acf = new ACF_Ensemble();
		acf.setClassifierType("WE");
		acf.saveResults(trainS, testS);
		double a = Experiments.singleSampleExperiment(train, test, acf, 0, preds);
	}
}
