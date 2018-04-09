package tsc_algorithms.COTE_FSS;

import java.util.Date;

import development.DataSets;
import fileIO.OutFile;
import tsc_algorithms.ElasticEnsemble;
import utilities.ClassifierTools;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.meta.timeseriesensembles.WeightedEnsemble;
import weka.core.*;
import weka.core.shapelet.QualityMeasures;
import weka.filters.timeseries.ACF;
import weka.filters.timeseries.PowerSpectrum;
import weka.filters.timeseries.shapelet_transforms.subclass.ShapeletTransformWithSubclassSampleAndLFDP;
import weka.filters.timeseries.shapelet_transforms.subsequenceDist.OnlineSubSeqDistance;

public class COTE_FSS extends AbstractClassifier {
	WeightedEnsemble change;
	WeightedEnsemble powerSpectrum;
	WeightedEnsemble shapelet;
	ElasticEnsemble ee;
	int nosTransforms = 4;
	Instances train;
	ShapeletTransformWithSubclassSampleAndLFDP shapeletT;
	double[] changeCVAccs;
	double[] psCVAccs;
	double[] shapeletCVAccs;
	double[] elasticCVAccs;
	WeightedEnsemble.WeightType weightType = WeightedEnsemble.WeightType.PROPORTIONAL;

	public void setWeightType(String s) {
		String str = s.toUpperCase();
		switch (str) {
		case "EQUAL":
		case "EQ":
		case "E":
			weightType = WeightedEnsemble.WeightType.EQUAL;
			break;
		case "BEST":
		case "B":
			weightType = WeightedEnsemble.WeightType.BEST;
			break;
		case "PROPORTIONAL":
		case "PROP":
		case "P":
			weightType = WeightedEnsemble.WeightType.PROPORTIONAL;
			break;
		/*
		 * NOT IMPLEMENTED YET IN THIS VERSION case "SIGNIFICANT_BINOMIAL": case
		 * "SIGB": case "SIG": case "S": w=WeightType.SIGNIFICANT_BINOMIAL;
		 * break; case "SIGNIFICANT_MCNEMAR": case "SIGM": case "SM":
		 * w=WeightType.SIGNIFICANT_MCNEMAR; break;
		 */
		default:
			throw new UnsupportedOperationException("Weighting method " + str + " not supported yet.");
		}
	}

	public void setWeightType(WeightedEnsemble.WeightType w) {
		weightType = w;
	}

	public double[] getCVAccs() {
		double[] cv = new double[changeCVAccs.length + psCVAccs.length + shapeletCVAccs.length + elasticCVAccs.length];
		System.arraycopy(changeCVAccs, 0, cv, 0, changeCVAccs.length);
		System.arraycopy(psCVAccs, 0, cv, changeCVAccs.length, psCVAccs.length);
		System.arraycopy(shapeletCVAccs, 0, cv, changeCVAccs.length + psCVAccs.length, shapeletCVAccs.length);
		System.arraycopy(elasticCVAccs, 0, cv, changeCVAccs.length + psCVAccs.length + shapeletCVAccs.length, elasticCVAccs.length);
		return cv;
	}

	@Override
	public void buildClassifier(Instances data) throws Exception {
		train = data;
		// ACF
		Instances changeTrain = ACF.formChangeCombo(train);
		// Power Spectrum
		PowerSpectrum ps = new PowerSpectrum();
		Instances psTrain = ps.process(train);
		shapeletT = new ShapeletTransformWithSubclassSampleAndLFDP();
		shapeletT.setSubSeqDistance(new OnlineSubSeqDistance());
		shapeletT.setNumberOfShapelets(train.numInstances() * 10);
		shapeletT.setDebug(false);
		shapeletT.supressOutput();
		shapeletT.turnOffLog();
		shapeletT.setQualityMeasure(QualityMeasures.ShapeletQualityChoice.F_STAT);
		Instances shapeletTrain = shapeletT.process(train);
		// Build all the classifiers
		change = new WeightedEnsemble();
		change.setWeightType(WeightedEnsemble.WeightType.PROPORTIONAL);
		change.buildClassifier(changeTrain);

		powerSpectrum = new WeightedEnsemble();
		powerSpectrum.setWeightType(WeightedEnsemble.WeightType.PROPORTIONAL);
		powerSpectrum.buildClassifier(psTrain);

		shapelet = new WeightedEnsemble();
		shapelet.setWeightType(WeightedEnsemble.WeightType.PROPORTIONAL);
		shapelet.buildClassifier(shapeletTrain);

		// Elastic ensemble
		ee = new ElasticEnsemble();
		ee.turnAllClassifiersOn();
		ee.setEnsembleType(ElasticEnsemble.EnsembleType.Prop);
		ee.buildClassifier(train);

		// We need the training CV for all the ensembles to weight the
		// classification.
		// Should probably just get the weights, that would be more sensible,
		// but
		// I need to ensure they are calculated in the same way in the EE
		changeCVAccs = change.getCVAccs();
		psCVAccs = powerSpectrum.getCVAccs();
		shapeletCVAccs = shapelet.getCVAccs();
		elasticCVAccs = ee.getCVAccs();
		// EE stores the accuracies range 0...100. I dont want to change this in
		// EE
		// because there may be side effects.
		for (int j = 0; j < elasticCVAccs.length; j++)
			elasticCVAccs[j] /= 100.0;
	}

	/*
	 * We store information on the last instance classified with
	 * classifyInstance this allows us to extract performance data on the
	 * individual classifiers if we wish
	 */
	double[] transformPreds = new double[nosTransforms];
	double[] allClassifierPreds;
	double cotePred;

	@Override
	public double classifyInstance(Instance data) throws Exception {
		// Form transforms. Annoyingly, the filters work with Instances rather
		// than with
		// a single instance
		Instances test = new Instances(train, 0);
		test.add(data);
		Instances changeTest = ACF.formChangeCombo(test);
		Instances psTest = new PowerSpectrum().process(test);
		Instances shapeletTest = shapeletT.process(test);
		transformPreds[0] = change.classifyInstance(changeTest.firstInstance());
		transformPreds[1] = powerSpectrum.classifyInstance(psTest.firstInstance());
		transformPreds[2] = shapelet.classifyInstance(shapeletTest.firstInstance());
		transformPreds[3] = ee.classifyInstance(data);

		// Recover component predictions
		double[] allVotes = new double[data.numClasses()];
		double[] changePreds = change.getPredictions();
		// Aggregate all votes: flat-COTE structure only.
		// Needs encapsulating, generalising and tidying up
		for (int j = 0; j < changePreds.length; j++)
			allVotes[(int) changePreds[j]] += changeCVAccs[j];
		double[] psPreds = powerSpectrum.getPredictions();
		for (int j = 0; j < psPreds.length; j++)
			allVotes[(int) psPreds[j]] += psCVAccs[j];

		double[] shapeletPreds = shapelet.getPredictions();
		for (int j = 0; j < shapeletPreds.length; j++)
			allVotes[(int) shapeletPreds[j]] += shapeletCVAccs[j];
		double[] eePreds = ee.getPredictions();
		for (int j = 0; j < eePreds.length; j++)
			allVotes[(int) eePreds[j]] += elasticCVAccs[j];
		// Find COTE prediction
		int coteVote = 0;
		for (int j = 1; j < allVotes.length; j++) {
			if (allVotes[coteVote] < allVotes[j])
				coteVote = j;
		}
		cotePred = coteVote;
		return cotePred;
	}

	/*
	 * @Override public double[] distributionForInstance(Instance data){ return
	 * null; }
	 */
	// Basic debug tests with small problems
	public static void main(String[] args) {
		String path = DataSets.problemPath;
		
		String[] problems={"Adiac","Beef","ChlorineConcentration","Coffee","DiatomSizeReduction","ECGFiveDays", "FaceFour","GunPoint", "ItalyPowerDemand","Lightning7", "MedicalImages", "MoteStrain", "SonyAIBORobotSurface1","SonyAIBORobotSurface2","Symbols","SyntheticControl","Trace", "TwoLeadECG" };
		System.out.println(problems.length);
		for (int i = 0; i < problems.length; i++) {
			System.out.print(problems[i] + "\t");
			System.out.print(new Date()+"\t");
			System.out.println (trainTestExample(path, problems[i]));
		}


	}

	public static double trainTestExample(String path, String problem) {
		COTE_FSS cote = new COTE_FSS();
		Instances train = ClassifierTools.loadData(path + problem + "\\" + problem + "_TRAIN");
		Instances test = ClassifierTools.loadData(path + problem + "\\" + problem + "_TEST");
		return ClassifierTools.singleTrainTestSplitAccuracy(cote, train, test);

	}

}
