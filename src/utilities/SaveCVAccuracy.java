package utilities;

import weka.core.Instances;

/**
 *
 * @author ajb
 */
public interface SaveCVAccuracy {
	public static int LENGTH_TRESH = 300;
	public static int INSTANCES_TRESH = 500;

	void setCVPath(String train);

	String getParameters();

	default int setNumberOfFolds(Instances data) {
		if (data.numInstances() < 100)
			return data.numInstances();
		if (data.numInstances() < INSTANCES_TRESH && data.numAttributes() - 1 < LENGTH_TRESH)
			return 100;
		return 10;
	}

}
