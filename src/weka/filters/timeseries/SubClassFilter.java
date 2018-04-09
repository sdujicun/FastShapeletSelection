package weka.filters.timeseries;

import java.util.List;

import weka.core.*;
import weka.filters.SimpleBatchFilter;
import weka.filters.timeseries.shapelet_transforms.subclass.SubclassSample;

public class SubClassFilter extends SimpleBatchFilter {

	@Override
	protected Instances determineOutputFormat(Instances inputFormat) throws Exception {
		

		// Set up instances size and format.

		FastVector atts = new FastVector();
		String name;
		for (int i = 0; i < inputFormat.numAttributes()-1; i++) {
			name = "CubClass_" + i;
			atts.addElement(new Attribute(name));
		}
		
		Attribute target = inputFormat.attribute(inputFormat.classIndex());

		FastVector vals = new FastVector(target.numValues());
		for (int i = 0; i < target.numValues(); i++) {
			vals.addElement(target.value(i));
		}
		atts.addElement(new Attribute(inputFormat.attribute(inputFormat.classIndex()).name(), vals));
		Instances result = new Instances("att" + inputFormat.relationName(), atts, inputFormat.numInstances());
		if (inputFormat.classIndex() >= 0) {
			result.setClassIndex(result.numAttributes() - 1);
		}

		return result;
		
		
	}

	@Override
	public String globalInfo() {

		return null;
	}

	@Override
	public Instances process(Instances instances) throws Exception {

		Instances output = determineOutputFormat(instances);

		Instance newInst, oldInst;
		List<Integer> list = new SubclassSample(). subClassSplitting(instances);
		int n = instances.numAttributes() - 1;
		for (int i = 0; i < list.size(); i++) {
			int j=list.get(i);
			oldInst = instances.instance(j);
			newInst = new DenseInstance(oldInst.numAttributes());
			for (int k = 0; k < n; k++) {
				newInst.setValue(k, oldInst.value(k));
			}

			newInst.setValue(output.classIndex(), instances.instance(j).classValue());
			output.add(newInst);
		}

		return output;
	}
	
	public Instances process(Instances instances, Instances add, int number) throws Exception {

		Instances output = determineOutputFormat(instances);

		Instance newInst, oldInst;
		int n = instances.numAttributes() - 1;
		for (int i = 0; i < instances.size(); i++) {
			oldInst = instances.instance(i);
			newInst = new DenseInstance(oldInst.numAttributes());
			for (int k = 0; k < n; k++) {
				newInst.setValue(k, oldInst.value(k));
			}

			newInst.setValue(output.classIndex(), instances.instance(i).classValue());
			output.add(newInst);
		}
		for (int i = 0; i < number; i++) {
			oldInst = add.instance(i);
			newInst = new DenseInstance(oldInst.numAttributes());
			for (int k = 0; k < n; k++) {
				newInst.setValue(k, oldInst.value(k));
			}

			newInst.setValue(output.classIndex(), add.instance(i).classValue());
			output.add(newInst);
		}

		return output;
	}
}