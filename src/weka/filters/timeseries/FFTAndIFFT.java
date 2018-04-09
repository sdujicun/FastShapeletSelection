package weka.filters.timeseries;

/* Performs a FFT of the data set. NOTE:
 * 1. If algorithm type is set to DFT, then this will only perform a FFT if the series is length power of 2.
 * otherwise it will perform the order m^2 DFT.
 * 2. If algorithm type is set to FFT, then, if the length is not a powerr of 2, it either truncates or pads 
 * (determined by the variable pad) with the mean the each series (i.e. each Instance) 
 * so that the new length is power of 2 by flag pad (default true)
 * 2. By default, stoAlgorithmTyperes the complex terms in order, so att 1 is real coeff of Fourier term 1, attribute 2 the imag etc
 * 3. Only stores the first half of the Fourier terms (which are duplicates of the second half)
 * 
 * Note that the series does store the first fourier term (series mean) and the 
 * imaginary part will always be zero
 */
import weka.core.*;
import weka.filters.SimpleBatchFilter;

public class FFTAndIFFT extends FFT {

	
	protected Instances determineOutputFormatWithN(Instances inputFormat) throws Exception {
		// Check all attributes are real valued, otherwise throw exception
		for (int i = 0; i < inputFormat.numAttributes(); i++)
			if (inputFormat.classIndex() != i)
				if (!inputFormat.attribute(i).isNumeric())
					throw new Exception("Non numeric attribute not allowed in FFT");

		int length = findLength(inputFormat);
		FastVector atts = new FastVector();
		String name;
		for (int i = 0; i < length; i++) {
			name = "FFTAndFFI_" + i;
			atts.addElement(new Attribute(name));
		}
		if (inputFormat.classIndex() >= 0) { // Classification set, set class
			// Get the class values as a fast vector
			Attribute target = inputFormat.attribute(inputFormat.classIndex());

			FastVector vals = new FastVector(target.numValues());
			for (int i = 0; i < target.numValues(); i++)
				vals.addElement(target.value(i));
			atts.addElement(new Attribute(inputFormat.attribute(inputFormat.classIndex()).name(), vals));
		}
		Instances result = new Instances("FFTAndFFI_" + inputFormat.relationName(), atts, inputFormat.numInstances());
		if (inputFormat.classIndex() >= 0)
			result.setClassIndex(result.numAttributes() - 1);
		return result;
	}

	
      
	//FFT and IFFT
	public Instances processWithN(Instances instances,double n) throws Exception {

		Instances output = determineOutputFormatWithN(instances);

		int originalLength = instances.numAttributes();
		if (instances.classIndex() >= 0) {
			originalLength--;
		}
		
		int fullLength = findLength(instances);
		int partLength=(int)(fullLength*n);
		if(partLength==0){
			throw new Exception("n is too large");
		}
		// For each data, first extract the relevant data
		// Note the transform will be at least twice as long as the original
		// Length is the number of COMPLEX terms, which is HALF the length of
		// the original series.

		for (int i = 0; i < instances.numInstances(); i++) {

			
			Complex[] c = new Complex[fullLength];
			int count = 0;
			double seriesTotal = 0;
			for (int j = 0; j < originalLength && count < c.length; j++) { 
				if (instances.classIndex() != j) {
					c[count] = new Complex(instances.instance(i).value(j), 0.0);
					seriesTotal += instances.instance(i).value(j);
					count++;
				}
			}
			// Add any Padding required
			double mean = seriesTotal / count;
			while (count < c.length)
				c[count++] = new Complex(mean, 0);
			// 2. Find FFT/DFT of series.
			if (algo == AlgorithmType.FFT)
				fft(c, c.length);
			else
				c = dft(c);
			// Extract out the terms and set the attributes.

			double[] returndata=ifft(c,partLength);
						
			Instance inst = new DenseInstance(fullLength + 1);
			for (int j = 0; j < fullLength; j++) {
				inst.setValue(j, returndata[j]);
				
			}
			
			if (instances.classIndex() >= 0)
				inst.setValue(output.classIndex(), instances.instance(i).classValue());

			output.add(inst);
		}
		return output;
	}

	
	public double[] ifft(Complex[] inData, int partLength) throws Exception {
		if(partLength>inData.length){
			throw new Exception("n is too large");
		}
		int N = inData.length;
		double[] returndata=new double[N];
		for (int i = 0; i < N; i++) {
			returndata[i]=0;
			for (int j = 0; j < partLength; j++) {
				//变换后数据不需要虚部
				returndata[i] += inData[j].real * Math.cos(2 * Math.PI * i * j / N) / N - inData[j].imag * Math.sin(2 * Math.PI * i * j / N) / N;
			}
		}
		return returndata;
	}
	
}
