package tsc_algorithms;

import java.util.ArrayList;
import java.util.Arrays;
import utilities.BitWord;

public class SFA {

	protected BitWord[][] SFAwords;	
	protected double[][] breakpoints;

	protected double inverseSqrtWindowSize;
	protected int windowSize;
	protected int wordLength;
	protected int alphabetSize;
	protected boolean norm;
	protected boolean numerosityReduction = true;
	protected static final long serialVersionUID = 1L;

	public SFA(int wordLength, int alphabetSize, int windowSize, boolean normalise) {
		this.wordLength = wordLength;
		this.alphabetSize = alphabetSize;
		this.windowSize = windowSize;
		this.inverseSqrtWindowSize = 1.0 / Math.sqrt(windowSize);
		this.norm = normalise;
	}	
	public SFA(int wordLength,  int windowSize, boolean normalise) {
		this.wordLength = wordLength;
		this.alphabetSize = 4;
		this.windowSize = windowSize;
		this.inverseSqrtWindowSize = 1.0 / Math.sqrt(windowSize);
		this.norm = normalise;
	}

	public SFA(SFA sfa, int wordLength) {
		this.wordLength = wordLength;

		this.windowSize = sfa.windowSize;
		this.inverseSqrtWindowSize = sfa.inverseSqrtWindowSize;
		this.alphabetSize = sfa.alphabetSize;
		this.norm = sfa.norm;
		this.numerosityReduction = sfa.numerosityReduction;
		// this.alphabet = sfa.alphabet;

		this.SFAwords = sfa.SFAwords;
		this.breakpoints = sfa.breakpoints;
	}

	

	public int getWindowSize() {
		return windowSize;
	}

	public int getWordLength() {
		return wordLength;
	}

	public int getAlphabetSize() {
		return alphabetSize;
	}

	public boolean isNorm() {
		return norm;
	}

	
	public int[] getParameters() {
		return new int[] { wordLength, alphabetSize, windowSize };
	}

	public void clean() {
		SFAwords = null;
	}

	protected double[][] slidingWindow(double[] data) {
		int numWindows = data.length - windowSize + 1;
		double[][] subSequences = new double[numWindows][windowSize];
		for (int windowStart = 0; windowStart < numWindows; ++windowStart) {
			System.arraycopy(data, windowStart, subSequences[windowStart], 0, windowSize);
		}

		return subSequences;
	}
	protected double[][] performDFT(double[][] windows) {
		double[][] dfts = new double[windows.length][wordLength];
		for (int i = 0; i < windows.length; ++i) {
			dfts[i] = DFT(windows[i]);
		}
		return dfts;
	}

	protected double stdDev(double[] series) {
		double sum = 0.0;
		double squareSum = 0.0;
		for (int i = 0; i < windowSize; i++) {
			sum += series[i];
			squareSum += series[i] * series[i];
		}

		double mean = sum / series.length;
		double variance = squareSum / series.length - mean * mean;
		return variance > 0 ? Math.sqrt(variance) : 1.0;
	}
	protected double[] DFT(double[] series) {
		int n = series.length;
		if(wordLength%2==1){
			wordLength++;
		}
		int outputLength = wordLength / 2;
		int start = (norm ? 1 : 0);
		double normalisingFactor = inverseSqrtWindowSize / stdDev(series);

		double[] dft = new double[outputLength * 2];

		for (int k = start; k < start + outputLength; k++) { // For each output
																// element
			float sumreal = 0;
			float sumimag = 0;
			for (int t = 0; t < n; t++) { // For each input element
				sumreal += series[t] * Math.cos(2 * Math.PI * t * k / n);
				sumimag += -series[t] * Math.sin(2 * Math.PI * t * k / n);
			}
			dft[(k - start) * 2] = sumreal * normalisingFactor;
			dft[(k - start) * 2 + 1] = sumimag * normalisingFactor;
		}
		return dft;
	}

	private double[] DFTunnormed(double[] series) {
		int n = series.length;
		int outputLength = wordLength / 2;
		int start = (norm ? 1 : 0);
		double[] dft = new double[outputLength * 2];
		double twoPi = 2 * Math.PI / n;

		for (int k = start; k < start + outputLength; k++) { // For each output
																// element
			float sumreal = 0;
			float sumimag = 0;
			for (int t = 0; t < n; t++) { // For each input element
				sumreal += series[t] * Math.cos(twoPi * t * k);
				sumimag += -series[t] * Math.sin(twoPi * t * k);
			}
			dft[(k - start) * 2] = sumreal;
			dft[(k - start) * 2 + 1] = sumimag;
		}
		return dft;
	}

	private double[] normalizeDFT(double[] dft, double std) {
		double normalisingFactor = (std > 0 ? 1.0 / std : 1.0) * inverseSqrtWindowSize;
		for (int i = 0; i < dft.length; i++) {
			dft[i] *= normalisingFactor;
		}
		return dft;
	}

	private double[][] performMFT(double[] series) {
		// ignore DC value?
		int startOffset = norm ? 2 : 0;
		int l = wordLength;
		l = l + l % 2; // make it even
		double[] phis = new double[l];
		for (int u = 0; u < phis.length; u += 2) {
			double uHalve = -(u + startOffset) / 2;
			phis[u] = realephi(uHalve, windowSize);
			phis[u + 1] = complexephi(uHalve, windowSize);
		}
		// means and stddev for each sliding window
		int end = Math.max(1, series.length - windowSize + 1);
		double[] means = new double[end];
		double[] stds = new double[end];
		calcIncreamentalMeanStddev(windowSize, series, means, stds);
		// holds the DFT of each sliding window
		double[][] transformed = new double[end][];
		double[] mftData = null;
		for (int t = 0; t < end; t++) {
			// use the MFT
			if (t > 0) {
				for (int k = 0; k < l; k += 2) {
					double real1 = (mftData[k] + series[t + windowSize - 1] - series[t - 1]);
					double imag1 = (mftData[k + 1]);
					double real = complexMulReal(real1, imag1, phis[k], phis[k + 1]);
					double imag = complexMulImag(real1, imag1, phis[k], phis[k + 1]);
					mftData[k] = real;
					mftData[k + 1] = imag;
				}
			} // use the DFT for the first offset
			else {
				mftData = Arrays.copyOf(series, windowSize);
				mftData = DFTunnormed(mftData);
			}
			// normalization for lower bounding
			transformed[t] = normalizeDFT(Arrays.copyOf(mftData, l), stds[t]);
		}
		return transformed;
	}

	private void calcIncreamentalMeanStddev(int windowLength, double[] series, double[] means, double[] stds) {
		double sum = 0;
		double squareSum = 0;
		// it is faster to multiply than to divide
		double rWindowLength = 1.0 / (double) windowLength;
		double[] tsData = series;
		for (int ww = 0; ww < windowLength; ww++) {
			sum += tsData[ww];
			squareSum += tsData[ww] * tsData[ww];
		}
		means[0] = sum * rWindowLength;
		double buf = squareSum * rWindowLength - means[0] * means[0];
		stds[0] = buf > 0 ? Math.sqrt(buf) : 0;
		for (int w = 1, end = tsData.length - windowLength + 1; w < end; w++) {
			sum += tsData[w + windowLength - 1] - tsData[w - 1];
			means[w] = sum * rWindowLength;
			squareSum += tsData[w + windowLength - 1] * tsData[w + windowLength - 1] - tsData[w - 1] * tsData[w - 1];
			buf = squareSum * rWindowLength - means[w] * means[w];
			stds[w] = buf > 0 ? Math.sqrt(buf) : 0;
		}
	}

	private static double complexMulReal(double r1, double im1, double r2, double im2) {
		return r1 * r2 - im1 * im2;
	}

	private static double complexMulImag(double r1, double im1, double r2, double im2) {
		return r1 * im2 + r2 * im1;
	}

	private static double realephi(double u, double M) {
		return Math.cos(2 * Math.PI * u / M);
	}

	private static double complexephi(double u, double M) {
		return -Math.sin(2 * Math.PI * u / M);
	}

	protected double[][] disjointWindows(double[] data) {
		int amount = (int) Math.ceil(data.length / (double) windowSize);
		double[][] subSequences = new double[amount][windowSize];

		for (int win = 0; win < amount; ++win) {
			int offset = Math.min(win * windowSize, data.length - windowSize);
			System.arraycopy(data, offset, subSequences[win], 0, windowSize);
		}

		return subSequences;
	}	
	
	
	protected double[][] MCB(ArrayList<ArrayList<Double>> data) {
		double[][][] dfts = new double[data.size()][][];

		int sample = 0;
		for (int i=0;i<data.size();i++) {
			dfts[sample++] = performDFT(disjointWindows(toArrayNoClass(data.get(i))));
		}

		int numInsts = dfts.length;
		int numWindowsPerInst = dfts[0].length;
		int totalNumWindows = numInsts * numWindowsPerInst;

		breakpoints = new double[wordLength][alphabetSize];

		for (int letter = 0; letter < wordLength; ++letter) { 
			double[] column = new double[totalNumWindows];
			for (int inst = 0; inst < numInsts; ++inst)
				for (int window = 0; window < numWindowsPerInst; ++window) {
					column[(inst * numWindowsPerInst) + window] = Math.round(dfts[inst][window][letter] * 100.0) / 100.0;
				}
			Arrays.sort(column);

			double binIndex = 0;
			double targetBinDepth = (double) totalNumWindows / (double) alphabetSize;

			for (int bp = 0; bp < alphabetSize - 1; ++bp) {
				binIndex += targetBinDepth;
				breakpoints[letter][bp] = column[(int) binIndex];
			}

			breakpoints[letter][alphabetSize - 1] = Double.MAX_VALUE; 
		}

		return breakpoints;
	}
		
	protected BitWord createWord(double[] dft) {
		BitWord word = new BitWord(wordLength);
		for (int l = 0; l < wordLength; ++l) {// for each letter
			for (int bp = 0; bp < alphabetSize; ++bp) {// run through
														// breakpoints until
														// right one found
				if (dft[l] <= breakpoints[l][bp]) {
					word.push(bp); // add corresponding letter to word
					break;
				}
			}
		}

		return word;
	}
	protected static double[] toArrayNoClass(ArrayList<Double> inst) {
		double[] data = new double[inst.size()];

		for (int i = 0; i < inst.size(); i++){
			data[i]=inst.get(i);
		}
		return data;
	}
	
	protected BitWord[] createSFAwords(ArrayList<Double> inst) {
		double[][] dfts2 = performMFT(toArrayNoClass(inst)); // approximation
		BitWord[] words2 = new BitWord[dfts2.length];
		for (int window = 0; window < dfts2.length; ++window)
			words2[window] = createWord(dfts2[window]);// discretisation

		return words2;
	}


	
	
	public void getSFAWord(ArrayList<ArrayList<Double>> data) {	
		breakpoints = MCB(data);
		SFAwords = new BitWord[data.size()][];
		for (int inst = 0; inst <data.size(); ++inst) {
			SFAwords[inst] = createSFAwords(data.get(inst));
		}
	}
	
}
